"""
Inferencia asíncrona reutilizando componentes del serving.

- AsyncInferenceEngine: motor asíncrono con límite de concurrencia.
- prepare_test_sample: construye payloads estilo API desde X_test/y_test.
- run_async_inference: orquesta inferencia batch + métricas agregadas.

Los esquemas de datos (resultados y resúmenes) viven en `src/utils/inference_models.py`.
"""

import asyncio
import time
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from config.settings import get_settings
from src.serving.registry import load_model_bundle, ModelBundle
from src.serving.service import ModelService
from src.serving.transforms import align_and_cast_dataframe, payload_to_dataframe
from src.serving.schemas import PredictRecord, PredictPayload

from src.utils.inference_models import (
    InferenceResult as InferenceResultModel,
    ConfidenceStats,
    BatchSummary,
    TestSampleMetadata,
)

settings = get_settings()
logger = settings.logger


class AsyncInferenceEngine:
    """
    Motor de inferencia asíncrona basado en componentes del serving.

    Args:
        tracking_uri: URI de tracking de MLflow.
        model_name: Nombre del modelo registrado.

    Raises:
        RuntimeError: Si no se puede cargar el bundle o el servicio del modelo.
    """

    def __init__(self, tracking_uri: str, model_name: str = "digital_orders_xgboost"):
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.model_service: ModelService | None = None
        self.bundle: ModelBundle | None = None

    async def initialize(self) -> None:
        """
        Inicializa MLflow, carga el bundle y crea el servicio de modelo.

        Raises:
            RuntimeError: Si falla la carga del modelo o la inicialización del servicio.
        """
        logger.info("🔄 Inicializando motor de inferencia...")
        mlflow.set_tracking_uri(self.tracking_uri)

        try:
            self.bundle = load_model_bundle(
                model_name=self.model_name,
                preferred_stage=None,
                tracking_uri=self.tracking_uri,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"❌ Error cargando bundle del modelo: {e}")
            raise RuntimeError("No fue posible cargar el bundle del modelo.") from e

        try:
            self.model_service = ModelService(self.bundle)
        except Exception as e:  # noqa: BLE001
            logger.error(f"❌ Error inicializando ModelService: {e}")
            raise RuntimeError("No fue posible inicializar el servicio del modelo.") from e

        logger.success(f"✅ Modelo cargado: {self.bundle.uri}")
        logger.info(f"🏷️  Clases: {self.model_service.class_names}")
        logger.info(f"📊 Features categóricas: {len(self.model_service.categorical_cols)}")
        logger.info(f"🔢 Columnas de entrada: {self.model_service.input_cols}")

    # -------------------------
    # Helpers privados
    # -------------------------
    def _create_payload_from_dict(self, data: dict[str, Any]) -> PredictPayload:
        """
        Convierte un dict en PredictPayload (valida con los schemas de la API).

        Args:
            data: Registro con las claves esperadas por la API.

        Returns:
            PredictPayload: Payload validado con un único registro.

        Raises:
            ValueError: Si la validación del schema falla.
        """
        try:
            record = PredictRecord(**data)
            return PredictPayload(records=[record])
        except Exception as e:
            logger.error(f"❌ Error creando payload: {e}")
            raise ValueError(f"Error validando datos de entrada: {e}") from e

    def _prepare_dataframe_with_transforms(self, data: dict[str, Any]) -> pd.DataFrame:
        """
        Aplica el pipeline de transforms para alinear y castear features.

        Args:
            data: Registro de entrada validado.

        Returns:
            pd.DataFrame: DataFrame alineado con el modelo (dtypes correctos).

        Raises:
            RuntimeError: Si el servicio de modelo no está inicializado.
        """
        if self.model_service is None:
            raise RuntimeError("ModelService no inicializado. Llama a initialize() primero.")

        payload = self._create_payload_from_dict(data)
        df = payload_to_dataframe(payload)
        df_aligned = align_and_cast_dataframe(
            df=df,
            input_cols=self.model_service.input_cols,
            categorical_cols=self.model_service.categorical_cols,
        )

        logger.debug(f"🔧 DataFrame preparado: {df_aligned.columns.tolist()}")
        logger.debug(
            "   Categorical count: %d",
            sum(df_aligned[c].dtype.name == "category" for c in df_aligned.columns),
        )
        return df_aligned

    # -------------------------
    # API pública
    # -------------------------
    async def predict_single(self, index: int, data: dict[str, Any]) -> InferenceResultModel:
        """
        Realiza la predicción para un único registro.

        Args:
            index: Índice lógico del registro (para trazabilidad).
            data: Datos del registro (dict).

        Returns:
            InferenceResultModel: Resultado tipado de inferencia.

        Raises:
            RuntimeError: Si el servicio no está inicializado.
            ValueError: Si el preprocesamiento o la predicción fallan.
        """
        if self.model_service is None:
            raise RuntimeError("ModelService no inicializado. Llama a initialize() primero.")

        start_time = time.time()

        try:
            df = self._prepare_dataframe_with_transforms(data)

            loop = asyncio.get_event_loop()
            probabilities, _processing_time_model = await loop.run_in_executor(
                None, self.model_service.predict_proba, df
            )

            prediction_results = self.model_service.predict_labels(probabilities)
            pred = prediction_results[0]  # un solo registro

            total_time = (time.time() - start_time) * 1000.0
            model_version = self.bundle.uri.split("/")[-1] if self.bundle else None

            return InferenceResultModel(
                index=index,
                input_data=data,
                probabilities=list(map(float, pred.probabilities)),
                predicted_class=str(pred.predicted_class),
                confidence=float(pred.confidence),
                processing_time_ms=float(total_time),
                model_version=model_version,
            )

        except Exception as e:  # noqa: BLE001
            logger.error(f"❌ Error en predicción {index}: {e}")
            raise

    async def predict_batch(self, data_batch: list[dict[str, Any]], max_concurrent: int = 5) -> list[InferenceResultModel]:
        """
        Realiza predicción batch con control de concurrencia.

        Args:
            data_batch: Lista de registros (dict) a predecir.
            max_concurrent: Máximo de tareas concurrentes.

        Returns:
            list[InferenceResultModel]: Resultados exitosos en el mismo orden relativo.

        Raises:
            ValueError: Si `max_concurrent` < 1.
        """
        if max_concurrent < 1:
            raise ValueError("max_concurrent debe ser ≥ 1.")

        logger.info(f"🚀 Iniciando inferencia batch: {len(data_batch)} registros")
        logger.info(f"🔄 Concurrencia máxima: {max_concurrent}")

        start_time = time.time()
        semaphore = asyncio.Semaphore(max_concurrent)

        async def predict_with_semaphore(i: int, row: dict[str, Any]) -> InferenceResultModel:
            async with semaphore:
                return await self.predict_single(i, row)

        tasks = [predict_with_semaphore(i, row) for i, row in enumerate(data_batch)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful: list[InferenceResultModel] = []
        errors: list[str] = []

        for i, r in enumerate(results):
            if isinstance(r, Exception):
                msg = f"Error en registro {i}: {r}"
                logger.error(f"❌ {msg}")
                errors.append(msg)
            else:
                successful.append(r)

        total_time = (time.time() - start_time) * 1000.0
        success_rate = (len(successful) / len(data_batch) * 100.0) if data_batch else 0.0
        avg_time = (
            sum(res.processing_time_ms for res in successful) / len(successful)
            if successful else 0.0
        )

        logger.success("✅ Inferencia batch completada:")
        logger.info(f"   📊 Exitosas: {len(successful)}/{len(data_batch)} ({success_rate:.1f}%)")
        logger.info(f"   ⚡ Tiempo total: {total_time:.2f}ms")
        logger.info(f"   📈 Promedio por predicción: {avg_time:.2f}ms")

        if errors:
            logger.warning(f"⚠️  Errores encontrados: {len(errors)}")

        return successful


# -------------------------
# Helpers para pruebas offline
# -------------------------
def prepare_test_sample(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_samples: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Construye una muestra de test lista para pasar al engine.

    Selecciona aleatoriamente `n_samples` filas de X_test/y_test y arma registros
    con las mismas claves que espera el endpoint de predicción.

    Args:
        X_test: Features de prueba alineadas con el modelo.
        y_test: Target de prueba.
        n_samples: Cantidad de ejemplos a muestrear.

    Returns:
        tuple[list[dict], dict]: (test_data, metadata_dict)
            - test_data: lista de registros estilo API.
            - metadata_dict: metadatos serializables (para logging/MLflow).

    Raises:
        ValueError: Si X_test o y_test están vacíos.
    """
    if X_test.empty or y_test.empty:
        raise ValueError("X_test e y_test no pueden estar vacíos.")

    n = min(n_samples, len(X_test))
    sample_indices = np.random.choice(len(X_test), size=n, replace=False)
    X_sample = X_test.iloc[sample_indices].copy()
    y_sample = y_test.iloc[sample_indices].copy()

    test_data: list[dict[str, Any]] = []
    for _, row in X_sample.iterrows():
        test_data.append(
            {
                "pais_cd": str(row["pais_cd"]),
                "tipo_cliente_cd": str(row["tipo_cliente_cd"]),
                "madurez_digital_cd": str(row["madurez_digital_cd"]),
                "estrellas_txt": int(row["estrellas_txt"]),
                "frecuencia_visitas_cd": str(row["frecuencia_visitas_cd"]),
                "cajas_fisicas": float(row["cajas_fisicas"]),
                "fecha_pedido_dt_day_of_week": int(row["fecha_pedido_dt_day_of_week"]),
                "fecha_pedido_dt_quarter": int(row["fecha_pedido_dt_quarter"]),
            }
        )

    metadata = TestSampleMetadata(
        sample_size=len(test_data),
        ground_truth_distribution={k: int(v) for k, v in y_sample.value_counts().to_dict().items()},
        sample_indices=list(map(int, sample_indices.tolist())),
        columns_order=list(map(str, X_sample.columns.tolist())),
    )

    logger.info(f"📋 Muestra de test preparada: {len(test_data)} registros")
    logger.info(f"🎯 Distribución real: {metadata.ground_truth_distribution}")
    logger.info(f"🔢 Orden de columnas: {metadata.columns_order}")

    return test_data, metadata.model_dump()


async def run_async_inference(
    test_data: list[dict[str, Any]],
    tracking_uri: str,
    model_name: str = "digital_orders_xgboost",
    max_concurrent: int = 3,
) -> tuple[list[InferenceResultModel], dict[str, Any]]:
    """
    Ejecuta inferencia asíncrona y devuelve resultados + resumen.

    Args:
        test_data: Registros estilo API a inferir.
        tracking_uri: URI de MLflow.
        model_name: Nombre del modelo (registro).
        max_concurrent: Máximo de tareas concurrentes.

    Returns:
        tuple: (results, summary_dict)
            - results: lista de InferenceResultModel.
            - summary_dict: BatchSummary serializado (dict).
    """
    start_time = time.time()

    engine = AsyncInferenceEngine(tracking_uri, model_name)
    await engine.initialize()

    results = await engine.predict_batch(test_data, max_concurrent=max_concurrent)

    total_time = (time.time() - start_time) * 1000.0

    if results:
        avg_conf = float(sum(r.confidence for r in results) / len(results))
        pred_dist: dict[str, int] = {}
        for r in results:
            pred_dist[r.predicted_class] = pred_dist.get(r.predicted_class, 0) + 1

        conf_stats = ConfidenceStats(
            min_confidence=float(min(r.confidence for r in results)),
            max_confidence=float(max(r.confidence for r in results)),
            avg_confidence=avg_conf,
            std_confidence=float(np.std([r.confidence for r in results])),
        )
    else:
        pred_dist = {}
        conf_stats = None

    summary = BatchSummary(
        total_samples=len(test_data),
        successful_predictions=len(results),
        success_rate=(len(results) / len(test_data) * 100.0) if test_data else 0.0,
        total_time_ms=float(total_time),
        avg_time_per_prediction_ms=(
            float(sum(r.processing_time_ms for r in results) / len(results))
            if results else 0.0
        ),
        prediction_distribution=pred_dist,
        confidence_stats=conf_stats,
        model_version=(results[0].model_version if results else None),
    )
    
    avg = summary.confidence_stats.avg_confidence if summary.confidence_stats else 0.0

    logger.success("🎉 Inferencia asíncrona completada:")
    logger.info(f"   📊 Success rate: {summary.success_rate:.1f}%")
    logger.info(f"   🎯 Predicciones: {summary.prediction_distribution}")
    logger.info(f"   💪 Confianza promedio: {avg:.3f}")

    return results, summary.model_dump()
