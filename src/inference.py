import asyncio
import pandas as pd
import numpy as np
import time
import mlflow
from config.settings import get_settings
from src.serving.registry import load_model_bundle, ModelBundle
from src.serving.transforms import payload_to_dataframe, align_and_cast_dataframe
from src.serving.service import ModelService
from src.serving.schemas import PredictRecord, PredictPayload
from dataclasses import dataclass

settings = get_settings()
logger = settings.logger

@dataclass
class InferenceResult:
    """Resultado de inferencia individual extendido para batch processing"""
    index: int
    input_data: dict
    probabilities: list[float]
    predicted_class: str
    confidence: float
    processing_time_ms: float
    model_version: str | None = None

class AsyncInferenceEngine:
    """Motor de inferencia asíncrona reutilizando componentes del serving"""
    
    def __init__(self, tracking_uri: str, model_name: str = "digital_orders_xgboost"):
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.model_service: ModelService | None = None
        self.bundle: ModelBundle | None = None
        
    async def initialize(self) -> None:
        """Inicializa el motor usando los mismos componentes que la API"""
        logger.info("🔄 Inicializando motor de inferencia...")
        
        mlflow.set_tracking_uri(self.tracking_uri)
        
        try:
            # Reutilizar registry.py para cargar modelo
            self.bundle = load_model_bundle(
                model_name=self.model_name,
                preferred_stage=None,
                tracking_uri=self.tracking_uri
            )
            
            # Reutilizar service.py
            self.model_service = ModelService(self.bundle)
            
            logger.success(f"✅ Modelo cargado: {self.bundle.uri}")
            logger.info(f"🏷️  Clases: {self.model_service.class_names}")
            logger.info(f"📊 Features categóricas: {len(self.model_service.categorical_cols)}")
            logger.info(f"🔢 Columnas de entrada: {self.model_service.input_cols}")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise
    
    def _create_payload_from_dict(self, data: dict) -> PredictPayload:
        """Convierte dict a PredictPayload usando schemas existentes"""
        try:
            # Validar y crear record usando schema
            record = PredictRecord(**data)
            # Crear payload usando el schema de la API
            payload = PredictPayload(records=[record])
            return payload
        except Exception as e:
            logger.error(f"❌ Error creando payload: {e}")
            raise ValueError(f"Error validando datos: {e}")
    
    def _prepare_dataframe_with_transforms(self, data: dict) -> pd.DataFrame:
        """Reutiliza transforms.py para preparar DataFrame"""
        # Usar pipeline completo de transforms
        payload = self._create_payload_from_dict(data)
        
        # Reutilizar transforms.py
        df = payload_to_dataframe(payload)
        
        # Alinear con modelo usando transforms existente
        df_aligned = align_and_cast_dataframe(
            df=df,
            input_cols=self.model_service.input_cols,
            categorical_cols=self.model_service.categorical_cols
        )
        
        logger.debug(f"🔧 DataFrame preparado: {df_aligned.columns.tolist()}")
        logger.debug(f"   Categorical count: {len([col for col in df_aligned.columns if df_aligned[col].dtype.name == 'category'])}")
        
        return df_aligned
    
    async def predict_single(self, index: int, data: dict) -> InferenceResult:
        """Predicción individual usando service.py"""
        start_time = time.time()
        
        try:
            # Preparar datos usando pipeline existente
            df = self._prepare_dataframe_with_transforms(data)
            
            # Predicción usando ModelService (thread pool para async)
            loop = asyncio.get_event_loop()
            probabilities, processing_time_model = await loop.run_in_executor(
                None, 
                self.model_service.predict_proba, 
                df
            )
            
            # Usar ModelService para convertir a resultados estructurados
            prediction_results = self.model_service.predict_labels(probabilities)
            result = prediction_results[0]  # Solo un registro
            
            # Tiempo total (incluyendo preprocessing)
            total_time = (time.time() - start_time) * 1000
            
            return InferenceResult(
                index=index,
                input_data=data,
                probabilities=result.probabilities,
                predicted_class=result.predicted_class,
                confidence=result.confidence,
                processing_time_ms=total_time,
                model_version=self.bundle.uri.split("/")[-1] if self.bundle else None
            )
            
        except Exception as e:
            logger.error(f"❌ Error en predicción {index}: {e}")
            raise
    
    async def predict_batch(self, data_batch: list[dict], max_concurrent: int = 5) -> list[InferenceResult]:
        """Predicción batch con concurrencia controlada"""
        logger.info(f"🚀 Iniciando inferencia batch: {len(data_batch)} registros")
        logger.info(f"🔄 Concurrencia máxima: {max_concurrent}")
        
        start_time = time.time()
        
        # Semáforo para controlar concurrencia
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def predict_with_semaphore(index: int, data: dict) -> InferenceResult:
            async with semaphore:
                return await self.predict_single(index, data)
        
        # Ejecutar todas las predicciones concurrentemente
        tasks = [
            predict_with_semaphore(i, data) 
            for i, data in enumerate(data_batch)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Procesar resultados
        successful_results = []
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = f"Error en registro {i}: {result}"
                logger.error(f"❌ {error_msg}")
                errors.append(error_msg)
            else:
                successful_results.append(result)
        
        total_time = (time.time() - start_time) * 1000
        success_rate = len(successful_results) / len(data_batch) * 100
        avg_time_per_prediction = sum(r.processing_time_ms for r in successful_results) / len(successful_results) if successful_results else 0
        
        logger.success(f"✅ Inferencia batch completada:")
        logger.info(f"   📊 Exitosas: {len(successful_results)}/{len(data_batch)} ({success_rate:.1f}%)")
        logger.info(f"   ⚡ Tiempo total: {total_time:.2f}ms")
        logger.info(f"   📈 Promedio por predicción: {avg_time_per_prediction:.2f}ms")
        
        if errors:
            logger.warning(f"⚠️  Errores encontrados: {len(errors)}")
        
        return successful_results

def prepare_test_sample(X_test: pd.DataFrame, y_test: pd.Series, n_samples: int = 10) -> tuple[list[dict], dict]:
    """Prepara muestra de test con metadata adicional"""
    
    # Tomar muestra aleatoria
    sample_indices = np.random.choice(len(X_test), size=min(n_samples, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_indices].copy()
    y_sample = y_test.iloc[sample_indices].copy()
    
    # Convertir a formato API (mantiene orden original de X_test)
    test_data = []
    for idx, (_, row) in enumerate(X_sample.iterrows()):
        data_point = {
            "pais_cd": str(row['pais_cd']),
            "tipo_cliente_cd": str(row['tipo_cliente_cd']),
            "madurez_digital_cd": str(row['madurez_digital_cd']),
            "estrellas_txt": int(row['estrellas_txt']),
            "frecuencia_visitas_cd": str(row['frecuencia_visitas_cd']),
            "cajas_fisicas": float(row['cajas_fisicas']),
            "fecha_pedido_dt_day_of_week": int(row['fecha_pedido_dt_day_of_week']),
            "fecha_pedido_dt_quarter": int(row['fecha_pedido_dt_quarter'])
        }
        test_data.append(data_point)
    
    # Metadata del sample
    ground_truth_distribution = y_sample.value_counts().to_dict()
    
    metadata = {
        "sample_size": len(test_data),
        "ground_truth_distribution": ground_truth_distribution,
        "sample_indices": sample_indices.tolist(),
        "columns_order": X_sample.columns.tolist()
    }
    
    logger.info(f"📋 Muestra de test preparada: {len(test_data)} registros")
    logger.info(f"🎯 Distribución real: {ground_truth_distribution}")
    logger.info(f"🔢 Orden de columnas: {metadata['columns_order']}")
    
    return test_data, metadata

async def run_async_inference(
    test_data: list[dict], 
    tracking_uri: str,
    model_name: str = "digital_orders_xgboost",
    max_concurrent: int = 3
) -> tuple[list[InferenceResult], dict]:
    """Ejecuta inferencia asíncrona con métricas detalladas"""
    
    start_time = time.time()
    
    # Inicializar motor
    engine = AsyncInferenceEngine(tracking_uri, model_name)
    await engine.initialize()
    
    # Ejecutar inferencias
    results = await engine.predict_batch(test_data, max_concurrent=max_concurrent)
    
    # Calcular métricas agregadas
    total_time = (time.time() - start_time) * 1000
    
    if results:
        avg_confidence = sum(r.confidence for r in results) / len(results)
        prediction_distribution = {}
        for result in results:
            pred_class = result.predicted_class
            prediction_distribution[pred_class] = prediction_distribution.get(pred_class, 0) + 1
        
        confidence_stats = {
            "min_confidence": min(r.confidence for r in results),
            "max_confidence": max(r.confidence for r in results),
            "avg_confidence": avg_confidence,
            "std_confidence": np.std([r.confidence for r in results])
        }
    else:
        prediction_distribution = {}
        confidence_stats = {}
    
    summary_metrics = {
        "total_samples": len(test_data),
        "successful_predictions": len(results),
        "success_rate": len(results) / len(test_data) * 100 if test_data else 0,
        "total_time_ms": total_time,
        "avg_time_per_prediction_ms": sum(r.processing_time_ms for r in results) / len(results) if results else 0,
        "prediction_distribution": prediction_distribution,
        "confidence_stats": confidence_stats,
        "model_version": results[0].model_version if results else None
    }
    
    logger.success(f"🎉 Inferencia asíncrona completada:")
    logger.info(f"   📊 Success rate: {summary_metrics['success_rate']:.1f}%")
    logger.info(f"   🎯 Predicciones: {prediction_distribution}")
    logger.info(f"   💪 Confianza promedio: {confidence_stats.get('avg_confidence', 0):.3f}")
    
    return results, summary_metrics
