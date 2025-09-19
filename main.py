"""
Entrypoint principal:
- Ejecuta ELT
- Split + selección de features
- Asegura modelo en MLflow (entrena si no existe)
- Corre inferencia asíncrona y loguea métricas/artefactos
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Callable

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

from config.settings import get_settings
from src.elt import ELTPipeline
from src.utils.schemas import ELTConfig
from src.split import split_train_val_test
from src.selection import feature_selection
from src.utils.mlflow_utils import ensure_experiment
from src.inference import run_async_inference, prepare_test_sample
from src.training import train_models


def ensure_model_available(
    tracking_uri: str,
    model_name: str,
    train_fn: Callable[[], None],    
) -> None:
    """
    Garantiza que exista al menos una versión registrada en el Model Registry.

    Si el modelo no existe o no tiene versiones, ejecuta `train_fn()` y revalida.

    Args:
        tracking_uri: URI del servidor de tracking de MLflow.
        model_name: Nombre del modelo en el Model Registry.
        train_fn: Función (sin argumentos) que entrena y registra el modelo.

    Raises:
        RuntimeError: Si tras entrenar no aparece una versión registrada.
        RestException: Si ocurren errores de cliente MLflow distintos a "no existe".
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    def _has_versions() -> bool:
        try:
            client.get_registered_model(model_name)
        except RestException as e:
            # Si no existe el Registered Model, retornamos False; otros errores se propagan.
            if getattr(e, "error_code", "") == "RESOURCE_DOES_NOT_EXIST":
                return False
            raise
        # Existe el Registered Model: verificar versiones
        versions = list(client.search_model_versions(f"name='{model_name}'"))
        return len(versions) > 0

    if _has_versions():
        return

    # No existe o sin versiones → entrenar
    train_fn()

    # Revalidar
    if not _has_versions():
        raise RuntimeError(f"El modelo '{model_name}' no quedó registrado tras el entrenamiento.")


def _align_dict_like(template: dict[str, int], source: dict[str, int]) -> dict[str, int]:
    """
    Devuelve `source` ordenado con las claves de `template`, dejando al final las sobrantes.

    Args:
        template: Diccionario cuyo orden de claves se usará como referencia.
        source: Diccionario a reordenar.

    Returns:
        dict[str, int]: `source` reordenado y con claves faltantes completadas en 0.
    """
    aligned: dict[str, int] = {k: source.get(k, 0) for k in template.keys()}
    for k, v in source.items():
        if k not in aligned:
            aligned[k] = v
    return aligned


async def main() -> None:
    """
    Orquestación principal del flujo ELT → split → selección → training (si falta) → inferencia.

    Raises:
        Exception: Propaga errores críticos de ELT, split/selection, entrenamiento o inferencia.
    """
    settings = get_settings()
    logger = settings.logger

    # ==============================
    # 0) MLflow: tracking + experimento
    # ==============================
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5555")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "digital_orders_new")
    model_name = "digital_orders_xgboost"

    mlflow.set_tracking_uri(tracking_uri)
    exp_id = ensure_experiment(experiment_name)

    logger.info(f"🧭 MLflow tracking_uri = {tracking_uri}")
    logger.info(f"🧪 MLflow experiment   = {experiment_name} (id={exp_id})")

    # ==============================
    # Run padre (inferencia)
    # ==============================
    with mlflow.start_run(run_name="async_inference_run"):
        mlflow.log_params({
            "codebase_entrypoint": "main.py",
            "mode": "inference",
            "target": "canal_pedido_cd",
            "async_inference": True,
        })

        # -------------------------------
        # 1) Procesar datos (ELT)
        # -------------------------------
        raw_root = Path("data/raw")
        output_root = Path("data/processed")

        config = ELTConfig(
            raw_root=raw_root,
            output_root=output_root,
            dtype_map={"estrellas_txt": "float64"},
        )

        try:
            elt_pipeline = ELTPipeline(config)
            df_processed = elt_pipeline.run()
            logger.success(f"🎯 Pipeline ELT ejecutada: {len(df_processed):,} registros")
        except Exception as e:
            logger.error(f"❌ Error en ELT: {e}")
            raise

        # -------------------------------
        # 2) Split (train/val/test)
        # -------------------------------
        target = "canal_pedido_cd"
        seed = 42
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
                data=df_processed, target=target, seed=seed
            )
            logger.success(f"📊 Split realizada: Test={X_test.shape}")
        except Exception as e:
            logger.error(f"❌ Error en split: {e}")
            raise

        # -------------------------------
        # 3) Feature Selection (alinear columnas)
        # -------------------------------
        try:
            X_train_sel, X_val_sel, X_test_sel = feature_selection(
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
            )
            logger.success(f"✨ Features seleccionadas: {X_test_sel.shape}")
        except Exception as e:
            logger.error(f"❌ Error en feature selection: {e}")
            raise

        # -------------------------------
        # 3.5) Asegurar modelo (entrenar si no existe)
        # -------------------------------
        def _train_if_needed() -> None:
            """Entrena y registra el modelo si no existe en el registry."""
            try:
                # `train_models` inicia un run anidado, loguea y registra el modelo.
                model, val_logloss, model_str = train_models(
                    X_train=X_train_sel,
                    X_val=X_val_sel,
                    X_test=X_test_sel,
                    y_train=y_train,
                    y_val=y_val,
                    y_test=y_test,
                    use_balanced_weights=True,
                    use_gpu=True,
                    early_stopping_rounds=50,
                )
                logger.success(f"✅ Modelo entrenado ({model_str}) | val_logloss={val_logloss:.5f}")
            except Exception as e:
                logger.error(f"❌ Error durante el entrenamiento: {e}")
                raise

        try:
            ensure_model_available(tracking_uri=tracking_uri, model_name=model_name, train_fn=_train_if_needed)
            logger.info(f"🧩 Modelo disponible en registry: {model_name}")
        except Exception as e:
            logger.error(f"❌ No fue posible asegurar el modelo '{model_name}': {e}")
            raise

        # -------------------------------
        # 4) Inferencia asíncrona
        # -------------------------------
        try:
            logger.info("🚀 Iniciando inferencia asíncrona...")

            # Preparar muestra de test
            test_sample, sample_metadata = prepare_test_sample(X_test_sel, y_test, n_samples=50)

            # Log de metadata del sample (guardar distribución real como artefacto JSON)
            mlflow.log_param("test_sample_size", sample_metadata["sample_size"])
            mlflow.log_text(
                json.dumps(sample_metadata["ground_truth_distribution"], indent=2, ensure_ascii=False),
                "artifacts/test_ground_truth_distribution.json",
            )

            # Ejecutar inferencia
            results, metrics = await run_async_inference(
                test_data=test_sample,
                tracking_uri=tracking_uri,
                model_name=model_name,
                max_concurrent=10,
            )

            if results:
                total_time = metrics["total_time_ms"]
                avg_time = metrics["avg_time_per_prediction_ms"]

                # Métricas agregadas de inferencia
                mlflow.log_metrics({
                    "inference_samples": metrics["successful_predictions"],
                    "inference_total_time_ms": total_time,
                    "inference_avg_time_ms": avg_time,
                    "inference_success_rate": metrics["success_rate"] / 100.0,
                })

                # Conteo por clase
                pred_distribution: dict[str, int] = metrics["prediction_distribution"]
                for class_name, count in pred_distribution.items():
                    mlflow.log_metric(f"pred_count_{class_name}", count)

                # Métricas de confianza
                confidence_stats: dict[str, float] = metrics.get("confidence_stats", {})
                if confidence_stats:
                    mlflow.log_metrics({
                        "avg_confidence": confidence_stats["avg_confidence"],
                        "min_confidence": confidence_stats["min_confidence"],
                        "max_confidence": confidence_stats["max_confidence"],
                        "std_confidence": confidence_stats["std_confidence"],
                    })

                # Guardar resultados detallados
                results_summary = [{
                    "index": r.index,
                    "predicted_class": r.predicted_class,
                    "confidence": r.confidence,
                    "probabilities": r.probabilities,
                    "processing_time_ms": r.processing_time_ms,
                    "model_version": r.model_version,
                } for r in results]

                with open("inference_results.json", "w", encoding="utf-8") as f:
                    json.dump(results_summary, f, indent=2, ensure_ascii=False)
                mlflow.log_artifact("inference_results.json")

                # Alinear ground truth al orden de las predicciones para logging legible
                ground_truth_dist: dict[str, int] = sample_metadata["ground_truth_distribution"]
                gt_aligned = _align_dict_like(pred_distribution, ground_truth_dist)

                # Logs al usuario
                logger.success("🎉 Inferencia asíncrona completada:")
                logger.info(f"   📊 Muestras procesadas: {len(results)}")
                logger.info(f"   ⚡ Tiempo promedio: {avg_time:.2f}ms")
                logger.info(f"   🎯 Distribución predicciones: {pred_distribution}")
                logger.info(f"   🏆 Ground truth: {gt_aligned}")
                logger.info(f"   💪 Confianza promedio: {confidence_stats.get('avg_confidence', 0.0):.3f}")

                # Guardar distribuciones como artefacto JSON conjunto
                with open("inference_distributions.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {"predictions": pred_distribution, "ground_truth_aligned": gt_aligned},
                        f, indent=2, ensure_ascii=False
                    )
                mlflow.log_artifact("inference_distributions.json")

            else:
                logger.error("❌ No se pudieron procesar predicciones")
                mlflow.log_metrics({
                    "inference_samples": 0,
                    "inference_success_rate": 0.0,
                })

        except Exception as e:
            logger.error(f"❌ Error en inferencia: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(main())
