import os
import asyncio
from pathlib import Path
import mlflow
from config.settings import get_settings
from src.utils.elt import ELTConfig, ELTPipeline
from src.split import split_train_val_test
from src.selection import feature_selection
from src.utils.mlflow_utils import ensure_experiment
from src.inference import run_async_inference, prepare_test_sample

async def main() -> None:
    settings = get_settings()
    logger = settings.logger

    # ==============================
    # 0) MLflow: tracking + experimento
    # ==============================
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5555")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "digital_orders_new")

    mlflow.set_tracking_uri(tracking_uri)
    exp_id = ensure_experiment(experiment_name)

    logger.info(f"🧭 MLflow tracking_uri = {tracking_uri}")
    logger.info(f"🧪 MLflow experiment   = {experiment_name} (id={exp_id})")

    # ==============================
    # Run padre (inferencia)
    # ==============================
    with mlflow.start_run(run_name="async_inference_run") as run:
        mlflow.log_params({
            "codebase_entrypoint": "inference_main.py",
            "mode": "inference",
            "target": "canal_pedido_cd",
            "async_inference": True,
        })

        # -------------------------------
        # 1) Procesar datos (misma lógica)
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
        # 2) Split (solo para obtener test set)
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
        # 3) Feature Selection (solo para alinear test)
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
        # 4) INFERENCIA ASÍNCRONA (¡CORREGIDO!)
        # -------------------------------
        try:
            logger.info("🚀 Iniciando inferencia asíncrona...")
            
            # ✅ CORRECTO: Desempacar la tupla
            test_sample, sample_metadata = prepare_test_sample(X_test_sel, y_test, n_samples=50)
            
            # Log metadata del sample
            mlflow.log_params({
                "test_sample_size": sample_metadata["sample_size"],
                "test_ground_truth_dist": sample_metadata["ground_truth_distribution"]
            })
            
            # ✅ CORRECTO: Solo pasar test_sample (list[dict])
            results, metrics = await run_async_inference(
                test_data=test_sample,  # ✅ Solo la lista de datos
                tracking_uri=tracking_uri,
                model_name="digital_orders_xgboost",
                max_concurrent=10
            )
            
            # Procesar y loggear resultados
            if results:
                total_time = metrics["total_time_ms"]
                avg_time = metrics["avg_time_per_prediction_ms"]
                
                # Métricas de inferencia
                mlflow.log_metrics({
                    "inference_samples": metrics["successful_predictions"],
                    "inference_total_time_ms": total_time,
                    "inference_avg_time_ms": avg_time,
                    "inference_success_rate": metrics["success_rate"] / 100  # Como decimal
                })
                
                # Distribución de predicciones
                pred_distribution = metrics["prediction_distribution"]
                for class_name, count in pred_distribution.items():
                    mlflow.log_metric(f"pred_count_{class_name}", count)
                
                # Métricas de confianza
                confidence_stats = metrics.get("confidence_stats", {})
                if confidence_stats:
                    mlflow.log_metrics({
                        "avg_confidence": confidence_stats["avg_confidence"],
                        "min_confidence": confidence_stats["min_confidence"],
                        "max_confidence": confidence_stats["max_confidence"],
                        "std_confidence": confidence_stats["std_confidence"]
                    })
                
                # Guardar resultados detallados
                results_summary = []
                for result in results:
                    results_summary.append({
                        "index": result.index,
                        "predicted_class": result.predicted_class,
                        "confidence": result.confidence,
                        "probabilities": result.probabilities,
                        "processing_time_ms": result.processing_time_ms,
                        "model_version": result.model_version
                    })
                
                import json
                with open("inference_results.json", "w") as f:
                    json.dump(results_summary, f, indent=2)
                mlflow.log_artifact("inference_results.json")
                
                # Comparar con ground truth
                ground_truth_dist = sample_metadata["ground_truth_distribution"]
                
                logger.success(f"🎉 Inferencia asíncrona completada:")
                logger.info(f"   📊 Muestras procesadas: {len(results)}")
                logger.info(f"   ⚡ Tiempo promedio: {avg_time:.2f}ms")
                logger.info(f"   🎯 Distribución predicciones: {pred_distribution}")
                logger.info(f"   🏆 Ground truth: {ground_truth_dist}")
                logger.info(f"   💪 Confianza promedio: {confidence_stats.get('avg_confidence', 0):.3f}")
                
            else:
                logger.error("❌ No se pudieron procesar predicciones")
                mlflow.log_metrics({
                    "inference_samples": 0,
                    "inference_success_rate": 0.0
                })
                
        except Exception as e:
            logger.error(f"❌ Error en inferencia: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(main())
