import os
from pathlib import Path
import mlflow
from config.settings import get_settings
from src.utils.elt import ELTConfig, ELTPipeline
from src.split import split_train_val_test
from src.selection import feature_selection
from src.training import train_models
from src.utils.mlflow_utils import ensure_experiment


def main() -> None:
    settings = get_settings()
    logger = settings.logger

    # ==============================
    # 0) MLflow: tracking + experimento (idempotente)
    # ==============================
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5555")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "digital_orders_new")

    mlflow.set_tracking_uri(tracking_uri)
    exp_id = ensure_experiment(experiment_name)  # crea / restaura / activa

    logger.info(f"🧭 MLflow tracking_uri = {tracking_uri}")
    logger.info(f"🧪 MLflow experiment   = {experiment_name} (id={exp_id})")

    # ==============================
    # Run padre (toda la ejecución)
    # ==============================
    with mlflow.start_run(run_name="pipeline_run") as run:
        mlflow.log_params({
            "codebase_entrypoint": "main.py",
            "random_seed": 42,
            "target": "canal_pedido_cd",
        })

        # -------------------------------
        # 1) Procesar datos crudos → processed
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
            logger.success(
                f"🎯 Pipeline ELT ejecutada con éxito. Registros procesados: {len(df_processed):,}"
            )
            # Log de dataset
            mlflow.log_metric("processed_rows", len(df_processed))
            mlflow.log_metric("processed_cols", df_processed.shape[1])
            mlflow.log_text(
                "\n".join(map(str, df_processed.columns.tolist())),
                "artifacts/columns_processed.txt"
            )
        except Exception as e:
            logger.error(f"❌ Error al ejecutar pipeline ELT: {e}")
            raise

        # -------------------------------
        # 2) Split train/val/test
        # -------------------------------
        target = "canal_pedido_cd"
        seed = 42

        try:
            X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
                data=df_processed, target=target, seed=seed
            )
            logger.success(
                f"📊 División realizada: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}"
            )

            mlflow.log_metrics({
                "train_rows": X_train.shape[0],
                "val_rows":   X_val.shape[0],
                "test_rows":  X_test.shape[0],
                "n_features_before_fs": X_train.shape[1],
            })
        except Exception as e:
            logger.error(f"❌ Error en split de datos: {e}")
            raise

        # -------------------------------
        # 3) Feature Selection
        # -------------------------------
        try:
            X_train_sel, X_val_sel, X_test_sel = feature_selection(
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
            )
            logger.success(
                f"✨ Feature selection completa. "
                f"Train={X_train_sel.shape}, Val={X_val_sel.shape}, Test={X_test_sel.shape}"
            )

            mlflow.log_metric("n_features_after_fs", X_train_sel.shape[1])
            mlflow.log_text(
                "\n".join(map(str, X_train_sel.columns.tolist())),
                "artifacts/selected_features.txt"
            )
        except Exception as e:
            logger.error(f"❌ Error en feature selection: {e}")
            raise

        # -------------------------------
        # 4) Training con XGBoost (run anidado)
        # -------------------------------
        try:
            model, score, model_name = train_models(
                X_train_sel,
                X_val_sel,
                X_test_sel,
                y_train,
                y_val,
                y_test,
                use_balanced_weights=True,
                use_gpu=True,   # GPU con fallback a CPU
            )
            logger.success(
                f"🤖 Entrenamiento completado con {model_name}. "
                f"LogLoss de validación: {score:.4f}"
            )
            mlflow.log_metric("val_logloss_final", float(score))
        except Exception as e:
            logger.error(f"❌ Error en entrenamiento: {e}")
            raise


if __name__ == "__main__":
    main()
