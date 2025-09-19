"""
Orquestación del entrenamiento con XGBoost + MLflow.
"""

import mlflow
import mlflow.xgboost
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss
from mlflow.models import infer_signature

from config.settings import get_settings
from src.utils.training_utils import (
    prepare_categorical_features,
    encode_target_like_train,
    compute_balanced_sample_weight,
    predict_with_best,
    predict_proba_with_best,
    make_xgb_params,
)

settings = get_settings()
logger = settings.logger


def train_models(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    use_balanced_weights: bool = True,
    use_gpu: bool = True,
    early_stopping_rounds: int = 50,
) -> tuple[XGBClassifier, float, str]:
    """
    Entrena un XGBClassifier multiclase con logging en MLflow.

    Aplica:
      - Features categóricas nativas (enable_categorical=True).
      - Target codificado (códigos numéricos).
      - Balanceo de clases opcional.
      - Early stopping y GPU con fallback a CPU.
      - Métrica principal: mlogloss.
      - Run anidado en MLflow (hereda tracking/experimento del proceso padre).

    Args:
        X_train: Features de entrenamiento.
        X_val: Features de validación.
        X_test: Features de prueba.
        y_train: Target de entrenamiento.
        y_val: Target de validación.
        y_test: Target de prueba.
        use_balanced_weights: Si se aplican pesos balanceados.
        use_gpu: Si se intenta entrenar en GPU.
        early_stopping_rounds: Rondas para early stopping.

    Returns:
        tuple[XGBClassifier, float, str]: (modelo, val_logloss, nombre_modelo)

    Raises:
        ValueError: Si los datos de entrada están vacíos o con clases no vistas.
        RuntimeError: Si falla el ajuste/transformación o el registro del modelo.
    """
    logger.info("🚀 Iniciando entrenamiento con XGBoost...")

    # Validaciones básicas
    if X_train.empty or y_train.empty:
        logger.error("❌ Datos de entrenamiento vacíos.")
        raise ValueError("X_train/y_train no pueden estar vacíos.")
    if X_val.empty or y_val.empty or X_test.empty or y_test.empty:
        logger.error("❌ Alguno de los datasets de validación/prueba está vacío.")
        raise ValueError("X_val/y_val/X_test/y_test no pueden estar vacíos.")

    # 1) Castear features categóricas
    cat_cols = prepare_categorical_features(X_train, X_val, X_test)

    # 2) Codificar solo el target
    try:
        y_train_enc, y_val_enc, y_test_enc, class_names = encode_target_like_train(
            y_train, y_val, y_test
        )
    except ValueError as e:
        logger.error(f"❌ Error al codificar target: {e}")
        raise

    # 3) Parámetros del modelo
    params = make_xgb_params(use_gpu=use_gpu, early_stopping_rounds=early_stopping_rounds)
    if use_gpu:
        logger.info("🟢 Se intentará entrenar con GPU (gpu_hist).")
    model = XGBClassifier(**params)

    # 4) sample_weight opcional
    fit_kwargs: dict[str, object] = {}
    if use_balanced_weights:
        try:
            sw = compute_balanced_sample_weight(y_train_enc)
            fit_kwargs["sample_weight"] = sw
            logger.info("⚖️  Usando sample_weight balanceado por clase.")
        except ValueError as e:
            logger.warning(f"⚠️ No se pudo calcular sample_weight balanceado: {e}")

    # =========================
    # Run anidado (cuelga del run activo)
    # =========================
    with mlflow.start_run(run_name="xgboost_training", nested=True):
        logger.info("🔗 Run de entrenamiento anidado al run activo de MLflow.")
        logger.info("📊 Entrenando XGBoost con early stopping…")

        # Log de datos/flags
        mlflow.log_params({
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "n_features": X_train.shape[1],
            "early_stopping_rounds": early_stopping_rounds,
            "used_balanced_weights": use_balanced_weights,
            "used_gpu": use_gpu,
            "categorical_features": True,
            "encoded_target": True,
        })
        if cat_cols:
            mlflow.log_text("\n".join(map(str, cat_cols)), "artifacts/categorical_features.txt")
        mlflow.log_text("\n".join(map(str, class_names)), "artifacts/class_names.txt")

        # 5) Entrenamiento con fallback CPU si GPU falla
        try:
            model.fit(
                X_train,
                y_train_enc,
                eval_set=[(X_val, y_val_enc)],
                verbose=False,
                **fit_kwargs,
            )
        except Exception as e:
            if use_gpu:
                logger.warning(f"⚠️ Entrenamiento con GPU falló ({e}). Reintentando en CPU…")
                params = make_xgb_params(use_gpu=False, early_stopping_rounds=early_stopping_rounds)
                model = XGBClassifier(**params)
                try:
                    model.fit(
                        X_train,
                        y_train_enc,
                        eval_set=[(X_val, y_val_enc)],
                        verbose=False,
                        **fit_kwargs,
                    )
                except Exception as e2:
                    logger.error(f"❌ Reintento en CPU falló: {e2}")
                    raise RuntimeError("Fallo el entrenamiento en GPU y CPU.") from e2
                logger.info("✅ Reintento en CPU exitoso.")
            else:
                logger.error(f"❌ Entrenamiento en CPU falló: {e}")
                raise RuntimeError("Fallo el entrenamiento en CPU.") from e

        # 6) Predicciones (mejor iteración)
        y_train_proba = predict_proba_with_best(model, X_train)
        y_val_proba = predict_proba_with_best(model, X_val)
        y_test_proba = predict_proba_with_best(model, X_test)

        y_train_pred = predict_with_best(model, X_train)
        y_val_pred = predict_with_best(model, X_val)
        y_test_pred = predict_with_best(model, X_test)

        # 7) Métricas
        labels_idx = list(range(len(class_names)))
        train_logloss = log_loss(y_train_enc, y_train_proba, labels=labels_idx)
        val_logloss = log_loss(y_val_enc, y_val_proba, labels=labels_idx)
        test_logloss = log_loss(y_test_enc, y_test_proba, labels=labels_idx)

        train_acc = accuracy_score(y_train_enc, y_train_pred)
        val_acc = accuracy_score(y_val_enc, y_val_pred)
        test_acc = accuracy_score(y_test_enc, y_test_pred)

        # 8) MLflow: hiperparámetros finales y métricas
        mlflow.log_params({
            "objective": model.objective,
            "eval_metric": "mlogloss",
            "enable_categorical": True,
            "n_estimators": model.n_estimators,
            "learning_rate": model.learning_rate,
            "max_depth": model.max_depth,
            "min_child_weight": model.min_child_weight,
            "subsample": model.subsample,
            "colsample_bytree": model.colsample_bytree,
            "reg_lambda": model.reg_lambda,
            "reg_alpha": model.reg_alpha,
            "tree_method": getattr(model, "tree_method", "unknown"),
            "predictor": getattr(model, "predictor", "unknown"),
        })

        best_iter = getattr(model, "best_iteration", None)
        best_score = getattr(model, "best_score", None)
        if best_iter is not None:
            mlflow.log_metric("best_iteration", int(best_iter))
        if best_score is not None:
            mlflow.log_metric("best_val_mlogloss", float(best_score))

        mlflow.log_metrics({
            "train_logloss": float(train_logloss),
            "val_logloss": float(val_logloss),
            "test_logloss": float(test_logloss),
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "test_accuracy": float(test_acc),
        })

        # 9) Reports
        mlflow.log_text(
            classification_report(y_val_enc, y_val_pred, target_names=class_names),
            "validation_classification_report.txt",
        )
        mlflow.log_text(
            classification_report(y_test_enc, y_test_pred, target_names=class_names),
            "test_classification_report.txt",
        )

        # 10) Firma + registro del modelo
        try:
            signature = infer_signature(X_val, model.predict_proba(X_val))
            mlflow.xgboost.log_model(
                model,
                artifact_path="xgboost_model",
                registered_model_name="digital_orders_xgboost",
                signature=signature,
            )
        except Exception as e:
            logger.warning(f"⚠️ No se pudo loggear/registrar el modelo: {e}")

        # Logs usuario
        logger.info("📈 Métricas de Rendimiento:")
        logger.info(f"   🎯 LogLoss → Train {train_logloss:.4f} | Val {val_logloss:.4f} | Test {test_logloss:.4f}")
        logger.info(f"   📊 Acc    → Train {train_acc:.4f} | Val {val_acc:.4f} | Test {test_acc:.4f}")
        if best_iter is not None and best_score is not None:
            logger.info(f"🏁 Best iteration: {best_iter}  |  Best val mlogloss: {best_score:.6f}")

    return model, float(val_logloss), "XGBoost"
