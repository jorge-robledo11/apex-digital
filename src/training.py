"""
Orquestación del entrenamiento con XGBoost + MLflow.

- Convierte features categóricas a dtype 'category' (vía utils).
- Codifica SOLO el target a códigos numéricos (vía utils).
- Entrena con early stopping y GPU con fallback a CPU.
- Calcula y empaqueta métricas con modelos Pydantic (src/utils/metrics.py).
- Loguea parámetros, métricas y artefactos en MLflow.
"""

from typing import Any

import mlflow
import mlflow.xgboost
import pandas as pd
from mlflow.models import infer_signature
from xgboost import XGBClassifier

from config.settings import get_settings
from src.utils.training_utils import (
    prepare_categorical_features,
    encode_target_like_train,
    compute_balanced_sample_weight,
    predict_with_best,
    predict_proba_with_best,
    make_xgb_params,
)
from src.utils.metrics import (
    DatasetShapeInfo,
    TrainingFlags,
    XGBHyperparams,
    BestIterationInfo,
    TrainingMetrics,
    compute_classification_metrics,
    log_training_metrics_to_mlflow,
)

settings = get_settings()
logger = settings.logger


def _safe_log_params(params: dict[str, Any]) -> None:
    """Loguea parámetros en MLflow sin romper el flujo si ocurre un error."""
    try:
        mlflow.log_params(params)
    except Exception as e:
        logger.warning(f"⚠️ No se pudieron loguear algunos parámetros: {e}")


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
    Entrena un `XGBClassifier` multiclase y registra el proceso en MLflow.

    Aplica:
      - Features categóricas nativas (enable_categorical=True).
      - Codificación del target (códigos numéricos).
      - Balanceo opcional de clases mediante sample_weight.
      - Early stopping y fallback automático a CPU si GPU falla.
      - Cálculo y empaquetado de métricas con modelos Pydantic.
      - Registro de métricas/artefactos en MLflow.

    Args:
        X_train: Matriz de features de entrenamiento.
        X_val: Matriz de features de validación.
        X_test: Matriz de features de prueba.
        y_train: Target de entrenamiento.
        y_val: Target de validación.
        y_test: Target de prueba.
        use_balanced_weights: Si se aplican pesos balanceados por clase.
        use_gpu: Si se intenta entrenar sobre GPU.
        early_stopping_rounds: Rondas máximas sin mejora para detener.

    Returns:
        tuple[XGBClassifier, float, str]: (modelo_entrenado, val_logloss, "XGBoost").

    Raises:
        ValueError: Si algún split requerido está vacío o el target tiene clases no vistas.
        RuntimeError: Si el entrenamiento falla en GPU y CPU, o si el cálculo de métricas no es consistente.
    """
    logger.info("🚀 Iniciando entrenamiento con XGBoost...")

    # -------------------------
    # Validaciones básicas
    # -------------------------
    if X_train.empty or y_train.empty:
        logger.error("❌ Datos de entrenamiento vacíos.")
        raise ValueError("X_train/y_train no pueden estar vacíos.")
    if X_val.empty or y_val.empty:
        logger.error("❌ Datos de validación vacíos.")
        raise ValueError("X_val/y_val no pueden estar vacíos.")
    if X_test.empty or y_test.empty:
        logger.error("❌ Datos de prueba vacíos.")
        raise ValueError("X_test/y_test no pueden estar vacíos.")

    # -------------------------
    # 1) Features categóricas
    # -------------------------
    cat_cols = prepare_categorical_features(X_train, X_val, X_test)

    # -------------------------
    # 2) Codificación del target
    # -------------------------
    try:
        y_train_enc, y_val_enc, y_test_enc, class_names = encode_target_like_train(
            y_train, y_val, y_test
        )
    except ValueError as e:
        logger.error(f"❌ Error al codificar target: {e}")
        raise

    # -------------------------
    # 3) Parámetros del modelo
    # -------------------------
    params = make_xgb_params(use_gpu=use_gpu, early_stopping_rounds=early_stopping_rounds)
    if use_gpu:
        logger.info("🟢 Se intentará entrenar con GPU (gpu_hist).")
    model = XGBClassifier(**params)

    # -------------------------
    # 4) sample_weight (opcional)
    # -------------------------
    fit_kwargs: dict[str, Any] = {}
    if use_balanced_weights:
        try:
            sw = compute_balanced_sample_weight(y_train_enc)
            fit_kwargs["sample_weight"] = sw
            logger.info("⚖️  Usando sample_weight balanceado por clase.")
        except ValueError as e:
            logger.warning(f"⚠️ No se pudo calcular sample_weight balanceado: {e}")

    # =========================================================
    # Run anidado (cuelga del run activo en el proceso padre)
    # =========================================================
    with mlflow.start_run(run_name="xgboost_training", nested=True):
        logger.info("🔗 Run de entrenamiento anidado al run activo de MLflow.")
        logger.info("📊 Entrenando XGBoost con early stopping…")

        # Log de tamaños y flags informativos
        _safe_log_params(
            {
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_test": len(X_test),
                "n_features": X_train.shape[1],
                "early_stopping_rounds": early_stopping_rounds,
                "used_balanced_weights": use_balanced_weights,
                "used_gpu": use_gpu,
                "categorical_features": True,
                "encoded_target": True,
            }
        )
        if cat_cols:
            try:
                mlflow.log_text("\n".join(map(str, cat_cols)), "artifacts/categorical_features.txt")
            except Exception as e:
                logger.warning(f"⚠️ No se pudieron loguear las columnas categóricas: {e}")
        try:
            mlflow.log_text("\n".join(map(str, class_names)), "artifacts/class_names.txt")
        except Exception as e:
            logger.warning(f"⚠️ No se pudieron loguear los nombres de clase: {e}")

        # -------------------------
        # 5) Entrenamiento (GPU → CPU fallback)
        # -------------------------
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
                    raise RuntimeError("Falló el entrenamiento tanto en GPU como en CPU.") from e2
                logger.info("✅ Reintento en CPU exitoso.")
            else:
                logger.error(f"❌ Entrenamiento en CPU falló: {e}")
                raise RuntimeError("Falló el entrenamiento en CPU.") from e

        # -------------------------
        # 6) Predicciones (mejor iteración)
        # -------------------------
        y_train_proba = predict_proba_with_best(model, X_train)
        y_val_proba = predict_proba_with_best(model, X_val)
        y_test_proba = predict_proba_with_best(model, X_test)

        y_train_pred = predict_with_best(model, X_train)
        y_val_pred = predict_with_best(model, X_val)
        y_test_pred = predict_with_best(model, X_test)

        # -------------------------
        # 7) Métricas (bundle Pydantic)
        # -------------------------
        try:
            losses, accuracies, reports, confusion = compute_classification_metrics(
                y_train_enc=y_train_enc,
                y_val_enc=y_val_enc,
                y_test_enc=y_test_enc,
                y_train_pred=y_train_pred,
                y_val_pred=y_val_pred,
                y_test_pred=y_test_pred,
                y_train_proba=y_train_proba,
                y_val_proba=y_val_proba,
                y_test_proba=y_test_proba,
                class_names=class_names,
            )
        except ValueError as e:
            logger.error(f"❌ Error en cálculo de métricas: {e}")
            raise RuntimeError("Cálculo de métricas inconsistente.") from e

        shapes = DatasetShapeInfo(
            n_train=len(X_train), n_val=len(X_val), n_test=len(X_test), n_features=X_train.shape[1]
        )
        flags = TrainingFlags(
            early_stopping_rounds=early_stopping_rounds,
            used_balanced_weights=use_balanced_weights,
            used_gpu=use_gpu,
        )
        xgb_hparams = XGBHyperparams(
            objective=model.objective,
            n_estimators=model.n_estimators,
            learning_rate=model.learning_rate,
            max_depth=model.max_depth,
            min_child_weight=model.min_child_weight,
            subsample=model.subsample,
            colsample_bytree=model.colsample_bytree,
            reg_lambda=model.reg_lambda,
            reg_alpha=model.reg_alpha,
            tree_method=getattr(model, "tree_method", "unknown"),
            predictor=getattr(model, "predictor", "unknown"),
        )
        best_iter_info = BestIterationInfo(
            best_iteration=getattr(model, "best_iteration", None),
            best_val_mlogloss=(
                float(getattr(model, "best_score"))
                if getattr(model, "best_score", None) is not None
                else None
            ),
        )

        bundle = TrainingMetrics(
            shapes=shapes,
            flags=flags,
            losses=losses,
            accuracies=accuracies,
            reports=reports,
            confusion=confusion,
            xgb_hyperparams=xgb_hparams,
            best_iter_info=best_iter_info,
        )

        # -------------------------
        # 8) Logging consolidado a MLflow
        # -------------------------
        try:
            log_training_metrics_to_mlflow(bundle, mlflow_module=mlflow)
        except Exception as e:  # pragma: no cover
            logger.warning(f"⚠️ No se pudieron loguear métricas en MLflow: {e}")

        # -------------------------
        # 9) Firma + registro del modelo
        # -------------------------
        try:
            signature = infer_signature(X_val, model.predict_proba(X_val))
        except Exception as e:  # pragma: no cover
            logger.warning(f"⚠️ No se pudo inferir la firma del modelo: {e}")
            signature = None

        try:
            mlflow.xgboost.log_model(
                model,
                artifact_path="xgboost_model",
                registered_model_name="digital_orders_xgboost",
                signature=signature,
            )
        except Exception as e:  # pragma: no cover
            logger.warning(f"⚠️ No se pudo loggear/registrar el modelo: {e}")

        # -------------------------
        # 10) Logs de cortesía
        # -------------------------
        logger.info("📈 Métricas de Rendimiento:")
        logger.info(
            f"   🎯 LogLoss → Train {losses.train_logloss:.4f} | "
            f"Val {losses.val_logloss:.4f} | Test {losses.test_logloss:.4f}"
        )
        logger.info(
            f"   📊 Acc    → Train {accuracies.train_accuracy:.4f} | "
            f"Val {accuracies.val_accuracy:.4f} | Test {accuracies.test_accuracy:.4f}"
        )
        if best_iter_info.best_iteration is not None and best_iter_info.best_val_mlogloss is not None:
            logger.info(
                f"🏁 Best iteration: {best_iter_info.best_iteration}  |  "
                f"Best val mlogloss: {best_iter_info.best_val_mlogloss:.6f}"
            )

    # Devolvemos el modelo y la métrica principal de validación
    return model, float(losses.val_logloss), "XGBoost"
