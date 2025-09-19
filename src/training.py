import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.utils.class_weight import compute_class_weight
from config.settings import get_settings
from src.utils.utils_fn import capture_variables

# =========================
# Logger global del proyecto (el mismo que usa main)
# =========================
settings = get_settings()
logger = settings.logger

# =========================
# Helpers (fuera de train)
# =========================
def prepare_categorical_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> list[str]:
    """Castea features categóricas a dtype 'category' (categorías tomadas de TRAIN)."""
    continuous, categoricals, discretes, temporaries = capture_variables(X_train)
    cat_cols = [c for c in categoricals if c in X_train.columns]

    if not cat_cols:
        logger.info("ℹ️ No se detectaron features categóricas para castear.")
        return []

    for c in cat_cols:
        cats = pd.Categorical(X_train[c]).categories
        X_train[c] = pd.Categorical(X_train[c], categories=cats)
        if c in X_val.columns:
            X_val[c] = pd.Categorical(X_val[c], categories=cats)
        if c in X_test.columns:
            X_test[c] = pd.Categorical(X_test[c], categories=cats)

    logger.info(f"🏷️ Casteadas a 'category': {cat_cols}")
    return cat_cols

def encode_target_like_train(
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Codifica SOLO el target con categorías de TRAIN (XGBoost sklearn API requiere códigos)."""
    y_train_cat = pd.Categorical(y_train)
    classes_ = list(map(str, y_train_cat.categories))

    def _enc(y: pd.Series) -> np.ndarray:
        y_cat = pd.Categorical(y, categories=y_train_cat.categories)
        if np.any(pd.isna(y_cat)):
            unseen = sorted(set(y[pd.isna(y_cat)].astype(str)))
            raise ValueError(f"Clases no vistas en TRAIN: {unseen}")
        return y_cat.codes

    y_train_enc = _enc(y_train)
    y_val_enc = _enc(y_val)
    y_test_enc = _enc(y_test)

    logger.info(f"🎯 Clases (orden): {classes_}")
    return y_train_enc, y_val_enc, y_test_enc, classes_

def compute_balanced_sample_weight(y_enc: np.ndarray) -> np.ndarray:
    """Sample_weight estilo class_weight='balanced' para multiclase."""
    classes = np.unique(y_enc)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_enc)
    w_map = {c: w for c, w in zip(classes, weights)}
    return np.array([w_map[v] for v in y_enc], dtype=float)

def _predict_with_best(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    best_iter = getattr(model, "best_iteration", None)
    if best_iter is not None:
        return model.predict(X, iteration_range=(0, best_iter + 1))
    return model.predict(X)

def _predict_proba_with_best(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    best_iter = getattr(model, "best_iteration", None)
    if best_iter is not None:
        return model.predict_proba(X, iteration_range=(0, best_iter + 1))
    return model.predict_proba(X)

# =========================
# Función principal
# =========================
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
    XGBoost multiclase con:
      - Features categóricas nativas (enable_categorical=True).
      - Target codificado (XGBoost sklearn API requiere códigos).
      - Balanceo de clases opcional.
      - Early stopping y GPU con fallback a CPU.
      - Métrica principal: mlogloss.
      - Run anidado en MLflow (hereda tracking/experimento del main).
    """
    logger.info("🚀 Iniciando entrenamiento con XGBoost...")

    # 1) Castear features categóricas
    cat_cols = prepare_categorical_features(X_train, X_val, X_test)

    # 2) Codificar SOLO el target (XGBoost sklearn API requiere códigos)
    y_train_enc, y_val_enc, y_test_enc, class_names = encode_target_like_train(
        y_train, y_val, y_test
    )

    # 3) Parámetros del modelo (early_stopping_rounds va en constructor)
    params: dict[str, object] = dict(
        objective="multi:softprob",
        eval_metric="mlogloss",
        enable_categorical=True,  # ✅ Solo para FEATURES
        early_stopping_rounds=early_stopping_rounds,  # ✅ En constructor, no en fit()
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.5,
        reg_alpha=0.0,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    if use_gpu:
        params.update(tree_method="gpu_hist", predictor="gpu_predictor")
        logger.info("🟢 Se intentará entrenar con GPU (gpu_hist).")
    else:
        params.update(tree_method="hist", predictor="auto")

    model = XGBClassifier(**params)

    # 4) sample_weight
    fit_kwargs: dict[str, object] = {}
    if use_balanced_weights:
        try:
            sw = compute_balanced_sample_weight(y_train_enc)
            fit_kwargs["sample_weight"] = sw
            logger.info("⚖️  Usando sample_weight balanceado por clase.")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo calcular sample_weight balanceado: {e}")

    # =========================
    # Run anidado (cuelga del run activo en main)
    # =========================
    with mlflow.start_run(run_name="xgboost_training", nested=True):
        logger.info("🔗 Run de entrenamiento anidado al run activo de MLflow.")
        logger.info("📊 Entrenando XGBoost con early stopping…")

        # Log info útil del dataset al inicio del run anidado
        mlflow.log_params({
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "n_features": X_train.shape[1],
            "early_stopping_rounds": early_stopping_rounds,
            "used_balanced_weights": use_balanced_weights,
            "used_gpu": use_gpu,
            "categorical_features": True,  # Features categóricas
            "encoded_target": True,       # Target encoded
        })
        
        # Artefactos informativos
        if cat_cols:
            mlflow.log_text("\n".join(map(str, cat_cols)), "artifacts/categorical_features.txt")
        mlflow.log_text("\n".join(map(str, class_names)), "artifacts/class_names.txt")

        # Fit con GPU y fallback a CPU - TARGETS ENCODED
        try:
            model.fit(
                X_train,
                y_train_enc,  # ✅ Target encoded
                eval_set=[(X_val, y_val_enc)],  # ✅ Validation encoded
                verbose=False,
                **fit_kwargs,
            )
        except Exception as e:
            if use_gpu:
                logger.warning(f"⚠️ Entrenamiento con GPU falló ({e}). Reintentando en CPU…")
                params.update(tree_method="hist", predictor="auto")
                # Recrear modelo con parámetros CPU
                model = XGBClassifier(**params)
                model.fit(
                    X_train,
                    y_train_enc,
                    eval_set=[(X_val, y_val_enc)],
                    verbose=False,
                    **fit_kwargs,
                )
                logger.info("✅ Reintento en CPU exitoso.")
            else:
                raise

        # 5) Predicciones (mejor iteración)
        y_train_proba = _predict_proba_with_best(model, X_train)
        y_val_proba = _predict_proba_with_best(model, X_val)
        y_test_proba = _predict_proba_with_best(model, X_test)

        y_train_pred = _predict_with_best(model, X_train)
        y_val_pred = _predict_with_best(model, X_val)
        y_test_pred = _predict_with_best(model, X_test)

        # 6) Métricas - USAR CÓDIGOS ENCODED
        labels_idx = list(range(len(class_names)))
        train_logloss = log_loss(y_train_enc, y_train_proba, labels=labels_idx)
        val_logloss = log_loss(y_val_enc, y_val_proba, labels=labels_idx)
        test_logloss = log_loss(y_test_enc, y_test_proba, labels=labels_idx)

        train_acc = accuracy_score(y_train_enc, y_train_pred)
        val_acc = accuracy_score(y_val_enc, y_val_pred)
        test_acc = accuracy_score(y_test_enc, y_test_pred)

        # 7) MLflow logging (hiperparámetros del modelo final)
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
            "train_logloss": train_logloss,
            "val_logloss": val_logloss,
            "test_logloss": test_logloss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
        })

        # Reports con nombres originales de clases
        mlflow.log_text(
            classification_report(y_val_enc, y_val_pred, target_names=class_names),
            "validation_classification_report.txt"
        )
        mlflow.log_text(
            classification_report(y_test_enc, y_test_pred, target_names=class_names),
            "test_classification_report.txt"
        )

        # ✅ Signature con features categóricas
        signature = infer_signature(X_val, model.predict_proba(X_val))
        
        # Log modelo entrenado (quedará colgado del run anidado)
        try:
            mlflow.xgboost.log_model(
                model,
                artifact_path="xgboost_model",
                registered_model_name="digital_orders_xgboost",
                signature=signature,
            )
        except Exception as e:
            logger.warning(f"⚠️ No se pudo loggear/registrar el modelo: {e}")

        # Logs al usuario
        logger.info("📈 Métricas de Rendimiento:")
        logger.info(f"   🎯 LogLoss → Train {train_logloss:.4f} | Val {val_logloss:.4f} | Test {test_logloss:.4f}")
        logger.info(f"   📊 Acc    → Train {train_acc:.4f} | Val {val_acc:.4f} | Test {test_acc:.4f}")
        if best_iter is not None and best_score is not None:
            logger.info(f"🏁 Best iteration: {best_iter}  |  Best val mlogloss: {best_score:.6f}")

    return model, val_logloss, "XGBoost"
