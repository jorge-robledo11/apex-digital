"""
Utilidades de entrenamiento (helpers puros y reutilizables).
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight

from config.settings import get_settings
from src.utils.utils_fn import capture_variables

settings = get_settings()
logger = settings.logger


def prepare_categorical_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> list[str]:
    """
    Castea features categóricas a dtype 'category' usando categorías de TRAIN.

    Args:
        X_train: Conjunto de entrenamiento.
        X_val: Conjunto de validación.
        X_test: Conjunto de prueba.

    Returns:
        list[str]: Nombres de columnas categóricas casteadas.

    Raises:
        ValueError: Si alguno de los datasets no tiene las columnas detectadas.
    """
    continuous, categoricals, discretes, temporaries = capture_variables(X_train)
    cat_cols = [c for c in categoricals if c in X_train.columns]

    if not cat_cols:
        logger.info("ℹ️ No se detectaron features categóricas para castear.")
        return []

    for c in cat_cols:
        cats = pd.Categorical(X_train[c]).categories
        try:
            X_train[c] = pd.Categorical(X_train[c], categories=cats)
            if c in X_val.columns:
                X_val[c] = pd.Categorical(X_val[c], categories=cats)
            if c in X_test.columns:
                X_test[c] = pd.Categorical(X_test[c], categories=cats)
        except Exception as e:
            logger.error(f"❌ Error casteando columna '{c}' a category: {e}")
            raise ValueError(f"No se pudo castear '{c}' a category.") from e

    logger.info(f"🏷️ Casteadas a 'category': {cat_cols}")
    return cat_cols


def encode_target_like_train(
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Codifica el target con las categorías de TRAIN (requerido por XGBoost sklearn API).

    Args:
        y_train: Target de entrenamiento.
        y_val: Target de validación.
        y_test: Target de prueba.

    Returns:
        tuple: (y_train_enc, y_val_enc, y_test_enc, class_names)

    Raises:
        ValueError: Si se encuentran clases no vistas en TRAIN.
    """
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
    """
    Calcula sample_weight estilo class_weight='balanced' para multiclase.

    Args:
        y_enc: Target codificado (códigos numéricos).

    Returns:
        np.ndarray: Vector de pesos por muestra.

    Raises:
        ValueError: Si el cálculo de pesos falla.
    """
    try:
        classes = np.unique(y_enc)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_enc)
        w_map = {c: w for c, w in zip(classes, weights)}
        return np.array([w_map[v] for v in y_enc], dtype=float)
    except Exception as e:
        raise ValueError(f"No se pudieron calcular los sample_weight balanceados: {e}") from e


def predict_with_best(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """
    Predicción de clases considerando la mejor iteración si existe.

    Args:
        model: Modelo XGBClassifier entrenado.
        X: Features de entrada.

    Returns:
        np.ndarray: Predicciones de clase.
    """
    best_iter = getattr(model, "best_iteration", None)
    if best_iter is not None:
        return model.predict(X, iteration_range=(0, best_iter + 1))
    return model.predict(X)


def predict_proba_with_best(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """
    Predicción de probabilidades considerando la mejor iteración si existe.

    Args:
        model: Modelo XGBClassifier entrenado.
        X: Features de entrada.

    Returns:
        np.ndarray: Probabilidades por clase.
    """
    best_iter = getattr(model, "best_iteration", None)
    if best_iter is not None:
        return model.predict_proba(X, iteration_range=(0, best_iter + 1))
    return model.predict_proba(X)


def make_xgb_params(use_gpu: bool, early_stopping_rounds: int) -> dict[str, object]:
    """
    Construye el diccionario de hiperparámetros base para XGBClassifier.

    Args:
        use_gpu: Si se debe intentar entrenar en GPU.
        early_stopping_rounds: Rondas para early stopping (en el constructor del estimador).

    Returns:
        dict[str, object]: Parámetros para inicializar XGBClassifier.
    """
    params: dict[str, object] = dict(
        objective="multi:softprob",
        eval_metric="mlogloss",
        enable_categorical=True,
        early_stopping_rounds=early_stopping_rounds,
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        # reg_lambda=1.5,
        # reg_alpha=0.0,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    if use_gpu:
        params.update(tree_method="gpu_hist", predictor="gpu_predictor")
    else:
        params.update(tree_method="hist", predictor="auto")
    return params
