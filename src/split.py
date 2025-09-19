import pandas as pd
from sklearn.model_selection import train_test_split
from config.settings import get_settings
from src.utils.utils_fn import capture_variables

settings = get_settings()
logger = settings.logger


def split_train_val_test(
    data: pd.DataFrame,
    target: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """
    Divide un DataFrame en conjuntos de entrenamiento, validación y prueba.

    La división se realiza en proporciones 60 % (train), 20 % (val) y 20 % (test).
    Antes de separar, convierte las variables discretas detectadas a tipo ``category``.

    Args:
        data: DataFrame con características y la columna objetivo.
        target: Nombre de la columna objetivo.
        seed: Semilla para la reproducibilidad.

    Returns:
        tuple:
            - X_train, X_val, X_test (pd.DataFrame): Conjuntos de características.
            - y_train, y_val, y_test (pd.Series): Conjuntos de la variable objetivo.

    Raises:
        KeyError: Si la columna `target` no existe en el DataFrame.
        ValueError: Si ocurre un error crítico durante la conversión de tipos
            o durante la división estratificada.
    """
    logger.info("🚀 Iniciando split_train_val_test()")

    if target not in data.columns:
        logger.error(f"❌ La columna objetivo '{target}' no existe en el DataFrame.")
        raise KeyError(f"La columna objetivo '{target}' no existe en el DataFrame.")

    # Detectar variables discretas y castear a category
    _, _, discretes, _ = capture_variables(data=data)
    logger.info(f"🔤 Variables discretas detectadas: {discretes}")
    for col in discretes:
        if col in data.columns:
            try:
                data[col] = data[col].astype("category")
            except Exception as e:
                logger.error(f"❌ No se pudo castear '{col}' a category: {e}")
                raise ValueError(f"No se pudo castear '{col}' a category.") from e
    logger.success("✅ Casteo de discretas completado")

    # Separar características y objetivo
    X = data.drop(columns=[target])
    y = data[target].squeeze()

    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=0.4,
            random_state=seed,
            stratify=y,
        )
    except ValueError as e:
        logger.error(f"❌ Error en la división train/temp: {e}")
        raise

    try:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,
            random_state=seed,
            stratify=y_temp,
        )
    except ValueError as e:
        logger.error(f"❌ Error en la división val/test: {e}")
        raise

    logger.success(
        f"🎯 Split completado -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
