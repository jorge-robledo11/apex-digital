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
    """
    logger.info("🚀 Iniciando split_train_val_test()")

    # Detectar variables discretas y castear a category
    _, _, discretes, _ = capture_variables(data=data)
    logger.info(f"🔤 Variables discretas detectadas: {discretes}")
    for col in discretes:
        if col in data.columns:
            try:
                data[col] = data[col].astype("category")
            except Exception as e:
                logger.warning(f"⚠️ No se pudo castear '{col}' a category: {e}")
    logger.success("✅ Casteo de discretas completado")

    # Separar características y objetivo
    X = data.drop(columns=[target])
    y = data[target].squeeze()

    # Split 60 % train / 40 % temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.4,
        random_state=seed,
        stratify=y,
    )

    # Split temp en 20 % val / 20 % test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=seed,
        stratify=y_temp,
    )

    logger.success(
        f"🎯 Split completado -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
