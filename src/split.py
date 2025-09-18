import pandas as pd
from sklearn.model_selection import train_test_split
from config.settings import get_settings
from src.utils.utils_fn import capture_variables  # para identificar las discretas

settings = get_settings()
logger = settings.logger


def split_train_val_test(
    data: pd.DataFrame,
    target: str,
    seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """
    Separa el DataFrame en conjuntos de entrenamiento, validación y prueba 
    en las proporciones 60%, 20% y 20%, respectivamente.

    Antes de dividir, convierte las variables discretas a tipo 'category'.

    Parámetros
    ----------
    data : pd.DataFrame
        DataFrame que contiene los features y la columna objetivo.
    target : str
        Nombre de la columna objetivo en 'data'.
    seed : int
        Semilla para la reproducibilidad.

    Retorna
    -------
    X_train, X_val, X_test : pd.DataFrame
        Conjuntos de características para entrenamiento, validación y prueba.
    y_train, y_val, y_test : pd.Series
        Conjuntos de la variable objetivo para entrenamiento, validación y prueba.
    """

    logger.info("🚀 Iniciando split_train_val_test()")

    # 1. Capturar tipos de variables
    _, _, discretes, _ = capture_variables(data=data)
    logger.info(f"🔤 Variables discretas detectadas: {discretes}")

    # 2. Castear discretas a 'category'
    for col in discretes:
        if col in data.columns:
            try:
                data[col] = data[col].astype("category")
            except Exception as e:
                logger.warning(f"⚠️ No se pudo castear '{col}' a category: {e}")
    logger.success("✅ Casteo de discretas completado")

    # 3. Separar características (X) y objetivo (y)
    X = data.drop(columns=[target])
    y = data[target].squeeze()

    # 4. Dividir en train (60%) y temp (40%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.4,
        random_state=seed,
        stratify=y
    )

    # 5. Dividir temp en val (20%) y test (20%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=seed,
        stratify=y_temp
    )

    logger.success(
        f"🎯 Split completado -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
