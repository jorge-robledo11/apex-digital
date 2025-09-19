import pandas as pd
from sklearn.pipeline import Pipeline
from feature_engine.selection import (
    DropFeatures, DropConstantFeatures, DropCorrelatedFeatures
)
from feature_engine.datetime import DatetimeFeatures

from config.settings import get_settings
from src.utils.utils_fn import capture_variables
from src.utils.processors import CramersVCorrelatedSelection

settings = get_settings()
logger = settings.logger


def feature_selection(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Selecciona features con extracción datetime y preserva nombres originales.

    Extrae `day_of_week` y `quarter` de columnas datetime, elimina IDs, casi-constantes,
    categóricas correlacionadas (Cramér's V) y numéricas correlacionadas (Pearson).

    Args:
        X_train: Conjunto de entrenamiento.
        X_val: Conjunto de validación.
        X_test: Conjunto de prueba.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Versiones transformadas
        de train, val y test.
    """
    # Preservar nombres originales para evitar efectos colaterales de conversiones
    original_columns = list(X_train.columns)

    # Copias defensivas sin forzar materialización a pandas si no es necesario
    X_train = X_train.to_pandas().copy() if hasattr(X_train, "to_pandas") else X_train.copy()
    X_val = X_val.to_pandas().copy() if hasattr(X_val, "to_pandas") else X_val.copy()
    X_test = X_test.to_pandas().copy() if hasattr(X_test, "to_pandas") else X_test.copy()

    # Reaplicar nombres originales
    X_train.columns = original_columns
    X_val.columns = [c for c in original_columns if c in X_val.columns]
    X_test.columns = [c for c in original_columns if c in X_test.columns]

    logger.info(f"🏷️ Columnas originales: {original_columns}")

    # Tipificación de variables
    continuous, categoricals, discretes, temporaries = capture_variables(data=X_train)
    logger.info(f"⏰ Variables datetime detectadas: {temporaries}")

    # IDs por patrón simple
    ids_to_drop = [c for c in original_columns if "_id" in str(c).lower()]
    logger.info(f"🧾 IDs detectados para eliminar: {ids_to_drop or 'ninguno'}")

    steps: list[tuple[str, object]] = []
    new_datetime_features: list[str] = []

    # 1) Extracción de features datetime (primero, para registrar nuevas columnas)
    if temporaries:
        steps.append((
            "datetime_features",
            DatetimeFeatures(
                variables=temporaries,
                features_to_extract=["day_of_week", "quarter"],
                drop_original=True,
            ),
        ))
        # Pre-fit para identificar columnas derivadas
        dt_probe = DatetimeFeatures(
            variables=temporaries,
            features_to_extract=["day_of_week", "quarter"],
            drop_original=False,
        )
        X_train_temp = dt_probe.fit_transform(X_train)
        new_datetime_features = [c for c in X_train_temp.columns if c not in original_columns]
        logger.info(f"⏰ Nuevas features datetime: {new_datetime_features}")

        # Tratar derivadas como categóricas
        categoricals.extend(new_datetime_features)

    # 2) Eliminación de IDs
    if ids_to_drop:
        steps.append(("drop_ids", DropFeatures(features_to_drop=ids_to_drop)))

    # Listas para cada etapa (evitar colisiones con IDs)
    vars_for_constant = [c for c in (continuous + categoricals) if c not in ids_to_drop]
    cat_for_cramers = [c for c in categoricals if c not in ids_to_drop]
    num_for_pearson = [c for c in (continuous + discretes) if c not in ids_to_drop]

    logger.info(f"📦 Candidatas a casi-constantes: {len(vars_for_constant)}")
    logger.info(f"🔣 Categóricas para Cramér's V: {len(cat_for_cramers)} (incluye datetime)")
    logger.info(f"📈 Numéricas para Pearson: {len(num_for_pearson)}")

    # 3) Casi-constantes
    steps.append((
        "drop_constant",
        DropConstantFeatures(
            variables=vars_for_constant if vars_for_constant else None,
            missing_values="ignore",
            tol=0.95,
        ),
    ))

    # 4) Categóricas correlacionadas (Cramér's V)
    if cat_for_cramers:
        steps.append((
            "drop_cat_correlated",
            CramersVCorrelatedSelection(
                variables=cat_for_cramers,
                threshold=0.80,
                selection_method="first",
            ),
        ))

    # 5) Numéricas correlacionadas (Pearson)
    if len(num_for_pearson) >= 2:
        steps.append((
            "drop_num_correlated",
            DropCorrelatedFeatures(
                variables=num_for_pearson,
                method="pearson",
                threshold=0.80,
                missing_values="ignore",
            ),
        ))

    pipe = Pipeline(steps)

    # Ajuste paso a paso para rastrear columnas creadas/eliminadas
    logger.info("🛠️ Ajustando pipeline de selección…")
    dropped_ids: list[str] = []
    dropped_const: list[str] = []
    dropped_cat: list[str] = []
    dropped_num: list[str] = []

    X_tmp = X_train.to_pandas().copy() if hasattr(X_train, "to_pandas") else X_train.copy()
    for name, transformer in pipe.steps:
        cols_before = list(X_tmp.columns)
        transformer.fit(X_tmp)
        X_tmp = transformer.transform(X_tmp)
        cols_after = list(X_tmp.columns)

        dropped_this_step = [c for c in cols_before if c not in cols_after]
        created_this_step = [c for c in cols_after if c not in cols_before]

        if name == "datetime_features":
            logger.info(f"⏰ Features datetime creadas: {created_this_step or 'ninguna'}")
        elif name == "drop_ids":
            dropped_ids = dropped_this_step
            logger.info(f"🧾 Eliminadas por ID: {dropped_ids or 'ninguna'}")
        elif name == "drop_constant":
            dropped_const = dropped_this_step
            logger.info(f"📦 Eliminadas por casi-constantes: {dropped_const or 'ninguna'}")
        elif name == "drop_cat_correlated":
            dropped_cat = dropped_this_step
            logger.info(f"🔣 Eliminadas por Cramér's V: {dropped_cat or 'ninguna'}")
        elif name == "drop_num_correlated":
            dropped_num = dropped_this_step
            logger.info(f"📈 Eliminadas por Pearson: {dropped_num or 'ninguna'}")

    logger.success("✅ Pipeline de selección ajustado.")

    # Transformaciones finales
    X_train_sel = X_tmp.reset_index(drop=True)
    X_val_sel = pipe.transform(X_val).reset_index(drop=True)
    X_test_sel = pipe.transform(X_test).reset_index(drop=True)

    # Convertir nuevas features datetime a category
    if new_datetime_features:
        existing = [f for f in new_datetime_features if f in X_train_sel.columns]
        if existing:
            logger.info(f"🏷️ Casteando features datetime a category: {existing}")
            for col in existing:
                X_train_sel[col] = X_train_sel[col].astype("category")
                if col in X_val_sel.columns:
                    X_val_sel[col] = X_val_sel[col].astype("category")
                if col in X_test_sel.columns:
                    X_test_sel[col] = X_test_sel[col].astype("category")
            for col in existing:
                logger.info(f"   {col}: {X_train_sel[col].dtype}")

    logger.success(
        f"🎯 Feature selection completa. "
        f"Train={X_train_sel.shape}, Val={X_val_sel.shape}, Test={X_test_sel.shape}"
    )
    logger.info(f"🏷️ Columnas finales: {list(X_train_sel.columns)}")
    logger.info(
        f"⏰ Features datetime finales: "
        f"{[f for f in new_datetime_features if f in X_train_sel.columns]}"
    )
    return X_train_sel, X_val_sel, X_test_sel
