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

    Raises:
        ValueError: Si no se detectan columnas de entrada o si falla la tipificación.
        RuntimeError: Si ocurre un error durante el ajuste o transformación del pipeline.
    """
    if X_train.empty:
        logger.error("❌ El DataFrame de entrenamiento está vacío.")
        raise ValueError("El DataFrame de entrenamiento no puede estar vacío.")

    original_columns = list(X_train.columns)
    if not original_columns:
        logger.error("❌ El DataFrame de entrenamiento no tiene columnas.")
        raise ValueError("No se detectaron columnas en el DataFrame de entrenamiento.")

    X_train = X_train.to_pandas().copy() if hasattr(X_train, "to_pandas") else X_train.copy()
    X_val = X_val.to_pandas().copy() if hasattr(X_val, "to_pandas") else X_val.copy()
    X_test = X_test.to_pandas().copy() if hasattr(X_test, "to_pandas") else X_test.copy()

    X_train.columns = original_columns
    X_val.columns = [c for c in original_columns if c in X_val.columns]
    X_test.columns = [c for c in original_columns if c in X_test.columns]

    logger.info(f"🏷️ Columnas originales: {original_columns}")

    try:
        continuous, categoricals, discretes, temporaries = capture_variables(data=X_train)
    except Exception as e:
        logger.error(f"❌ Error al capturar variables: {e}")
        raise ValueError("No se pudieron tipificar las variables de entrada.") from e

    logger.info(f"⏰ Variables datetime detectadas: {temporaries}")

    ids_to_drop = [c for c in original_columns if "_id" in str(c).lower()]
    logger.info(f"🧾 IDs detectados para eliminar: {ids_to_drop or 'ninguno'}")

    steps: list[tuple[str, object]] = []
    new_datetime_features: list[str] = []

    if temporaries:
        try:
            steps.append((
                "datetime_features",
                DatetimeFeatures(
                    variables=temporaries,
                    features_to_extract=["day_of_week", "quarter"],
                    drop_original=True,
                ),
            ))
            dt_probe = DatetimeFeatures(
                variables=temporaries,
                features_to_extract=["day_of_week", "quarter"],
                drop_original=False,
            )
            X_train_temp = dt_probe.fit_transform(X_train)
            new_datetime_features = [c for c in X_train_temp.columns if c not in original_columns]
            logger.info(f"⏰ Nuevas features datetime: {new_datetime_features}")
            categoricals.extend(new_datetime_features)
        except Exception as e:
            logger.error(f"❌ Error en extracción de features datetime: {e}")
            raise RuntimeError("Fallo la extracción de features datetime.") from e

    if ids_to_drop:
        steps.append(("drop_ids", DropFeatures(features_to_drop=ids_to_drop)))

    vars_for_constant = [c for c in (continuous + categoricals) if c not in ids_to_drop]
    cat_for_cramers = [c for c in categoricals if c not in ids_to_drop]
    num_for_pearson = [c for c in (continuous + discretes) if c not in ids_to_drop]

    logger.info(f"📦 Candidatas a casi-constantes: {len(vars_for_constant)}")
    logger.info(f"🔣 Categóricas para Cramér's V: {len(cat_for_cramers)} (incluye datetime)")
    logger.info(f"📈 Numéricas para Pearson: {len(num_for_pearson)}")

    steps.append((
        "drop_constant",
        DropConstantFeatures(
            variables=vars_for_constant if vars_for_constant else None,
            missing_values="ignore",
            tol=0.95,
        ),
    ))

    if cat_for_cramers:
        steps.append((
            "drop_cat_correlated",
            CramersVCorrelatedSelection(
                variables=cat_for_cramers,
                threshold=0.80,
                selection_method="first",
            ),
        ))

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

    logger.info("🛠️ Ajustando pipeline de selección…")
    X_tmp = X_train.to_pandas().copy() if hasattr(X_train, "to_pandas") else X_train.copy()
    try:
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
                logger.info(f"🧾 Eliminadas por ID: {dropped_this_step or 'ninguna'}")
            elif name == "drop_constant":
                logger.info(f"📦 Eliminadas por casi-constantes: {dropped_this_step or 'ninguna'}")
            elif name == "drop_cat_correlated":
                logger.info(f"🔣 Eliminadas por Cramér's V: {dropped_this_step or 'ninguna'}")
            elif name == "drop_num_correlated":
                logger.info(f"📈 Eliminadas por Pearson: {dropped_this_step or 'ninguna'}")
    except Exception as e:
        logger.error(f"❌ Error al ajustar pipeline: {e}")
        raise RuntimeError("Error durante el ajuste del pipeline de selección.") from e

    logger.success("✅ Pipeline de selección ajustado.")

    try:
        X_train_sel = X_tmp.reset_index(drop=True)
        X_val_sel = pipe.transform(X_val).reset_index(drop=True)
        X_test_sel = pipe.transform(X_test).reset_index(drop=True)
    except Exception as e:
        logger.error(f"❌ Error al transformar datasets: {e}")
        raise RuntimeError("Error durante la transformación de los datasets.") from e

    if new_datetime_features:
        existing = [f for f in new_datetime_features if f in X_train_sel.columns]
        if existing:
            logger.info(f"🏷️ Casteando features datetime a category: {existing}")
            for col in existing:
                try:
                    X_train_sel[col] = X_train_sel[col].astype("category")
                    if col in X_val_sel.columns:
                        X_val_sel[col] = X_val_sel[col].astype("category")
                    if col in X_test_sel.columns:
                        X_test_sel[col] = X_test_sel[col].astype("category")
                except Exception as e:
                    logger.error(f"❌ Error al castear feature '{col}' a category: {e}")
                    raise ValueError(f"No se pudo castear '{col}' a category.") from e

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
