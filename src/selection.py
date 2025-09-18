import pandas as pd
from sklearn.pipeline import Pipeline
from feature_engine.selection import (
    DropFeatures, DropConstantFeatures, DropCorrelatedFeatures
)

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
    """Feature selection manteniendo nombres originales de columnas"""
    
    # Preservar nombres originales ANTES de cualquier conversión
    original_columns = list(X_train.columns)
    
    # Crear copias SIN forzar conversión de columnas
    X_train = X_train.to_pandas().copy() if hasattr(X_train, 'to_pandas') else X_train.copy()
    X_val = X_val.to_pandas().copy() if hasattr(X_val, 'to_pandas') else X_val.copy()
    X_test = X_test.to_pandas().copy() if hasattr(X_test, 'to_pandas') else X_test.copy()
    
    # IMPORTANTE: Asegurar que las columnas mantengan sus nombres originales
    X_train.columns = original_columns
    X_val.columns = [c for c in original_columns if c in X_val.columns]
    X_test.columns = [c for c in original_columns if c in X_test.columns]
    
    logger.info(f"🏷️ Columnas originales: {original_columns}")
    
    # Capturar variables usando nombres reales
    continuous, categoricals, discretes, temporaries = capture_variables(data=X_train)
    
    # Detectar IDs usando nombres reales
    ids_to_drop = [c for c in original_columns if "_id" in str(c).lower()]
    logger.info(f"🧾 IDs detectados para eliminar: {ids_to_drop or '— ninguno —'}")
    
    # Listas usando nombres reales
    vars_for_constant = [c for c in (continuous + categoricals) if c not in ids_to_drop]
    cat_for_cramers = [c for c in categoricals if c not in ids_to_drop]
    num_for_pearson = [c for c in (continuous + discretes) if c not in ids_to_drop]
    
    logger.info(f"📦 Candidatas a casi-constantes: {len(vars_for_constant)}")
    logger.info(f"🔣 Categóricas para Cramér's V: {len(cat_for_cramers)}")
    logger.info(f"📈 Numéricas para Pearson: {len(num_for_pearson)}")
    
    # Pipeline steps
    steps: list[tuple[str, object]] = []
    if ids_to_drop:
        steps.append(("drop_ids", DropFeatures(features_to_drop=ids_to_drop)))
    
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
    
    # Tracking de eliminadas por paso
    logger.info("🛠️ Ajustando pipeline de selección…")
    dropped_ids, dropped_const, dropped_cat, dropped_num = [], [], [], []
    
    # Fit y track manual para logging correcto
    X_tmp = X_train.to_pandas().copy() if hasattr(X_train, 'to_pandas') else X_train.copy()
    for name, transformer in pipe.steps:
        cols_before = list(X_tmp.columns)
        transformer.fit(X_tmp)
        X_tmp = transformer.transform(X_tmp)
        cols_after = list(X_tmp.columns)
        
        # Calcular eliminadas por diferencia de nombres
        dropped_this_step = [c for c in cols_before if c not in cols_after]
        
        if name == "drop_ids":
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
    
    # Resultado final usando nombres reales
    all_dropped = set(dropped_ids + dropped_const + dropped_cat + dropped_num)
    final_columns = [c for c in original_columns if c not in all_dropped]
    
    X_train_sel = X_train[final_columns].reset_index(drop=True)
    X_val_sel = X_val[[c for c in final_columns if c in X_val.columns]].reset_index(drop=True)
    X_test_sel = X_test[[c for c in final_columns if c in X_test.columns]].reset_index(drop=True)
    
    logger.success(
        f"🎯 Feature selection completa. "
        f"Train={X_train_sel.shape}, Val={X_val_sel.shape}, Test={X_test_sel.shape}"
    )
    logger.info(f"🏷️  Columnas finales: {list(X_train_sel.columns)}")
    
    return X_train_sel, X_val_sel, X_test_sel

