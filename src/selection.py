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
    """Feature selection con extracción de features datetime y manteniendo nombres originales de columnas"""
    
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
    
    logger.info(f"⏰ Variables datetime detectadas: {temporaries}")
    
    # Detectar IDs usando nombres reales
    ids_to_drop = [c for c in original_columns if "_id" in str(c).lower()]
    logger.info(f"🧾 IDs detectados para eliminar: {ids_to_drop or '— ninguno —'}")
    
    # Pipeline steps - COMENZAR CON DATETIME FEATURES
    steps: list[tuple[str, object]] = []
    
    # 1. Extraer features de datetime PRIMERO (usar directamente temporaries)
    if temporaries:
        steps.append((
            "datetime_features",
            DatetimeFeatures(
                variables=temporaries,
                features_to_extract=["day_of_week", "quarter"],
                drop_original=True  # Eliminar columna original datetime
            )
        ))
        logger.info(f"⏰ Extrayendo día de semana y trimestre de: {temporaries}")
        
        # Pre-fit para detectar nuevas columnas que se crearán
        datetime_transformer = DatetimeFeatures(
            variables=temporaries,
            features_to_extract=["day_of_week", "quarter"],
            drop_original=False
        )
        X_train_temp = datetime_transformer.fit_transform(X_train)
        new_datetime_features = [col for col in X_train_temp.columns if col not in original_columns]
        logger.info(f"⏰ Nuevas features datetime: {new_datetime_features}")
        
        # Actualizar categoricals con las nuevas features datetime
        categoricals.extend(new_datetime_features)
    else:
        new_datetime_features = []
    
    # 2. Drop de IDs
    if ids_to_drop:
        steps.append(("drop_ids", DropFeatures(features_to_drop=ids_to_drop)))
    
    # 3. Listas usando nombres reales (ahora incluye nuevas features datetime)
    vars_for_constant = [c for c in (continuous + categoricals) if c not in ids_to_drop]
    cat_for_cramers = [c for c in categoricals if c not in ids_to_drop]
    num_for_pearson = [c for c in (continuous + discretes) if c not in ids_to_drop]
    
    logger.info(f"📦 Candidatas a casi-constantes: {len(vars_for_constant)}")
    logger.info(f"🔣 Categóricas para Cramér's V: {len(cat_for_cramers)} (incluye datetime)")
    logger.info(f"📈 Numéricas para Pearson: {len(num_for_pearson)}")
    
    # 4. Drop constantes
    steps.append((
        "drop_constant",
        DropConstantFeatures(
            variables=vars_for_constant if vars_for_constant else None,
            missing_values="ignore",
            tol=0.95,
        ),
    ))
    
    # 5. Drop categóricas correlacionadas
    if cat_for_cramers:
        steps.append((
            "drop_cat_correlated",
            CramersVCorrelatedSelection(
                variables=cat_for_cramers,
                threshold=0.80,
                selection_method="first",
            ),
        ))
    
    # 6. Drop numéricas correlacionadas
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
    
    # 7. Crear pipeline
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
    
    # Resultado final: usar pipeline transformado directamente
    X_train_sel = X_tmp.reset_index(drop=True)
    X_val_sel = pipe.transform(X_val).reset_index(drop=True)
    X_test_sel = pipe.transform(X_test).reset_index(drop=True)
    
    # ✨ NUEVO: Castear features datetime a categorical
    if new_datetime_features:
        existing_datetime_features = [f for f in new_datetime_features if f in X_train_sel.columns]
        if existing_datetime_features:
            logger.info(f"🏷️ Casteando features datetime a category: {existing_datetime_features}")
            
            # Castear en todos los datasets
            for col in existing_datetime_features:
                X_train_sel[col] = X_train_sel[col].astype('category')
                if col in X_val_sel.columns:
                    X_val_sel[col] = X_val_sel[col].astype('category')
                if col in X_test_sel.columns:
                    X_test_sel[col] = X_test_sel[col].astype('category')
            
            # Log de tipos finales
            for col in existing_datetime_features:
                if col in X_train_sel.columns:
                    logger.info(f"   {col}: {X_train_sel[col].dtype}")
    
    logger.success(
        f"🎯 Feature selection completa. "
        f"Train={X_train_sel.shape}, Val={X_val_sel.shape}, Test={X_test_sel.shape}"
    )
    logger.info(f"🏷️ Columnas finales: {list(X_train_sel.columns)}")
    logger.info(f"⏰ Features datetime finales: {[f for f in new_datetime_features if f in X_train_sel.columns]}")
    
    return X_train_sel, X_val_sel, X_test_sel
