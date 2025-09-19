import pandas as pd
from fastapi import HTTPException
from .schemas import PredictPayload

def payload_to_dataframe(payload: PredictPayload) -> pd.DataFrame:
    """Convierte payload a DataFrame con tipos pandas correctos"""
    
    records_data = [record.model_dump() for record in payload.records]
    df = pd.DataFrame(records_data)
    
    print(f"🔍 DataFrame desde Pydantic (antes de categorical):")
    print(f"   Dtypes: {df.dtypes.to_dict()}")
    
    # ✅ TODAS las categóricas (incluyendo datetime features)
    categorical_mappings = {
        'pais_cd': ['GT', 'EC', 'PE', 'SV'],
        'tipo_cliente_cd': ['TIENDA', 'MINIMARKET', 'MAYORISTA'],
        'madurez_digital_cd': ['BAJA', 'MEDIA', 'ALTA'],
        'frecuencia_visitas_cd': ['LM', 'L', 'LMI', 'LMV'],
        'fecha_pedido_dt_day_of_week': [0, 1, 2, 3, 4, 5, 6],  # Como ints
        'fecha_pedido_dt_quarter': [1, 2, 3, 4]  # Como ints
    }
    
    # Aplicar pd.Categorical con categorías fijas
    for col, categories in categorical_mappings.items():
        if col in df.columns:
            if col in ['fecha_pedido_dt_day_of_week', 'fecha_pedido_dt_quarter']:
                # Para datetime: mantener como int pero hacer categorical
                df[col] = pd.Categorical(df[col], categories=categories)
            else:
                # Para strings: categorical directo
                df[col] = pd.Categorical(df[col], categories=categories)
    
    print(f"🎯 DataFrame después de categorical:")
    print(f"   Dtypes: {df.dtypes.to_dict()}")
    print(f"   Categorical columns: {[col for col in df.columns if df[col].dtype.name == 'category']}")
    
    return df

def align_and_cast_dataframe(
    df: pd.DataFrame,
    input_cols: list[str] | None,
    categorical_cols: list[str],
) -> pd.DataFrame:
    """Solo reordena columnas"""
    
    if input_cols:
        missing = [c for c in input_cols if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Faltan columnas: {missing}")
        df = df[input_cols]
    
    print(f"✅ DataFrame final para XGBoost:")
    print(f"   Dtypes: {df.dtypes.to_dict()}")
    print(f"   Categorical count: {len([col for col in df.columns if df[col].dtype.name == 'category'])}")
    
    return df
