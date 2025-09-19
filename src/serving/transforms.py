import pandas as pd
from fastapi import HTTPException
from .schemas import PredictPayload


def payload_to_dataframe(payload: PredictPayload) -> pd.DataFrame:
    if payload.records is not None:
        return pd.DataFrame(payload.records)
    if payload.columns is not None:
        return pd.DataFrame(payload.columns)
    raise HTTPException(status_code=400, detail="Provee 'records' o 'columns'.")


def align_and_cast_dataframe(
    df: pd.DataFrame,
    input_cols: list[str] | None,
    categorical_cols: list[str],
) -> pd.DataFrame:
    # Validar/ordenar columnas esperadas por signature
    if input_cols:
        missing = [c for c in input_cols if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Faltan columnas: {missing}")
        df = df[input_cols]

    # Casteo a category (XGBoost enable_categorical=True)
    for c in categorical_cols:
        if c in df.columns:
            df[c] = pd.Categorical(df[c])

    return df
