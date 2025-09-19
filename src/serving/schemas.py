from typing import Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict

class PredictRecord(BaseModel):
    """Schema robusto con validación estricta"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    # Categóricas - validadas por Pydantic
    pais_cd: Literal["GT", "EC", "PE", "SV"] = Field(..., description="Código del país")
    tipo_cliente_cd: Literal["TIENDA", "MINIMARKET", "MAYORISTA"] = Field(..., description="Tipo de cliente")
    madurez_digital_cd: Literal["BAJA", "MEDIA", "ALTA"] = Field(..., description="Madurez digital")
    frecuencia_visitas_cd: Literal["LM", "L", "LMI", "LMV"] = Field(..., description="Frecuencia de visitas")
    
    # Numéricos - validados por Pydantic
    estrellas_txt: Literal[1, 2, 3] = Field(..., description="Rating de estrellas")
    cajas_fisicas: float = Field(..., ge=0, description="Número de cajas físicas")
    fecha_pedido_dt_day_of_week: int = Field(..., ge=0, le=6, description="Día de la semana")
    fecha_pedido_dt_quarter: int = Field(..., ge=1, le=4, description="Trimestre")
    
    @field_validator('estrellas_txt', mode='before')
    @classmethod
    def validate_estrellas(cls, v):
        if isinstance(v, (float, str)):
            try:
                return int(float(v))
            except (ValueError, TypeError):
                raise ValueError(f"estrellas_txt debe ser 1, 2 o 3, recibido: {v}")
        return v

class PredictPayload(BaseModel):
    records: list[PredictRecord] = Field(..., min_length=1, max_length=100)

class PredictionResult(BaseModel):
    probabilities: list[float] = Field(..., description="[DIGITAL, TELEFONO, VENDEDOR]")
    predicted_class: str = Field(..., description="Clase predicha")
    confidence: float = Field(..., description="Probabilidad máxima")

class PredictResponse(BaseModel):
    predictions: list[PredictionResult]
    model_version: str | None = None
    processing_time_ms: float | None = None
