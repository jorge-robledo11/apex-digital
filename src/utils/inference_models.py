"""
Esquemas de datos para la inferencia asíncrona (Pydantic).

- InferenceResult: resultado unitario de inferencia.
- ConfidenceStats: resumen de confianza en batch.
- BatchSummary: métricas agregadas del batch.
- TestSampleMetadata: metadatos de la muestra de prueba.
"""

from typing import Any
from math import isfinite

from pydantic import BaseModel, Field, ConfigDict, field_validator


class InferenceResult(BaseModel):
    """Resultado de inferencia individual."""
    model_config = ConfigDict(frozen=True)

    index: int = Field(..., ge=0)
    input_data: dict[str, Any]
    probabilities: list[float]
    predicted_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time_ms: float = Field(..., ge=0.0)
    model_version: str | None = None

    @field_validator("probabilities")
    @classmethod
    def _validate_probs(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("probabilities no puede estar vacío.")
        if any(p is None or not isfinite(p) or p < 0.0 for p in v):
            raise ValueError("probabilities debe contener números reales ≥ 0.")
        return v


class ConfidenceStats(BaseModel):
    """Estadísticos de confianza en un batch."""
    model_config = ConfigDict(frozen=True)

    min_confidence: float = Field(..., ge=0.0, le=1.0)
    max_confidence: float = Field(..., ge=0.0, le=1.0)
    avg_confidence: float = Field(..., ge=0.0, le=1.0)
    std_confidence: float = Field(..., ge=0.0)


class BatchSummary(BaseModel):
    """Métricas agregadas para un batch de inferencia."""
    model_config = ConfigDict(frozen=True)

    total_samples: int = Field(..., ge=0)
    successful_predictions: int = Field(..., ge=0)
    success_rate: float = Field(..., ge=0.0, le=100.0)
    total_time_ms: float = Field(..., ge=0.0)
    avg_time_per_prediction_ms: float = Field(..., ge=0.0)
    prediction_distribution: dict[str, int]
    confidence_stats: ConfidenceStats | None = None
    model_version: str | None = None


class TestSampleMetadata(BaseModel):
    """Metadatos de la muestra de test utilizada en inferencia offline."""
    model_config = ConfigDict(frozen=True)

    sample_size: int = Field(..., ge=1)
    ground_truth_distribution: dict[str, int]
    sample_indices: list[int]
    columns_order: list[str]
