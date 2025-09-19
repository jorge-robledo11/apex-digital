from typing import Any
from pydantic import BaseModel


class PredictPayload(BaseModel):
    records: list[dict[str, Any]] | None = None
    columns: dict[str, list[Any]] | None = None


class PredictResponse(BaseModel):
    predictions: list[dict[str, Any]] | None = None
    errors: list[str] | None = None
