"""
Modelos y utilidades para métricas de entrenamiento y evaluación.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator, ConfigDict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    confusion_matrix,
)


# =========================
# Modelos de datos (Pydantic)
# =========================
class LossMetrics(BaseModel):
    """Métricas de pérdida multi-clase."""
    model_config = ConfigDict(frozen=True)

    train_logloss: float = Field(..., ge=0.0)
    val_logloss: float = Field(..., ge=0.0)
    test_logloss: float = Field(..., ge=0.0)


class AccuracyMetrics(BaseModel):
    """Exactitud por split."""
    model_config = ConfigDict(frozen=True)

    train_accuracy: float = Field(..., ge=0.0, le=1.0)
    val_accuracy: float = Field(..., ge=0.0, le=1.0)
    test_accuracy: float = Field(..., ge=0.0, le=1.0)


class ClassificationTextReports(BaseModel):
    """Reportes de clasificación en texto plano (sklearn.classification_report)."""
    model_config = ConfigDict(frozen=True)

    val_report: str
    test_report: str


class ConfusionMatrixBundle(BaseModel):
    """Matriz de confusión y etiquetas asociadas."""
    model_config = ConfigDict(frozen=True)

    labels: list[str]
    val_confusion: list[list[int]]
    test_confusion: list[list[int]]

    @field_validator("val_confusion", "test_confusion")
    @classmethod
    def _non_empty(cls, v: list[list[int]]) -> list[list[int]]:
        if not v:
            raise ValueError("La matriz de confusión no puede estar vacía.")
        return v


class DatasetShapeInfo(BaseModel):
    """Tamaños de splits y número de features."""
    model_config = ConfigDict(frozen=True)

    n_train: int = Field(..., ge=1)
    n_val: int = Field(..., ge=1)
    n_test: int = Field(..., ge=1)
    n_features: int = Field(..., ge=1)


class TrainingFlags(BaseModel):
    """Flags relevantes usados en entrenamiento."""
    model_config = ConfigDict(frozen=True)

    early_stopping_rounds: int = Field(..., ge=1)
    used_balanced_weights: bool
    used_gpu: bool
    categorical_features: bool = True
    encoded_target: bool = True


class XGBHyperparams(BaseModel):
    """
    Hiperparámetros relevantes del modelo final.

    Si algún valor llega como ``None`` desde el estimador (común en ciertas
    versiones de XGBoost tras el fit), se aplican **por defecto** los valores
    oficiales de XGBoost antes de validar:
      - n_estimators=100, learning_rate=0.3, max_depth=6, min_child_weight=1.0
      - subsample=1.0, colsample_bytree=1.0
      - reg_lambda=1.0, reg_alpha=0.0
      - tree_method="auto", predictor="auto"
      - objective="binary:logistic" (si no se especificó)
    """
    model_config = ConfigDict(frozen=True)

    objective: str
    eval_metric: str = "mlogloss"
    enable_categorical: bool = True  # tu flujo lo usa en True
    n_estimators: int
    learning_rate: float
    max_depth: int
    min_child_weight: float | int
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    reg_alpha: float
    tree_method: str
    predictor: str

    # Normalizaciones a defaults de XGBoost (se ejecutan ANTES de validar tipo)
    @field_validator("n_estimators", mode="before")
    @classmethod
    def _def_n_estimators(cls, v: int | None) -> int:
        return 100 if v is None else v

    @field_validator("learning_rate", mode="before")
    @classmethod
    def _def_learning_rate(cls, v: float | None) -> float:
        return 0.3 if v is None else v

    @field_validator("max_depth", mode="before")
    @classmethod
    def _def_max_depth(cls, v: int | None) -> int:
        return 6 if v is None else v

    @field_validator("min_child_weight", mode="before")
    @classmethod
    def _def_min_child_weight(cls, v: float | int | None) -> float | int:
        return 1.0 if v is None else v

    @field_validator("subsample", mode="before")
    @classmethod
    def _def_subsample(cls, v: float | None) -> float:
        return 1.0 if v is None else v

    @field_validator("colsample_bytree", mode="before")
    @classmethod
    def _def_colsample_bytree(cls, v: float | None) -> float:
        return 1.0 if v is None else v

    @field_validator("reg_lambda", mode="before")
    @classmethod
    def _def_reg_lambda(cls, v: float | None) -> float:
        return 1.0 if v is None else v

    @field_validator("reg_alpha", mode="before")
    @classmethod
    def _def_reg_alpha(cls, v: float | None) -> float:
        return 0.0 if v is None else v

    @field_validator("tree_method", mode="before")
    @classmethod
    def _def_tree_method(cls, v: str | None) -> str:
        return "auto" if v in (None, "unknown") else v

    @field_validator("predictor", mode="before")
    @classmethod
    def _def_predictor(cls, v: str | None) -> str:
        return "auto" if v in (None, "unknown") else v

    @field_validator("objective", mode="before")
    @classmethod
    def _def_objective(cls, v: str | None) -> str:
        return "binary:logistic" if v in (None, "unknown") else v


class BestIterationInfo(BaseModel):
    """Información de mejor iteración (si aplica)."""
    model_config = ConfigDict(frozen=True)

    best_iteration: int | None = None
    best_val_mlogloss: float | None = None


class TrainingMetrics(BaseModel):
    """Paquete completo de métricas y metadatos del entrenamiento."""
    model_config = ConfigDict(frozen=True)

    shapes: DatasetShapeInfo
    flags: TrainingFlags
    losses: LossMetrics
    accuracies: AccuracyMetrics
    reports: ClassificationTextReports
    confusion: ConfusionMatrixBundle | None = None
    xgb_hyperparams: XGBHyperparams | None = None
    best_iter_info: BestIterationInfo | None = None

    def to_flat_metrics(self) -> dict[str, float]:
        """Convierte a un dict plano (keys pensadas para mlflow.log_metrics)."""
        out: dict[str, float] = {
            "train_logloss": self.losses.train_logloss,
            "val_logloss": self.losses.val_logloss,
            "test_logloss": self.losses.test_logloss,
            "train_accuracy": self.accuracies.train_accuracy,
            "val_accuracy": self.accuracies.val_accuracy,
            "test_accuracy": self.accuracies.test_accuracy,
        }
        if self.best_iter_info and self.best_iter_info.best_iteration is not None:
            out["best_iteration"] = float(self.best_iter_info.best_iteration)
        if self.best_iter_info and self.best_iter_info.best_val_mlogloss is not None:
            out["best_val_mlogloss"] = self.best_iter_info.best_val_mlogloss
        return out

    def dumps_json(self) -> str:
        """Serializa el paquete completo a JSON (para artefactos)."""
        return self.model_dump_json(indent=2, by_alias=False)


# =========================
# Cálculo de métricas
# =========================
def compute_classification_metrics(
    y_train_enc: np.ndarray,
    y_val_enc: np.ndarray,
    y_test_enc: np.ndarray,
    y_train_pred: np.ndarray,
    y_val_pred: np.ndarray,
    y_test_pred: np.ndarray,
    y_train_proba: np.ndarray,
    y_val_proba: np.ndarray,
    y_test_proba: np.ndarray,
    class_names: list[str],
) -> tuple[LossMetrics, AccuracyMetrics, ClassificationTextReports, ConfusionMatrixBundle]:
    """Calcula logloss/accuracy, reports y matrices de confusión.

    Args:
        y_*_enc: Targets codificados (enteros desde 0..K-1) por split.
        y_*_pred: Predicciones (códigos enteros) por split.
        y_*_proba: Probabilidades (N×K) por split.
        class_names: Nombres de clases en el mismo orden de los códigos.

    Returns:
        tuple: (LossMetrics, AccuracyMetrics, ClassificationTextReports, ConfusionMatrixBundle)

    Raises:
        ValueError: Si las dimensiones no coinciden o K no es consistente.
    """
    # Validaciones básicas
    k = len(class_names)
    for proba in (y_train_proba, y_val_proba, y_test_proba):
        if proba.ndim != 2 or proba.shape[1] != k:
            raise ValueError("Las probabilidades deben ser (N×K) y K coincidir con class_names.")
    labels_idx = list(range(k))

    # Pérdidas
    train_log = log_loss(y_train_enc, y_train_proba, labels=labels_idx)
    val_log = log_loss(y_val_enc, y_val_proba, labels=labels_idx)
    test_log = log_loss(y_test_enc, y_test_proba, labels=labels_idx)
    losses = LossMetrics(
        train_logloss=float(train_log),
        val_logloss=float(val_log),
        test_logloss=float(test_log),
    )

    # Exactitud
    acc = AccuracyMetrics(
        train_accuracy=float(accuracy_score(y_train_enc, y_train_pred)),
        val_accuracy=float(accuracy_score(y_val_enc, y_val_pred)),
        test_accuracy=float(accuracy_score(y_test_enc, y_test_pred)),
    )

    # Reportes
    reports = ClassificationTextReports(
        val_report=classification_report(y_val_enc, y_val_pred, target_names=class_names),
        test_report=classification_report(y_test_enc, y_test_pred, target_names=class_names),
    )

    # Matrices de confusión
    cm_val = confusion_matrix(y_val_enc, y_val_pred, labels=labels_idx)
    cm_test = confusion_matrix(y_test_enc, y_test_pred, labels=labels_idx)
    confusion = ConfusionMatrixBundle(
        labels=list(class_names),
        val_confusion=cm_val.tolist(),
        test_confusion=cm_test.tolist(),
    )

    return losses, acc, reports, confusion


# =========================
# Helpers para MLflow
# =========================
def log_training_metrics_to_mlflow(
    metrics: TrainingMetrics,
    *,
    mlflow_module: Any,
    reports_artifact_dir: str = "classification_reports",
    full_json_artifact: str = "metrics_bundle.json",
) -> None:
    """Loguea métricas y artefactos en MLflow a partir del paquete `TrainingMetrics`.

    Args:
        metrics: Paquete con métricas/flags/hparams/shapes.
        mlflow_module: Referencia al módulo `mlflow` ya importado.
        reports_artifact_dir: Directorio relativo para guardar reportes de texto.
        full_json_artifact: Nombre del artefacto JSON con el paquete completo.

    Raises:
        ValueError: Si `mlflow_module` no expone funciones esperadas.
    """
    if not hasattr(mlflow_module, "log_metrics") or not hasattr(mlflow_module, "log_text"):
        raise ValueError("El argumento mlflow_module no parece ser el módulo de MLflow.")

    # Métricas planas
    mlflow_module.log_metrics(metrics.to_flat_metrics())

    # Reportes
    mlflow_module.log_text(metrics.reports.val_report, f"{reports_artifact_dir}/val_report.txt")
    mlflow_module.log_text(metrics.reports.test_report, f"{reports_artifact_dir}/test_report.txt")

    # Paquete completo como JSON (útil para auditoría)
    mlflow_module.log_text(metrics.dumps_json(), full_json_artifact)
