import numpy as np
import pandas as pd
from .registry import ModelBundle


class ModelService:
    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle

    @property
    def class_names(self) -> list[str] | None:
        return self.bundle.class_names

    @property
    def input_cols(self) -> list[str] | None:
        return self.bundle.input_cols

    @property
    def categorical_cols(self) -> list[str]:
        return self.bundle.categorical_cols

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        # mlflow.pyfunc para XGBoost multiclass devuelve proba
        return self.bundle.pyfunc_model.predict(df)

    def predict_labels(self, proba: np.ndarray) -> tuple[list[int], list[str] | None]:
        idx = np.argmax(proba, axis=1).tolist()
        labels = [self.class_names[i] for i in idx] if self.class_names else None
        return idx, labels
