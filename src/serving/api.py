import os
import mlflow
from fastapi import FastAPI, HTTPException

from .registry import load_model_bundle
from .schemas import PredictPayload
from .service import ModelService
from .transforms import payload_to_dataframe, align_and_cast_dataframe


def create_app() -> FastAPI:
    app = FastAPI(title="digital-orders-model", version="1.0.0")

    # Config MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5555")
    mlflow.set_tracking_uri(tracking_uri)

    # Cargar modelo + metadatos
    bundle = load_model_bundle(
        model_name=os.getenv("MODEL_NAME", "digital_orders_xgboost"),
        preferred_stage=os.getenv("MODEL_STAGE", "Production"),
        tracking_uri=tracking_uri,
    )
    service = ModelService(bundle)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "tracking_uri": tracking_uri,
            "model_uri": bundle.uri,
            "has_signature": service.input_cols is not None,
            "class_names": service.class_names,
        }

    @app.post("/predict_proba")
    def predict_proba(payload: PredictPayload):
        df = payload_to_dataframe(payload)
        df = align_and_cast_dataframe(df, service.input_cols, service.categorical_cols)
        try:
            proba = service.predict_proba(df)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {
            "class_names": service.class_names,
            "probabilities": proba.tolist(),
        }

    @app.post("/predict")
    def predict(payload: PredictPayload):
        df = payload_to_dataframe(payload)
        df = align_and_cast_dataframe(df, service.input_cols, service.categorical_cols)
        try:
            proba = service.predict_proba(df)
            idx, labels = service.predict_labels(proba)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {
            "preds": idx,
            "labels": labels or idx,  # si no hay nombres, devolvemos índices
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serving.api:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
