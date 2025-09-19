import os
from dataclasses import dataclass
from typing import Any
import mlflow
from mlflow.tracking import MlflowClient


@dataclass
class ModelBundle:
    uri: str
    pyfunc_model: Any
    class_names: list[str] | None
    categorical_cols: list[str]
    input_cols: list[str] | None


def _resolve_model_uri(model_name: str, preferred_stage: str | None) -> str:
    client = MlflowClient()
    if preferred_stage and preferred_stage.lower() != "none":
        latest = client.get_latest_versions(model_name, [preferred_stage])
        if latest:
            return f"models:/{model_name}/{preferred_stage}"
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(f"No hay versiones para el modelo '{model_name}'")
    latest = max(versions, key=lambda v: int(v.version))
    return f"models:/{model_name}/{latest.version}"


def _get_model_version_obj(client: MlflowClient, model_name: str, uri: str):
    # Devuelve el objeto ModelVersion para poder leer run_id
    if uri.endswith(("Production", "Staging")):
        stage = uri.split("/")[-1]
        mv_list = client.get_latest_versions(model_name, [stage])
        return mv_list[0] if mv_list else None
    version = uri.split("/")[-1]
    return client.get_model_version(model_name, version)


def _safe_read_artifact_txt(client: MlflowClient, run_id: str, path: str) -> list[str] | None:
    try:
        local = client.download_artifacts(run_id, path)
        with open(local, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        return lines
    except Exception:
        return None


def _get_input_cols_from_signature(uri: str) -> list[str] | None:
    try:
        info = mlflow.models.get_model_info(uri)
        if info and info.signature and info.signature.inputs:
            return [f.name for f in info.signature.inputs.inputs]
    except Exception:
        pass
    return None


def load_model_bundle(
    model_name: str | None = None,
    preferred_stage: str | None = None,
    tracking_uri: str | None = None,
) -> ModelBundle:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_name = model_name or os.getenv("MODEL_NAME", "digital_orders_xgboost")
    preferred_stage = preferred_stage or os.getenv("MODEL_STAGE", "Production")

    uri = _resolve_model_uri(model_name, preferred_stage)
    pyfunc_model = mlflow.pyfunc.load_model(uri)

    client = MlflowClient()
    mv = _get_model_version_obj(client, model_name, uri)

    class_names: list[str] | None = None
    categorical_cols: list[str] = []

    if mv is not None:
        run_id = mv.run_id
        class_names = _safe_read_artifact_txt(client, run_id, "artifacts/class_names.txt")
        cat_cols = _safe_read_artifact_txt(client, run_id, "artifacts/categorical_features.txt")
        if cat_cols:
            categorical_cols = cat_cols

    input_cols = _get_input_cols_from_signature(uri)

    return ModelBundle(
        uri=uri,
        pyfunc_model=pyfunc_model,
        class_names=class_names,
        categorical_cols=categorical_cols,
        input_cols=input_cols,
    )
