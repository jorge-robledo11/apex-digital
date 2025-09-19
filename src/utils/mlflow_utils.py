"""
Utilidades para gestión idempotente de experimentos en MLflow.
"""

import mlflow
from mlflow.tracking import MlflowClient


def ensure_experiment(name: str, artifact_location: str | None = None) -> str:
    """
    Garantiza que exista un experimento con `name` y lo selecciona como activo.

    Si no existe, lo crea; si está eliminado, lo restaura. Operación idempotente.

    Args:
        name: Nombre del experimento.
        artifact_location: Ubicación de artefactos para creación inicial.

    Returns:
        str: `experiment_id` del experimento solicitado.

    Raises:
        mlflow.exceptions.MlflowException: Errores al crear, restaurar o consultar.
    """
    client = MlflowClient()
    exp = client.get_experiment_by_name(name)

    if exp is None:
        if artifact_location:
            experiment_id = client.create_experiment(
                name=name,
                artifact_location=artifact_location,
            )
        else:
            experiment_id = client.create_experiment(name=name)
        mlflow.set_experiment(experiment_name=name)
        return experiment_id

    if exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)

    mlflow.set_experiment(experiment_name=name)
    return exp.experiment_id
