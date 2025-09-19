from mlflow.tracking import MlflowClient
import mlflow

def ensure_experiment(name: str, artifact_location: str | None = None) -> str:
    """
    Devuelve el experiment_id de `name`. Si no existe lo crea,
    y si está 'deleted' lo restaura. Idempotente.
    """
    client = MlflowClient()
    exp = client.get_experiment_by_name(name)
    if exp is None:
        # crear nuevo
        if artifact_location:
            experiment_id = client.create_experiment(name=name, artifact_location=artifact_location)
        else:
            experiment_id = client.create_experiment(name=name)
        mlflow.set_experiment(experiment_name=name)
        return experiment_id

    if exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)

    # activar (por nombre) y devolver id
    mlflow.set_experiment(experiment_name=name)
    return exp.experiment_id
