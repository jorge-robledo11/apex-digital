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
    """Resuelve URI del modelo SIN usar APIs deprecadas"""
    client = MlflowClient()
    
    try:
        print(f"🔍 Buscando modelo: {model_name} (stage: {preferred_stage})")
        
        # Obtener todas las versiones del modelo
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise RuntimeError(f"No hay versiones para el modelo '{model_name}'")
        
        print(f"📋 Encontradas {len(versions)} versiones del modelo")
        
        # Si se especifica stage (Production/Staging), buscar versiones con ese stage
        if preferred_stage and preferred_stage.lower() not in ["none", "null"]:
            stage_versions = [v for v in versions if v.current_stage == preferred_stage]
            if stage_versions:
                # Usar la versión más reciente de ese stage
                latest = max(stage_versions, key=lambda v: int(v.version))
                print(f"✅ Usando modelo con stage '{preferred_stage}': versión {latest.version}")
                return f"models:/{model_name}/{latest.version}"
            else:
                print(f"⚠️  No se encontraron versiones con stage '{preferred_stage}', usando la más reciente")
        
        # Fallback: usar la versión más reciente sin importar el stage
        latest = max(versions, key=lambda v: int(v.version))
        print(f"✅ Usando versión más reciente: {latest.version}")
        return f"models:/{model_name}/{latest.version}"
        
    except Exception as e:
        print(f"❌ Error al resolver modelo '{model_name}': {e}")
        raise RuntimeError(f"Error al resolver modelo '{model_name}': {e}")

def _get_model_version_obj(client: MlflowClient, model_name: str, uri: str):
    """Devuelve el objeto ModelVersion SIN usar APIs deprecadas"""
    try:
        # Extraer version del URI
        version = uri.split("/")[-1]
        
        # Si termina en stage name, obtener versión actual
        if version in ["Production", "Staging"]:
            stage = version
            versions = client.search_model_versions(f"name='{model_name}'")
            stage_versions = [v for v in versions if v.current_stage == stage]
            if stage_versions:
                latest = max(stage_versions, key=lambda v: int(v.version))
                return latest
            return None
        
        # Si es número de versión directamente
        if version.isdigit():
            return client.get_model_version(model_name, version)
        
        return None
        
    except Exception as e:
        print(f"⚠️  Error obteniendo ModelVersion: {e}")
        return None

def _safe_read_artifact_txt(client: MlflowClient, run_id: str, path: str) -> list[str] | None:
    """Lee archivos de texto desde artifacts de forma segura"""
    try:
        print(f"📄 Intentando leer artifact: {path}")
        local = client.download_artifacts(run_id, path)
        with open(local, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        print(f"✅ Artifact leído exitosamente: {len(lines)} líneas")
        return lines
    except Exception as e:
        print(f"⚠️  No se pudo leer artifact {path}: {e}")
        return None

def _get_input_cols_from_signature(uri: str) -> list[str] | None:
    """Extrae nombres de columnas desde signature del modelo"""
    try:
        print(f"🔍 Extrayendo signature desde: {uri}")
        info = mlflow.models.get_model_info(uri)
        if info and info.signature and info.signature.inputs:
            cols = [f.name for f in info.signature.inputs.inputs]
            print(f"✅ Signature encontrada: {len(cols)} columnas")
            return cols
    except Exception as e:
        print(f"⚠️  Error extrayendo signature: {e}")
    
    print("⚠️  No se pudo extraer signature, usando None")
    return None

def load_model_bundle(
    model_name: str | None = None,
    preferred_stage: str | None = None,
    tracking_uri: str | None = None,
) -> ModelBundle:
    """Carga modelo y metadatos desde MLflow Model Registry"""
    
    print("🚀 Iniciando carga de ModelBundle...")
    
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"🔗 MLflow tracking URI configurado: {tracking_uri}")

    model_name = model_name or os.getenv("MODEL_NAME", "digital_orders_xgboost")
    preferred_stage = preferred_stage or os.getenv("MODEL_STAGE", "None")
    
    print(f"📋 Configuración:")
    print(f"   - Modelo: {model_name}")
    print(f"   - Stage preferido: {preferred_stage}")
    print(f"   - Tracking URI: {mlflow.get_tracking_uri()}")

    try:
        # Resolver URI del modelo
        uri = _resolve_model_uri(model_name, preferred_stage)
        print(f"🎯 URI resuelto: {uri}")
        
        # Cargar modelo
        print("📦 Cargando modelo...")
        pyfunc_model = mlflow.pyfunc.load_model(uri)
        print("✅ Modelo cargado exitosamente")

        # Obtener metadatos
        client = MlflowClient()
        mv = _get_model_version_obj(client, model_name, uri)

        class_names: list[str] | None = None
        categorical_cols: list[str] = []

        if mv is not None:
            run_id = mv.run_id
            print(f"📋 Cargando metadatos desde run: {run_id}")
            
            # Intentar cargar class names desde diferentes ubicaciones
            class_names = _safe_read_artifact_txt(client, run_id, "artifacts/class_names.txt")
            if not class_names:
                class_names = _safe_read_artifact_txt(client, run_id, "class_names.txt")
            
            # Si no se encuentran, usar defaults para tu problema específico
            if not class_names:
                class_names = ["DIGITAL", "TELEFONO", "VENDEDOR"]
                print("⚠️  Usando class_names por defecto: ['DIGITAL', 'TELEFONO', 'VENDEDOR']")
            
            # Intentar cargar categorical features
            cat_cols = _safe_read_artifact_txt(client, run_id, "artifacts/categorical_features.txt")
            if not cat_cols:
                cat_cols = _safe_read_artifact_txt(client, run_id, "categorical_features.txt")
            
            if cat_cols:
                categorical_cols = cat_cols
                print(f"✅ Features categóricas cargadas: {len(categorical_cols)}")
            else:
                # Defaults comunes para tu problema
                categorical_cols = [
                    "pais_cd", "tipo_cliente_cd", "madurez_digital_cd", 
                    "estrellas_txt", "frecuencia_visitas_cd",
                    "fecha_pedido_dt_day_of_week", "fecha_pedido_dt_quarter"
                ]
                print(f"⚠️  Usando categorical_cols por defecto: {len(categorical_cols)} columnas")
        else:
            print("⚠️  No se pudo obtener ModelVersion, usando defaults")
            class_names = ["DIGITAL", "TELEFONO", "VENDEDOR"]
            categorical_cols = [
                "pais_cd", "tipo_cliente_cd", "madurez_digital_cd", 
                "estrellas_txt", "frecuencia_visitas_cd",
                "fecha_pedido_dt_day_of_week", "fecha_pedido_dt_quarter"
            ]

        # Obtener columnas de input desde signature
        input_cols = _get_input_cols_from_signature(uri)

        print("🎉 ModelBundle creado exitosamente:")
        print(f"   - URI: {uri}")
        print(f"   - Classes: {class_names}")
        print(f"   - Input cols: {len(input_cols) if input_cols else 0}")
        print(f"   - Categorical cols: {len(categorical_cols)}")

        return ModelBundle(
            uri=uri,
            pyfunc_model=pyfunc_model,
            class_names=class_names,
            categorical_cols=categorical_cols,
            input_cols=input_cols,
        )
        
    except Exception as e:
        print(f"❌ Error fatal cargando modelo: {e}")
        raise RuntimeError(f"No se pudo cargar el modelo '{model_name}': {e}")
