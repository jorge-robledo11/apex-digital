from dataclasses import dataclass
from pathlib import Path
import fireducks.pandas as pd
from config.settings import get_settings

# Obtener configuración global y logger
settings = get_settings()
logger = settings.logger

# ---------------------------
# Configuración declarativa
# ---------------------------
@dataclass(frozen=True)
class ELTConfig:
    raw_root: Path
    output_root: Path
    coerce_dtypes: bool = True
    fail_if_empty: bool = True
    dtype_map: dict[str, str] | None = None

# ---------------------------
# Funciones puras (reusables)
# ---------------------------
def read_parquet_folder(path: Path, *, verbose: bool = True) -> pd.DataFrame:
    """Carga y concatena todos los .parquet del folder, eliminando duplicados por archivo."""
    archivos: list[Path] = list(path.glob("*.parquet"))
    if not archivos:
        msg = f"No se encontraron archivos parquet en '{path}'"
        if verbose:
            logger.error(f"❌ {msg}")
        return pd.DataFrame()

    if verbose:
        logger.info(f"📁 Archivos encontrados: {[p.name for p in archivos]}")

    dfs: list[pd.DataFrame] = []
    filas_total = 0
    for archivo in archivos:
        df = pd.read_parquet(archivo)
        df = df.drop_duplicates()
        dfs.append(df)
        filas_total += len(df)
        if verbose:
            logger.debug(f"{archivo.name}: {df.shape[0]:,} filas × {df.shape[1]} columnas")

    if verbose:
        logger.info("=" * 50)
        logger.info("CONCATENANDO ARCHIVOS")
        logger.info("=" * 50)

    df_final = pd.concat(dfs, ignore_index=True)

    if verbose:
        logger.info(f"Total esperado: {filas_total:,} filas")
        logger.info(f"Dataset final: {df_final.shape[0]:,} filas × {df_final.shape[1]} columnas")
        if len(df_final) == filas_total:
            logger.success("✅ Concatenación exitosa - todas las filas conservadas")
        else:
            logger.warning("⚠️ ALERTA: Se perdieron filas en la concatenación")

        if "cliente_id" in df_final.columns:
            clientes_unicos = df_final["cliente_id"].nunique()
            total_filas = len(df_final)
            logger.info(f"Clientes únicos: {clientes_unicos:,}")
            logger.info(f"Total registros: {total_filas:,}")
            logger.info(f"Promedio registros por cliente: {total_filas/clientes_unicos:.1f}")

    return df_final


def validate_and_cast(df: pd.DataFrame, config: ELTConfig) -> pd.DataFrame:
    """Valida mínimamente y castea tipos (incluye fechas)."""
    if df.empty:
        if config.fail_if_empty:
            logger.error("❌ El DataFrame de entrada está vacío.")
            raise ValueError("El DataFrame de entrada está vacío.")
        logger.warning("⚠️ DataFrame vacío recibido, se continúa sin procesamiento.")
        return df

    if config.coerce_dtypes and config.dtype_map:
        for col, dtype in config.dtype_map.items():
            if col in df.columns:
                try:
                    if dtype.startswith("float"):
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    else:
                        df[col] = df[col].astype(dtype)
                except Exception as e:
                    logger.warning(f"No se pudo castear '{col}' a {dtype}. Error: {e}")
                    df[col] = pd.to_numeric(df[col], errors="ignore")

    if "fecha_pedido_dt" in df.columns:
        df["fecha_pedido_dt"] = pd.to_datetime(df["fecha_pedido_dt"], errors="coerce")

    logger.success("✅ Validación y casting completados")
    return df


def aggregate_to_customer(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega a nivel cliente siguiendo tu lógica original (último canal, sumas de negocio, etc.)."""
    if df.empty:
        logger.warning("⚠️ DataFrame vacío en etapa de agregación")
        return df

    sort_cols: list[str] = [c for c in ["cliente_id", "fecha_pedido_dt"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    agregaciones: dict[str, str] = {
        "pais_cd": "first",
        "region_comercial_txt": "first",
        "agencia_id": "first",
        "ruta_id": "first",
        "tipo_cliente_cd": "first",
        "madurez_digital_cd": "first",
        "estrellas_txt": "first",
        "frecuencia_visitas_cd": "first",
        "facturacion_usd_val": "sum",
        "materiales_distintos_val": "sum",
        "cajas_fisicas": "sum",
        "fecha_pedido_dt": "max",
        "canal_pedido_cd": "last",
    }

    out = df.groupby("cliente_id").agg(agregaciones).reset_index()
    out = out.drop_duplicates(subset=["cliente_id"], keep="last")

    logger.success(f"✅ Agregación completada: {len(out):,} clientes únicos")
    return out


def write_processed(df: pd.DataFrame, config: ELTConfig) -> Path:
    """Escribe el dataset procesado en un único parquet fijo."""
    base = config.output_root
    base.mkdir(parents=True, exist_ok=True)

    out_path = base / "data_final.parquet"
    df.to_parquet(out_path, index=False)

    logger.info(f"💾 Archivo procesado guardado en: {out_path}")
    return out_path


class ELTPipeline:
    def __init__(self, config: ELTConfig):
        self.config = config
        self._df_raw: pd.DataFrame | None = None
        self._df_valid: pd.DataFrame | None = None
        self._df_agg: pd.DataFrame | None = None

    def run(self) -> pd.DataFrame:
        logger.info("🚀 Iniciando pipeline ELT")
        self._df_raw = read_parquet_folder(self.config.raw_root)
        self._df_valid = validate_and_cast(self._df_raw, self.config)
        self._df_agg = aggregate_to_customer(self._df_valid)
        write_processed(self._df_agg, self.config)
        logger.success("🎉 Pipeline ELT completada")
        return self._df_agg
