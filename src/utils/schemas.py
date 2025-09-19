"""
Esquemas y constantes para el pipeline ELT.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# ---------------------------
# Tipos y constantes
# ---------------------------
AggFunc = Literal["first", "sum", "max", "last"]  # tipo de agregación permitido
AggregationSpec = dict[str, AggFunc]

DEFAULT_AGGREGATIONS: AggregationSpec = {
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

RAW_GLOB_PATTERN = "*.parquet"
OUTPUT_FILENAME = "data_final.parquet"

# ---------------------------
# Configuración declarativa
# ---------------------------
@dataclass(frozen=True)
class ELTConfig:
    """Configuración inmutable para el pipeline ELT.

    Atributos:
        raw_root: Carpeta con archivos de entrada (.parquet).
        output_root: Carpeta de salida para el resultado procesado.
        coerce_dtypes: Si se deben forzar tipos de columnas.
        fail_if_empty: Si debe fallar ante entradas vacías.
        dtype_map: Mapeo columna→dtype para casteo selectivo.
    """
    raw_root: Path
    output_root: Path
    coerce_dtypes: bool = True
    fail_if_empty: bool = True
    dtype_map: dict[str, str] | None = None
