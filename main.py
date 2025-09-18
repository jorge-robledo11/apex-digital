from pathlib import Path
from config.settings import get_settings
from src.utils.elt import ELTConfig, ELTPipeline

def main() -> None:
    settings = get_settings()
    logger = settings.logger

    # Rutas
    raw_root = Path("data/raw")
    output_root = Path("data/processed")

    # Config sin required_cols
    config = ELTConfig(
        raw_root=raw_root,
        output_root=output_root,
        dtype_map={"estrellas_txt": "float64"},
    )

    # Ejecutar pipeline
    try:
        pipeline = ELTPipeline(config)
        df_processed = pipeline.run()
        logger.success(f"🎯 Pipeline ejecutada con éxito. Registros procesados: {len(df_processed):,}")
    except Exception as e:
        logger.error(f"❌ Error al ejecutar pipeline: {e}")
        raise

if __name__ == "__main__":
    main()
