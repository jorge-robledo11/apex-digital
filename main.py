from pathlib import Path
from re import X
from config.settings import get_settings
from src.utils.elt import ELTConfig, ELTPipeline
from src.split import split_train_val_test
from src.selection import feature_selection


def main() -> None:
    settings = get_settings()
    logger = settings.logger

    # -------------------------------
    # 1) Procesar datos crudos → processed
    # -------------------------------
    raw_root = Path("data/raw")
    output_root = Path("data/processed")

    config = ELTConfig(
        raw_root=raw_root,
        output_root=output_root,
        dtype_map={"estrellas_txt": "float64"},
    )

    try:
        elt_pipeline = ELTPipeline(config)
        df_processed = elt_pipeline.run()
        logger.success(
            f"🎯 Pipeline ELT ejecutada con éxito. Registros procesados: {len(df_processed):,}"
        )
    except Exception as e:
        logger.error(f"❌ Error al ejecutar pipeline ELT: {e}")
        raise

    # -------------------------------
    # 2) Split train/val/test
    # -------------------------------
    target = "canal_pedido_cd"
    seed = 42

    try:
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
            data=df_processed, target=target, seed=seed
        )
        logger.success(
            f"📊 División realizada: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}"
        )
    except Exception as e:
        logger.error(f"❌ Error en split de datos: {e}")
        raise

    # -------------------------------
    # 3) Feature Selection (sin parámetro 'features')
    #    La función ya detecta automáticamente IDs ('_id'), constantes,
    #    categóricas redundantes (Cramér’s V) y continuas correlacionadas (Pearson).
    # -------------------------------
    try:
        X_train_sel, X_val_sel, X_test_sel = feature_selection(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
        )
        logger.success(
            f"✨ Feature selection completa. "
            f"Train={X_train_sel.shape}, Val={X_val_sel.shape}, Test={X_test_sel.shape}"
        )
    except Exception as e:
        logger.error(f"❌ Error en feature selection: {e}")
        raise

    print(X_train_sel.head())
    
if __name__ == "__main__":
    main()
