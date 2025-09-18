"""
Configuración centralizada de la aplicación.
"""

import sys
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from .logger import LoggerSettings, LoggerProtocol


class Settings(BaseSettings):
    """
    Configuración global de la aplicación.
    - logger_cfg: configuración del logger (cargable por env vars).
    - logger: instancia REAL de Loguru (se inicializa en get_settings()).
    """
    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra='ignore',
        env_nested_delimiter='__',
    )

    logger_cfg: LoggerSettings = Field(default_factory=LoggerSettings)
    logger: LoggerProtocol | None = None


# Singleton
_settings_instance: Settings | None = None


def get_settings(**overrides) -> Settings:
    """
    Devuelve el singleton de Settings y configura `settings.logger`
    usando LoggerSettings.setup_logger() una sola vez.
    """
    global _settings_instance
    if _settings_instance is None:
        try:
            _settings_instance = Settings(**overrides)
            _settings_instance.logger = _settings_instance.logger_cfg.setup_logger()
            _settings_instance.logger.success("✅ Configuración cargada y validada con éxito.")
        except Exception as e:
            # Fallback si algo falla antes de configurar Loguru
            print("❌ Error Crítico al cargar configuración.", file=sys.stderr)
            print(f"Detalles: {e}", file=sys.stderr)
            sys.exit(1)
    return _settings_instance
