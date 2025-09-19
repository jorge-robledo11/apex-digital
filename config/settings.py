"""
Configuración centralizada de la aplicación.
"""

import sys
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .logger import LoggerSettings, LoggerProtocol


class Settings(BaseSettings):
    """Configuración global de la aplicación.

    Atributos:
        logger_cfg: Configuración del logger (cargable por variables de entorno).
        logger: Instancia de logger conforme a LoggerProtocol (se inicializa en `get_settings`).
    """

    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",
    )

    logger_cfg: LoggerSettings = Field(default_factory=LoggerSettings)
    logger: LoggerProtocol | None = None


# Singleton
_settings_instance: Settings | None = None


def get_settings(**overrides: Any) -> Settings:
    """Devuelve la instancia singleton de `Settings`.

    Inicializa `settings.logger` llamando una única vez a
    `LoggerSettings.setup_logger()`.

    Args:
        **overrides: Claves y valores para sobreescribir campos de `Settings`
            en la construcción inicial (por ejemplo, `logger_cfg=...`).

    Returns:
        Settings: Instancia única con la configuración cargada.

    Raises:
        SystemExit: Si ocurre un error crítico al cargar la configuración
            antes de que el logger esté disponible.
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
