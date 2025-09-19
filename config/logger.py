"""
Configuración centralizada para el logging de la aplicación.
"""

import sys
from pathlib import Path
from typing import Protocol, runtime_checkable, overload

from loguru import logger as _logger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@runtime_checkable
class LoggerProtocol(Protocol):
    # Métodos comunes
    def debug(self, __message: str, *args: object, **kwargs: object) -> None: ...
    def info(self, __message: str, *args: object, **kwargs: object) -> None: ...
    def success(self, __message: str, *args: object, **kwargs: object) -> None: ...
    def warning(self, __message: str, *args: object, **kwargs: object) -> None: ...
    def error(self, __message: str, *args: object, **kwargs: object) -> None: ...
    def exception(self, __message: str, *args: object, **kwargs: object) -> None: ...

    # Gestión de sinks
    def add(self, __sink: object, *args: object, **kwargs: object) -> int: ...

    @overload
    def remove(self) -> None: ...
    @overload
    def remove(self, __handler_id: int) -> None: ...


class LoggerSettings(BaseSettings):
    """Configuración centralizada para el logger de Loguru."""
    model_config = SettingsConfigDict(extra='ignore', case_sensitive=False)

    level: str = Field(default='DEBUG', validation_alias='LOG_LEVEL')
    colorize: bool = Field(default=True, validation_alias='LOG_COLORIZE')
    file_level: str = Field(default='DEBUG', validation_alias='LOG_FILE_LEVEL')
    rotation: str = Field(default='10 MB', validation_alias='LOG_ROTATION')
    retention: str = Field(default='7 days', validation_alias='LOG_RETENTION')

    console_format: str = Field(
        default=(
            '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
            '<level>{level: <8}</level> | '
            '<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - '
            '<level>{message}</level>'
        ),
        validation_alias='LOG_CONSOLE_FORMAT',
    )
    file_format: str = Field(
        default=(
            '{time:YYYY-MM-DD HH:mm:ss}|{level: <8}|{process.id}|'
            '{name}.{function}:{line}|{message}'
        ),
        validation_alias='LOG_FILE_FORMAT',
    )

    def setup_logger(self, log_dir: Path | None = None) -> LoggerProtocol:
        """
        Configura el logger global de Loguru usando los valores actuales.
        Devuelve la instancia de logger lista para usar (cumple LoggerProtocol).
        """
        logger = _logger
        logger.remove()

        # Consola
        logger.add(
            sys.stdout,
            level=self.level,
            colorize=self.colorize,
            format=self.console_format,
            enqueue=True,
        )

        # Archivo
        if log_dir is None:
            log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_dir / "app.log",
            level=self.file_level,
            rotation=self.rotation,
            retention=self.retention,
            format=self.file_format,
            enqueue=True,
        )

        # Asegura a Pylance que cumple el protocolo
        assert isinstance(logger, LoggerProtocol)
        return logger
