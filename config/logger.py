"""
Configuración centralizada para el logging de la aplicación Ajover.

Incluye parámetros para Loguru y opciones avanzadas de logging.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggerSettings(BaseSettings):
    """
    Configuración centralizada para el logger de Loguru.

    Permite controlar el comportamiento y los formatos del logging a través de variables
    de entorno o valores por defecto.

    - level: Nivel de log para la consola (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - colorize: Usar colores en la salida de consola
    - file_level: Nivel de log para el archivo (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - rotation: Tamaño máximo del archivo antes de rotar
    - retention: Cuánto tiempo conservar los logs rotados
    """

    model_config = SettingsConfigDict(extra='ignore', case_sensitive=False)

    level: str = Field(
        default='DEBUG',
        validation_alias='LOG_LEVEL',
        description='Nivel de log para la consola (DEBUG, INFO, WARNING, ERROR, CRITICAL)',
    )

    colorize: bool = Field(
        default=True,
        validation_alias='LOG_COLORIZE',
        description='Usar colores en la salida de consola',
    )

    file_level: str = Field(
        default='DEBUG',
        validation_alias='LOG_FILE_LEVEL',
        description='Nivel de log para el archivo',
    )

    rotation: str = Field(
        default='10 MB',
        validation_alias='LOG_ROTATION',
        description='Tamaño máximo del archivo antes de rotar',
    )

    retention: str = Field(
        default='7 days',
        validation_alias='LOG_RETENTION',
        description='Cuánto tiempo conservar los logs rotados',
    )

    console_format: str = Field(
        default=(
            '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
            '<level>{level: <8}</level> | '
            '<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - '
            '<level>{message}</level>'
        ),
        validation_alias='LOG_CONSOLE_FORMAT',
        description='Formato de los logs para la consola',
    )    
    
    file_format: str = Field(
        default=(
            '{time:YYYY-MM-DD HH:mm:ss}|{level: <8}|{process.id}|'
            '{name}.{function}:{line}|{message}'
        ),
        validation_alias='LOG_FILE_FORMAT',
        description='Formato de los logs para el archivo',
    )