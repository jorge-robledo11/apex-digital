"""
Este paquete 'config' centraliza toda la configuración de la aplicación.

El único punto de entrada para el resto de la aplicación es la función 'get_settings',
que devuelve un objeto de configuración global y validado.
"""

# Importa la función singleton desde el módulo de settings
from .settings import get_settings

__all__ = ['get_settings']