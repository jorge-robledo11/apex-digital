# fastapi/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# 1) Paquetes del sistema mínimos (incluye curl para healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
 && rm -rf /var/lib/apt/lists/*

# 2) Instalar uv vía pip (binario empaquetado)
RUN python -m pip install --upgrade pip && pip install "uv>=0.4.0"

# 3) Copiar metadatos primero (aprovecha la caché de Docker)
COPY pyproject.toml /app/pyproject.toml
# Si tienes uv.lock, descomenta la línea siguiente
# COPY uv.lock /app/uv.lock

# 4) Resolver dependencias del proyecto al sistema (PEP 621 / pyproject)
#    Sin venv: las instala en el site-packages del sistema
RUN uv pip install --system .

# 5) Copiar el código de la app
COPY src/ /app/src/

# 6) Usuario no-root
RUN useradd -m runner && chown -R runner:runner /app
USER runner

EXPOSE 8000

# 7) Ejecutar Uvicorn (puedes ajustar workers)
CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
