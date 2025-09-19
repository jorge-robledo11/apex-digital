FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# 1) Paquetes del sistema mínimos para serving
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
 && rm -rf /var/lib/apt/lists/*

# 2) Instalar uv
RUN python -m pip install --upgrade pip && pip install "uv>=0.4.0"

# 3) Instalar solo dependencias de serving (mínimas)
RUN uv pip install --system \
    "fastapi>=0.104.0" \
    "uvicorn[standard]>=0.24.0" \
    "pydantic>=2.11.9" \
    "pydantic-settings>=2.10.1" \
    "mlflow>=2.8.0" \
    "xgboost>=2.0.0" \
    "scikit-learn>=1.3.0" \
    "pandas>=2.0.0" \
    "numpy>=1.24.0" \
    "loguru>=0.7.0"

# 4) Copiar solo el código de serving
COPY src/serving/ /app/src/serving/

# 5) Usuario no-root
RUN useradd -m runner && chown -R runner:runner /app
USER runner

EXPOSE 8000

# 6) Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 7) Comando optimizado para producción
CMD ["uvicorn", "src.serving.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--access-log", \
     "--log-level", "info"]
