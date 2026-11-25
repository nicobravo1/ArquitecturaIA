# Construir desde la raíz del repo:  docker build -f api/Dockerfile -t pagoseguro-fraud-api .
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/models

WORKDIR /app
COPY api/requirements.txt api/requirements.txt
RUN pip install -r api/requirements.txt

COPY fraud_core/ fraud_core/
COPY api/ api/
# Artefactos del modelo (metadata/umbrales + fallback local). En producción
# pueden montarse como volumen o descargarse de S3 al iniciar.
COPY models/ models/

# Contenedor endurecido: usuario sin privilegios, sin shell de login
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin app \
    && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health').status!=200)"

# 2 workers por contenedor (c6i/m5.large = 2 vCPU). Escalar horizontalmente con réplicas.
CMD ["uvicorn", "api.app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--proxy-headers", "--forwarded-allow-ips", "*"]
