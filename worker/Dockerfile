# Construir desde la raíz del repo:  docker build -f worker/Dockerfile -t pagoseguro-fraud-worker .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/models

WORKDIR /app
COPY worker/requirements.txt worker/requirements.txt
RUN pip install -r worker/requirements.txt

COPY fraud_core/ fraud_core/
COPY worker/ worker/
# El worker debe usar EXACTAMENTE la misma versión de modelo que el endpoint
COPY models/metadata.json models/xgboost-model models/

RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin app \
    && chown -R app /app
USER app

EXPOSE 9100
CMD ["python", "-m", "worker.app.main"]
