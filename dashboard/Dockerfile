# Construir desde la raíz del repo:  docker build -f dashboard/Dockerfile -t pagoseguro-fraud-dashboard .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app
COPY dashboard/requirements.txt dashboard/requirements.txt
RUN pip install -r dashboard/requirements.txt

# El dashboard solo habla con la API: no necesita el modelo ni fraud_core
COPY dashboard/ dashboard/

RUN useradd --system --uid 10001 --create-home --shell /usr/sbin/nologin app \
    && chown -R app /app
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health').status!=200)"
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
