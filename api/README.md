# API de inferencia – FastAPI

API REST que el sistema de pagos llama de forma **síncrona** para decidir cada transacción.
Documentación interactiva **Swagger** en `/docs`, **ReDoc** en `/redoc` y el esquema
**OpenAPI** en `/openapi.json`.

```
POST /v1/transactions/score
  → validación Pydantic + API key + rate limit
  → fraud_core.build_features                    (mismas variables que el entrenamiento)
  → SageMaker endpoint  ──(falla)──▶ modelo local (fallback de alta disponibilidad)
  → APPROVE / REVIEW / REJECT                    (umbrales de models/metadata.json)
  → evento a RabbitMQ  (exchange fraud.events → cola shap.explanations → worker SHAP)
  → respuesta
```

## Endpoints

| Método | Ruta | Auth | Descripción |
|---|---|---|---|
| POST | `/v1/transactions/score` | API key | Evalúa una transacción |
| POST | `/v1/transactions/score/batch` | API key | Evalúa hasta `MAX_BATCH_SIZE` transacciones |
| GET | `/v1/transactions/{id}/explanation` | API key | Reporte SHAP de auditoría (generado por el worker; 404 si aún está en cola) |
| GET | `/v1/transactions?decision=&limit=&hours=` | API key | Transacciones auditadas recientes |
| GET | `/v1/stats?hours=` | API key | Conteos, tasa de rechazo, montos y tiempo SHAP por decisión |
| GET | `/v1/model` | API key | Versión del modelo, umbrales y métricas de test |
| GET | `/health` | — | Liveness/readiness (estado de RabbitMQ y de la BD de auditoría) |
| GET | `/metrics` | — | Métricas Prometheus (latencia, decisiones, eventos, errores, distribución de probabilidades) |

Códigos de error: `401` API key inválida · `413` lote demasiado grande · `422` payload inválido ·
`429` rate limit (con `Retry-After`) · `503` scoring no disponible.

## Ejecutar en local

```bash
pip install -r api/requirements.txt
# backend local (sin AWS), sin broker
SCORER_BACKEND=local API_KEYS=demo-key uvicorn api.app.main:create_app --factory --port 8000
```

```bash
curl -s -X POST localhost:8000/v1/transactions/score \
  -H "X-API-Key: demo-key" -H "Content-Type: application/json" \
  -d '{"transaction_id":"tx-1","Time":406,"Amount":0,"V1":-2.31,"V2":1.95,"V3":-1.61,"V4":3.99,"V5":-0.52,"V6":-1.42,"V7":-2.53,"V8":1.39,"V9":-2.77,"V10":-2.77,"V11":3.20,"V12":-2.89,"V13":-0.59,"V14":-4.28,"V15":0.38,"V16":-1.14,"V17":-2.83,"V18":-0.01,"V19":0.41,"V20":0.12,"V21":0.51,"V22":-0.03,"V23":-0.46,"V24":0.32,"V25":0.04,"V26":0.17,"V27":0.26,"V28":-0.14}'
```

Con Docker (desde la raíz del repo):

```bash
docker build -f api/Dockerfile -t pagoseguro-fraud-api .
```

```bash
docker run -p 8000:8000 -e API_KEYS=demo-key -e SCORER_BACKEND=sagemaker \
  -e SAGEMAKER_ENDPOINT=pagoseguro-fraud -e AWS_REGION=us-east-1 \
  -v ~/.aws:/home/app/.aws:ro pagoseguro-fraud-api
```

## Configuración (variables de entorno)

| Variable | Defecto | Descripción |
|---|---|---|
| `SCORER_BACKEND` | `sagemaker` | `sagemaker` (endpoint AWS) o `local` (modelo en proceso) |
| `SAGEMAKER_ENDPOINT` | de `models/deployment.json` | Nombre del endpoint |
| `AWS_REGION` | de `models/deployment.json` | Región del endpoint |
| `LOCAL_FALLBACK` | `true` | Usar el modelo local si SageMaker falla |
| `API_KEYS` | *(vacío = sin auth, límite por IP)* | Keys válidas separadas por coma |
| `RATE_LIMIT_PER_MINUTE` | `600` | Requests por minuto por cliente |
| `RABBITMQ_URL` | *(vacío = sin eventos)* | p. ej. `amqp://user:pass@rabbitmq:5672/` (`amqps://` para TLS) |
| `RABBITMQ_EXCHANGE` | `fraud.events` | Exchange topic |
| `RABBITMQ_PUBLISH_TIMEOUT_MS` | `50` | Tiempo máximo de publicación (no bloquear el pago) |
| `DATABASE_URL` | *(vacío = consultas de auditoría deshabilitadas)* | Misma BD que escribe el worker |
| `MAX_BATCH_SIZE` | `100` | Tamaño máximo de lote |
| `MODEL_DIR` | `models/` | Carpeta con `metadata.json` y `xgboost-model` |
| `CORS_ORIGINS` | *(vacío)* | Orígenes permitidos (p. ej. el dashboard) |

## Seguridad

- **Autenticación:** API key en `X-API-Key`, comparación en tiempo constante; en logs y
  métricas solo aparece un hash de la key.
- **Rate limiting:** ventana deslizante de 60 s por cliente (en memoria). Con varias
  réplicas: Redis/ElastiCache o throttling de API Gateway/WAF.
- **TLS:** terminado en el balanceador (ALB/API Gateway con certificado ACM) o en nginx
  delante de uvicorn; RabbitMQ con `amqps://`. En el VPC, el tráfico a SageMaker va por
  HTTPS (boto3) con firma SigV4.
- **PCI-DSS:** el PAN (`card_number`) es opcional, se valida y **solo se conserva
  enmascarado** (`************1234`): nunca se registra ni se publica en RabbitMQ.
- **Contenedor endurecido:** imagen slim, usuario sin privilegios (uid 10001), healthcheck.
- **Resiliencia:** si RabbitMQ cae, el pago **no se bloquea** (`explanation_status=unavailable`)
  y la reconexión se hace en segundo plano; si SageMaker cae, se usa el modelo local.

## Medición local de referencia

Backend local + RabbitMQ con *publisher confirms*, 300 requests secuenciales (contenedor de
desarrollo, modelo entrenado con datos sintéticos): **p50 7,6 ms · p95 9,1 ms · p99 10,1 ms**.
Con SageMaker se suma la latencia de red al endpoint (medir con
`python -m sagemaker_deploy.invoke_test`).
