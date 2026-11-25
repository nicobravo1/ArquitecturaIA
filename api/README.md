# API de inferencia – FastAPI (pendiente)

`POST /v1/transactions/score` → `fraud_core.build_features` → endpoint SageMaker → decisión
(`APPROVE` / `REVIEW` / `REJECT` según `threshold` y `review_threshold` de `models/metadata.json`)
→ publica el evento en RabbitMQ para el worker SHAP.
Seguridad: API key, rate limiting, TLS. Documentación OpenAPI en `/docs`.
