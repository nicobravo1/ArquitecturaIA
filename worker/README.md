# Worker de explicabilidad SHAP

Proceso **asíncrono** que consume los eventos `transaction.scored` publicados por la API,
calcula la explicación SHAP de cada decisión y guarda un **reporte de auditoría inmutable**
en PostgreSQL (Amazon RDS en AWS). Queda fuera del camino crítico del pago: la API
responde en milisegundos y la explicación llega unos ms después (patrón **CQRS**: la API
*decide*, el worker *escribe el modelo de lectura*, la API/dashboard *consultan*).

```
API ──publish──▶ fraud.events ──transaction.scored──▶ shap.explanations ──▶ worker
                                                           │ (rechazo)          │
                                                           ▼                    ▼
                                                 shap.explanations.dlq    PostgreSQL fraud_audit
                                                                                │
                                   GET /v1/transactions/{id}/explanation ◀──────┘
```

## Qué guarda por transacción (`fraud_audit`)

`transaction_id` (PK), fechas de scoring y explicación, decisión, probabilidad, umbrales,
versión del modelo, backend usado, cliente (hash), tarjeta enmascarada, monto, variables,
`shap_base_value`, `shap_values` (las 30 variables), `top_reasons` (top-K con dirección),
`explanation_text` legible para el auditor y `explain_ms`.

Ejemplo de `explanation_text`:

> Rechazada: probabilidad de fraude 95.8% (umbral de rechazo 55.9%). Factores que aumentan
> el riesgo: V4=2.70 (+1.28); V5=2.90 (+1.04); V3=2.39 (+0.84); V6=2.17 (+0.64); V1=2.34 (+0.63).

## Decisiones de diseño

- **SHAP exacto con XGBoost nativo** (`pred_contribs=True`, TreeSHAP) en vez de
  `shap.TreeExplainer`: con shap 0.54 + xgboost 3.0, `TreeExplainer` calcula mal el valor
  base y la explicación no suma la predicción. Los tests verifican
  `base + Σ SHAP = logit(probabilidad)`.
- **Solo REJECT y REVIEW llevan SHAP** (`EXPLAIN_DECISIONS`): el regulador exige explicar
  cada rechazo; las aprobaciones se registran sin SHAP. Con 1M tx/día y ~0,2 % de fraude,
  esto reduce el cómputo SHAP en más de 99 %.
- **Misma versión de modelo que el endpoint:** si el evento trae otro `model_version`, el
  mensaje va a la DLQ (una explicación de otro modelo no es auditable). Al desplegar un
  modelo nuevo se actualizan endpoint y worker juntos; los mensajes en DLQ se pueden reprocesar.
- **Entrega al-menos-una-vez + idempotencia:** `ack` manual después del commit en la BD;
  `transaction_id` es PK con `ON CONFLICT DO NOTHING`, así una reentrega no duplica.
- **Errores:** permanentes (JSON inválido, campos faltantes, PAN sin enmascarar, versión de
  modelo distinta) → DLQ inmediata. Transitorios (BD caída) → reintento con backoff
  exponencial (`x-retries`, 0,5 s · 1 s · 2 s) y luego DLQ.
- **Escalado horizontal:** varios workers pueden consumir la misma cola (competing
  consumers); `WORKER_PREFETCH` controla cuántos mensajes procesa cada uno en paralelo.

## Ejecutar

```bash
pip install -r worker/requirements.txt
RABBITMQ_URL=amqp://guest:guest@localhost:5672/ \
DATABASE_URL=postgresql+psycopg://fraud:fraud@localhost:5432/fraud_audit \
python -m worker.app.main
```

| Variable | Defecto | Descripción |
|---|---|---|
| `RABBITMQ_URL` | `amqp://guest:guest@localhost:5672/` | Broker |
| `RABBITMQ_EXCHANGE` | `fraud.events` | Exchange topic |
| `DATABASE_URL` | `sqlite:///audit.db` | `postgresql+psycopg://…` en producción |
| `MODEL_DIR` | `models/` | `metadata.json` + `xgboost-model` |
| `EXPLAIN_DECISIONS` | `REJECT,REVIEW` | Decisiones con SHAP completo |
| `SHAP_TOP_K` | `5` | Razones principales en el reporte |
| `WORKER_PREFETCH` | `32` | Mensajes en vuelo por worker |
| `WORKER_MAX_RETRIES` | `3` | Reintentos ante errores transitorios |
| `WORKER_METRICS_PORT` | `9100` | Métricas Prometheus |

Métricas Prometheus (`:9100/metrics`): `fraud_worker_messages_total{result}`
(stored/duplicate/invalid/retried/failed), `fraud_worker_shap_seconds`,
`fraud_worker_lag_seconds` (scoring → registro de auditoría).

## Medición local de referencia

600 transacciones vía API con RabbitMQ y PostgreSQL 16 locales (modelo entrenado con datos
sintéticos): 600 registros guardados, SHAP en 332 (REJECT/REVIEW), **SHAP ≈ 4,7 ms** por
transacción y **retraso medio scoring → auditoría ≈ 1,3 ms**.
