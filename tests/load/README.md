# Pruebas de carga – Locust

`locustfile.py` simula el tráfico del caso contra la API:

| Perfil | Peso | Qué hace |
|---|---|---|
| `PaymentGateway` | 95 % | `POST /v1/transactions/score`, llegadas de **Poisson** a ~1 req/s por usuario |
| `BatchClient` | 3 % | `POST /v1/transactions/score/batch` con 20 transacciones |
| `RiskAnalyst` | 2 % | stats, listados, explicaciones SHAP y health (uso del dashboard) |

Formas de carga (`LOAD_SHAPE`, escala `LOAD_SCALE`; 1.0 = cifras del caso; `TIME_SCALE` acorta etapas):

| `LOAD_SHAPE` | Carga | Objetivo |
|---|---|---|
| `steady` | rampa → 150 TPS sostenidos | Promedio del caso |
| `spike` | 150 → 400 → **1.000 TPS** → 150 | Pico del caso / auto-scaling |
| `stress` | escalones 100 → 2.000 TPS | Punto de quiebre |

Al terminar se evalúa el **SLA** sobre el scoring: p99 ≤ `SLA_P99_MS` (90 ms, P99 del caso) y
errores ≤ 1 %. Si no se cumple, Locust sale con código 1 (útil en CI).

```bash
API_KEY=demo-key LOAD_SHAPE=steady locust -f tests/load/locustfile.py \
  --host http://localhost:8000 --headless --only-summary \
  --csv tests/load/reports/steady --html tests/load/reports/steady.html
```

```bash
python tests/load/summarize.py tests/load/reports/steady   # tabla Markdown + gráficos PNG
```

Con interfaz web: `docker compose --profile load up -d` → http://localhost:8089.
Para más de ~500 TPS desde una máquina: `--processes -1`.

## Resultados de referencia (entorno local)

Entorno: **un contenedor de 4 vCPU compartido** por Locust, la API (2–3 procesos uvicorn,
backend local), el worker SHAP, RabbitMQ 3.12 y PostgreSQL 16. Modelo entrenado con datos
sintéticos (la latencia no depende de la calidad del modelo). Gráficos y tablas en
[`docs/evidencias/carga_local/`](../../docs/evidencias/carga_local/).

| Prueba | Carga real | Requests | Errores | p50 | p95 | p99 | SLA p99 ≤ 90 ms |
|---|---|---|---|---|---|---|---|
| `steady` (escala 1.0) | ~150 TPS | 11.745 | 0 % | 5 ms | 10 ms | 25 ms | ✅ |
| `spike` (escala 0.4, pico 400 TPS) | hasta ~390 TPS | 20.794 | 0 % | 7 ms | 53 ms | 190 ms | ❌ |
| `stress` (escala 0.25) | 14 → 470 TPS | — | 0 % | ver tabla | | | ✅ hasta ~210 TPS |

Escalones de `stress` (percentiles de la ventana de cada escalón):

| TPS medidos | 14 | 40 | 89 | 152 | 214 | 300 | 410 | 469 |
|---|---|---|---|---|---|---|---|---|
| p50 (ms) | 6 | 5 | 5 | 5 | 5 | 5 | 6 | 7 |
| p95 (ms) | 14 | 15 | 16 | 15 | 15 | 24 | 54 | 63 |
| p99 (ms) | 19 | 26 | 42 | 58 | 69 | 120 | 200 | 190 |

**Lectura:** sin errores en ningún escenario. En este contenedor compartido el SLA se cumple
hasta ~210 TPS (cubre el promedio del caso, 150 TPS); sobre ~300 TPS la cola p99 crece porque
generador de carga, API, worker, broker y base de datos compiten por las mismas 4 CPU. Para el
pico de 1.000 TPS se escala horizontalmente la API (réplicas detrás del balanceador) y el
endpoint de SageMaker (auto-scaling), en nodos dedicados. Con SageMaker como backend se suma
la latencia de red al endpoint: medirla en AWS con `python -m sagemaker_deploy.invoke_test`.

## Optimizaciones aplicadas a partir de estas pruebas

1. **Ruta caliente sin pandas ni DMatrix** (`fraud_core.features_matrix` + `inplace_predict`):
   construir un DataFrame (2,2 ms) y un `DMatrix` (2,6 ms) por request se reemplazó por un
   arreglo NumPy y `inplace_predict` (**0,13 ms, ~37× más rápido**). Tiempo de servidor por
   request: ~7 ms → ~3 ms; throughput por conexión: 113 → 225 req/s. Un test verifica que las
   probabilidades son idénticas a la ruta anterior.
2. **Llegadas de Poisson en el generador:** con esperas fijas todos los usuarios disparaban en el
   mismo instante cada segundo (ráfagas de 60 requests) y la prueba medía colas artificiales
   (p50 ≈ 250 ms con el servidor respondiendo en 4 ms).
3. **`FastHttpUser`** (geventhttpclient) en vez de `HttpUser` para que el generador no distorsione.
4. **Un proceso uvicorn por contenedor** y escalado por réplicas: cada réplica expone sus propias
   métricas a Prometheus.

## Oportunidades de optimización pendientes

- **Inserciones por lotes en el worker** (las aprobaciones se insertan una a una; con
  `executemany` o `COPY` bajaría el costo de CPU de PostgreSQL y del worker).
- **Serialización con `orjson`** (`ORJSONResponse`) para reducir el costo de JSON en la API.
- **Rate limiting en Redis/API Gateway** para que el límite sea global entre réplicas.
- **Feature store (Redis)** para variables agregadas en tiempo real (monto en los últimos 5 min)
  descrito en el caso, presupuestando < 10 ms.
