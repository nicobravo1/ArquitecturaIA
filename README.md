# ArquitecturaIA – Detección de fraude en tiempo real (Caso 3: PagoSeguro)

Proyecto EP3 de **ITY1102 Arquitectura de Sistemas de Inteligencia Artificial** (Duoc UC).
Enunciado: [`docs/CASO_3.pdf`](docs/CASO_3.pdf) · Pauta: [`docs/EP3_pauta.pdf`](docs/EP3_pauta.pdf)

**Meta de negocio:** fraude efectivo < 0,1 % y falsos positivos < 5 %, decidiendo en ≤ 100 ms con explicación SHAP de cada rechazo.

## Arquitectura

```
Cliente / Postman ──HTTPS──▶ FastAPI (api/) ──text/csv──▶ SageMaker Endpoint (XGBoost)
        ▲                       │  decisión APPROVE / REVIEW / REJECT
        │                       └──evento──▶ RabbitMQ ──▶ Worker SHAP (worker/) ──▶ PostgreSQL (auditoría)
Streamlit (dashboard/) ◀────────────────────────────────────────────────────────────┘
```

## Estructura

| Carpeta | Contenido | Estado |
|---|---|---|
| `docs/` | Enunciado del caso y pauta EP3 | ✅ |
| `data/` | `creditcard.csv` (Kaggle, no versionado) | ✅ instrucciones |
| `fraud_core/` | Ingeniería de variables compartida (notebook, API, worker) | ✅ |
| `notebooks/` | `01_entrenamiento_xgboost.ipynb` – EDA, entrenamiento, umbral, métricas, SHAP, export | ✅ |
| `models/` | Artefactos generados (`xgboost-model`, `model.tar.gz`, `metadata.json`) | generado |
| `sagemaker/` | Despliegue del endpoint + auto-scaling | ⏳ |
| `api/` | FastAPI (OpenAPI, API key, rate limiting) | ⏳ |
| `worker/` | Consumidor RabbitMQ que calcula SHAP | ⏳ |
| `dashboard/` | Streamlit | ⏳ |
| `tests/` | `unit/` (pytest) y `load/` (Locust) | 🟡 |
| `infra/` | docker-compose, Dockerfiles, despliegue AWS | ⏳ |

## Inicio rápido

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip
jupyter notebook notebooks/01_entrenamiento_xgboost.ipynb   # Run All
pytest
```

Variables opcionales del notebook: `DATA_PATH` (ruta alternativa del CSV), `N_TRIALS` (configuraciones a probar, por defecto 12), `MLFLOW_TRACKING_URI`.

## Modelo

- **XGBoost** binario, 30 variables (`V1..V28`, `Amount`, `hour`), `scale_pos_weight` para el desbalance (0,17 % de fraude).
- Split estratificado 70/15/15, búsqueda aleatoria de hiperparámetros con early stopping sobre PR-AUC.
- Umbral elegido en validación: máximo recall con precisión ≥ 95 % (falsos positivos < 5 % de los bloqueos) + zona de revisión manual.
- Versión de `xgboost` fijada a la del contenedor de SageMaker (`3.0-5`) para que el modelo exportado cargue sin incompatibilidades.
