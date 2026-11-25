# Dashboard – Streamlit

Interfaz de monitoreo y auditoría para analistas de riesgo. **Solo consume la API REST**
(no toca la base de datos ni el modelo): es un cliente más de la arquitectura, con la
misma autenticación por API key.

| Pestaña | Contenido | Endpoints usados |
|---|---|---|
| 📊 Monitor | KPIs (transacciones, tasa de rechazo y revisión, monto bloqueado, tiempo SHAP), decisiones en el tiempo, distribución de probabilidades con umbrales, últimas transacciones | `/v1/stats`, `/v1/transactions`, `/v1/model` |
| 🔍 Explicaciones | Reporte SHAP de cada rechazo/revisión: texto para el auditor, top factores (rojo aumenta el riesgo, azul lo reduce), tabla completa | `/v1/transactions/{id}/explanation` |
| 🧪 Probar transacción | Formulario (monto, hora, tarjeta, V1..V28) a partir de ejemplos reales; muestra decisión, latencia y la explicación cuando el worker la publica | `/v1/transactions/score` |
| 🧠 Modelo | Versión, umbrales, métricas de test y cumplimiento de objetivos de negocio | `/v1/model` |

La barra lateral muestra el estado de la API, RabbitMQ y la base de auditoría, y permite
ajustar la ventana de tiempo y la actualización automática (cada 5–60 s).

## Ejecutar

```bash
pip install -r dashboard/requirements.txt
API_URL=http://localhost:8000 API_KEY=demo-key streamlit run dashboard/app.py
```

Con Docker (desde la raíz del repo):

```bash
docker build -f dashboard/Dockerfile -t pagoseguro-fraud-dashboard .
```

```bash
docker run -p 8501:8501 -e API_URL=http://<host-api>:8000 -e API_KEY=demo-key pagoseguro-fraud-dashboard
```

| Variable | Descripción |
|---|---|
| `API_URL` | URL base de la API (por defecto `http://localhost:8000`) |
| `API_KEY` | API key para el header `X-API-Key` (editable en la barra lateral) |
| `SAMPLES_PATH` | Opcional: `models/sample_payloads.json` para ofrecer transacciones del set de test |

## Colores

Las decisiones usan colores de **estado** (verde aprobada, ámbar revisión, rojo rechazada)
siempre con ícono y etiqueta, nunca solo color. Las contribuciones SHAP usan un par
**divergente** rojo/azul (aumenta / reduce el riesgo) y la distribución de probabilidades
una sola serie azul.
