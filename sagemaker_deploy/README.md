# Despliegue en AWS SageMaker

Scripts en Python con **boto3** (sin depender del SDK `sagemaker`) para publicar el modelo
del notebook en un **endpoint de inferencia en tiempo real** con el contenedor XGBoost
integrado de SageMaker (`sagemaker-xgboost:3.0-5`).

| Script | Qué hace |
|---|---|
| `deploy.py` | Valida artefactos → sube `model.tar.gz` a S3 (SSE-KMS, sin acceso público) → `Model` → `EndpointConfig` (+ Data Capture) → crea/actualiza `Endpoint` → auto-scaling → `models/deployment.json` |
| `invoke_test.py` | Envía transacciones reales (legítimas y fraudes), verifica que el endpoint da **la misma probabilidad que el modelo local** y mide latencia p50/p95/p99 |
| `cloudwatch_metrics.py` | Descarga métricas de CloudWatch (Invocations, ModelLatency, errores 4XX/5XX, CPU) a `models/cloudwatch_metrics.csv` (evidencia para el informe) |
| `autoscaling.py` | Registra/ajusta el auto-scaling de un endpoint existente |
| `teardown.py` | Elimina endpoint, configs, modelos y auto-scaling (**ejecutar al terminar cada sesión**) |

## Paso a paso (AWS Academy Learner Lab)

1. Ejecuta el notebook `notebooks/01_entrenamiento_xgboost.ipynb` → genera `models/model.tar.gz` y `models/metadata.json`.
2. *Start Lab* en el Learner Lab → **AWS Details** → copia las credenciales a `~/.aws/credentials`
   (expiran con la sesión; repetir cada vez).
3. Despliega (tarda 5-10 min):

   ```bash
   python -m sagemaker_deploy.deploy
   ```

   Opciones útiles: `--instance-type ml.c5.large`, `--max-instances 2`, `--serverless`,
   `--role arn:aws:iam::<cuenta>:role/LabRole`, `--no-wait`.
4. Prueba el endpoint:

   ```bash
   python -m sagemaker_deploy.invoke_test --requests 200
   ```

5. Evidencias de CloudWatch (tras generar tráfico, p. ej. con Locust):

   ```bash
   python -m sagemaker_deploy.cloudwatch_metrics --hours 2
   ```

6. Al terminar:

   ```bash
   python -m sagemaker_deploy.teardown --yes
   ```

Prueba manual con la AWS CLI (30 valores: `V1..V28, Amount, hour`):

```bash
aws sagemaker-runtime invoke-endpoint --endpoint-name pagoseguro-fraud \
  --content-type text/csv --accept application/json \
  --body "$(python -c 'import json;from fraud_core import build_features,to_sagemaker_csv;print(to_sagemaker_csv(build_features(json.load(open("models/sample_payloads.json"))["fraud"][:1])))')" \
  --cli-binary-format raw-in-base64-out /dev/stdout
```

### Notas de AWS Academy

- No se pueden crear roles IAM: se usa **`LabRole`** como rol de ejecución (por defecto).
- Región por defecto **us-east-1**. El Learner Lab limita tipos y cantidad de instancias:
  si `ml.m5.large` falla, prueba `ml.c5.large` o `ml.t2.medium`, y baja `--max-instances`.
- Si el registro de auto-scaling falla por permisos, el despliegue continúa y lo informa.
- Las credenciales y el endpoint **siguen cobrando** al presupuesto del lab mientras existan:
  usa `teardown.py` al terminar.

## Decisiones de diseño

- **Contenedor integrado en vez de uno propio:** sin Dockerfile ni ECR que mantener, imagen
  parcheada por AWS (hardened container), soporta `text/csv` y JSON. La versión de `xgboost`
  usada al entrenar se fija igual a la del contenedor (`3.0.5` ↔ `3.0-5`); `deploy.py`
  se niega a desplegar si no coinciden.
- **Endpoint en tiempo real (no serverless) por defecto:** el caso exige ≤ 100 ms en el flujo
  de pago síncrono; Serverless Inference tiene *cold starts* de segundos. `--serverless`
  queda como opción barata para desarrollo/demo.
- **Data Capture al 20 %** (configurable): guarda en S3 entradas y salidas del modelo para
  auditoría CMF y para medir *data drift* / *concept drift* (SageMaker Model Monitor).
- **Actualización blue/green:** volver a correr `deploy.py` crea un nuevo `EndpointConfig`
  y ejecuta `UpdateEndpoint`; SageMaker levanta la nueva flota antes de retirar la anterior
  (sin downtime). Cada `Model` lleva el `model_version` del notebook (trazabilidad con MLflow).
- **Auto-scaling target tracking** sobre `SageMakerVariantInvocationsPerInstance`:
  scale-out en 60 s ante picos, scale-in en 300 s para evitar oscilaciones.

## Escalabilidad y costos

**Capacidad requerida (caso):** 150 TPS promedio = 9.000 invocaciones/min; picos de
1.000 TPS = 60.000/min.

**Cómo fijar `--target-invocations`:** con Locust se mide el máximo de RPS que sostiene
**una** instancia sin superar el P99 objetivo; luego
`target = 0,7 × RPS_máx × 60` (30 % de holgura). El valor por defecto (3.000/min = 50 RPS
por instancia) es conservador para que el lab escale visiblemente con pocas instancias.

**Costo** = precio/hora de la instancia × 730 h × instancias promedio (+ S3 y CloudWatch,
marginales). Precios **referenciales on-demand us-east-1** — confirmar en
<https://aws.amazon.com/sagemaker/pricing/> antes de citarlos en el informe:

| Configuración | Instancias | USD/h aprox. | USD/mes aprox. |
|---|---|---|---|
| Demo (lab) `ml.m5.large` | 1 | ~0,12 | ~85 |
| Producción base `ml.c5.xlarge` (promedio 150 TPS) | 2 (Multi-AZ) | ~0,20 c/u | ~300 |
| Producción con picos (auto-scaling) | 2 → 8 | — | ~300 – 900 |
| Serverless (dev, 2 GB) | por uso | por GB-s | < 10 con poco tráfico |

Todas las opciones quedan dentro del presupuesto del caso (US$ 3.000-5.000/mes).
Ahorros adicionales: *Savings Plans* de SageMaker (hasta ~64 %) y bajar `--min-instances`
fuera de horario punta con *scheduled scaling*.
