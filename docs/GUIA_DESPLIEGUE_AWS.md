# Guía de despliegue en AWS (AWS Academy Learner Lab)

Paso a paso para dejar el sistema funcionando en AWS y reunir las evidencias que pide la
pauta EP3. Tiempo total aproximado: **45–60 min** (la mayor parte es espera).

```
                ┌───────────── EC2 t3.large (docker compose) ─────────────┐
Internet ─HTTPS:8443─▶ Caddy ─▶ API FastAPI ──HTTPS/SigV4──▶ SageMaker endpoint (XGBoost)
        ─HTTPS:8444─▶ Caddy ─▶ Dashboard          │                  (auto-scaling 1-4)
                │                 RabbitMQ ◀──────┘──▶ Worker SHAP ──▶ PostgreSQL
                └──── logs ──▶ CloudWatch Logs /pagoseguro/fraud ───────────────────┘
                     métricas ──▶ CloudWatch (SageMaker + EC2)
```

## Antes de empezar: qué haces tú y qué hacen los scripts

**No hay que crear nada a mano en la consola de AWS** (ni la instancia EC2, ni el endpoint,
ni el bucket). Todos los comandos de esta guía se ejecutan en una **terminal de tu computador**,
dentro de la carpeta del repositorio; los scripts usan tus credenciales del lab para crear los
recursos en AWS. La consola web solo se usa para *mirar* y tomar capturas.

| Paso | Tú haces | El script hace |
|---|---|---|
| 1 | Iniciar el Learner Lab y copiar las credenciales a tu computador | — |
| 2 | Ejecutar el notebook con el dataset | Entrena el modelo y lo guarda en `models/` |
| 3 | `python -m sagemaker_deploy.deploy` | Crea el endpoint de SageMaker |
| 4 | `python -m infra.aws.ec2 up` | **Crea la instancia EC2**, el security group y la key, e instala todo adentro |
| 5 | Esperar ~10 min y `python -m infra.aws.ec2 status` | Indica si está listo y muestra las URLs |
| 6 | Abrir las URLs en el navegador / hacer la demo | — |
| 7 | `python -m infra.aws.ec2 down` y `python -m sagemaker_deploy.teardown --yes` | Borra todo |

## 0. Requisitos en tu computador

- Clonar el repositorio y entrar a su carpeta:

  ```bash
  git clone https://github.com/nicobravo1/ArquitecturaIA.git
  ```

  ```bash
  cd ArquitecturaIA
  ```

- Instalar la AWS CLI si no la tienes (https://aws.amazon.com/cli/), solo para verificar las credenciales.

- Python 3.12+, git y este repositorio clonado.
- Dependencias: `pip install -r requirements-dev.txt`
- Dataset en `data/creditcard.csv` (ver `data/README.md`).

## 1. Iniciar el lab y configurar credenciales

1. En AWS Academy → *Learner Lab* → **Start Lab** (esperar el círculo verde).
2. **AWS Details** → *AWS CLI: Show* → copiar el bloque completo (empieza con `[default]`)
   y pegarlo en el archivo de credenciales de tu computador, reemplazando su contenido:
   - Linux / macOS: `~/.aws/credentials`
   - Windows: `C:\Users\<tu usuario>\.aws\credentials` (crear la carpeta `.aws` si no existe)

   Las credenciales **expiran al terminar la sesión del lab**: repetir este paso cada vez.
3. Verificar:

   ```bash
   aws sts get-caller-identity
   ```

   ```bash
   export AWS_REGION=us-east-1
   ```

## 2. Entrenar el modelo con el dataset real

Abrir `notebooks/01_entrenamiento_xgboost.ipynb` y ejecutar todo (*Run All*). Genera en `models/`:
`xgboost-model`, `model.tar.gz`, `metadata.json` y `sample_payloads.json`.

> **Evidencia (IE1, 2.3.d):** capturas de las métricas de test (accuracy, precision, recall, F1,
> ROC-AUC, PR-AUC), la matriz de confusión, la curva PR, el gráfico SHAP y la tabla de
> objetivos de negocio.

## 3. Desplegar el modelo en SageMaker

```bash
python -m sagemaker_deploy.deploy
```

```bash
python -m sagemaker_deploy.invoke_test --requests 200
```

Si el lab rechaza `ml.m5.large`, usar `--instance-type ml.c5.large` (o `ml.t2.medium`) y/o
`--max-instances 2`. Detalle en `sagemaker_deploy/README.md`.

> **Evidencia (IE1, IE4):** consola de SageMaker → *Inference → Endpoints* (estado
> `InService`, configuración, auto-scaling en *Endpoint runtime settings*) y la salida de
> `invoke_test` (probabilidad del endpoint = probabilidad local; latencia p50/p95/p99).

## 4. Desplegar API, worker y dashboard en EC2

```bash
python -m infra.aws.ec2 up
```

El script:

1. sube los artefactos del modelo a S3 (la instancia descarga **la misma versión** que usa el endpoint);
2. crea un *key pair* (`infra/aws/pagoseguro-fraud-key.pem`) y un *security group* con 8443
   (API) y 8444 (dashboard) abiertos, y SSH solo desde tu IP;
3. genera API keys y contraseñas en `infra/aws/.secrets.env` (**no se versiona**);
4. lanza una `t3.large` con Amazon Linux 2023 y el perfil `LabInstanceProfile`, que instala
   Docker, clona el repo y levanta el stack con `docker compose`.

Esperar **6–10 minutos** y revisar:

```bash
python -m infra.aws.ec2 status
```

Cuando `API /health` responda `200 {"status":"ok",...,"scorer":"sagemaker"...}` está listo.

Opciones: `--instance-type t3.medium` (más barato), `--public-cidr <IP>/32` (restringir
acceso), `--ssh-cidr none`. Si el lab no permite crear key pairs o el perfil tiene otro
nombre, usar `--key-name` / `--instance-profile`.

### Accesos

| Qué | URL | Credencial |
|---|---|---|
| API (Swagger) | `https://<IP>:8443/docs` | header `X-API-Key: <CLIENT_API_KEY>` |
| Dashboard | `https://<IP>:8444` | usuario `analista`, clave `DASHBOARD_PASSWORD` |

Ambos valores están en `infra/aws/.secrets.env`. El certificado TLS lo emite la CA interna de
Caddy: el navegador muestra una advertencia (continuar) y en curl se usa `-k`.

> **Si el lab se cierra:** AWS Academy detiene la instancia y al reiniciar cambia la IP
> pública. Al iniciar de nuevo el lab, encender la instancia desde la consola EC2; el
> servicio `pagoseguro.service` detecta la nueva IP y levanta todo solo. `ec2 status`
> muestra la IP nueva.

## 5. Demostración (IE9): entrada → procesamiento → modelo → resultados

```bash
source infra/aws/.secrets.env
```

```bash
IP=$(python -c "import json;print(json.load(open('infra/aws/.state.json'))['public_ip'])")
```

```bash
python -c "import json;print(json.dumps(json.load(open('models/sample_payloads.json'))['fraud'][0] | {'transaction_id':'demo-001','card_number':'4111111111111234'}))" > /tmp/fraude.json
```

```bash
curl -sk -X POST "https://$IP:8443/v1/transactions/score" \
  -H "X-API-Key: $CLIENT_API_KEY" -H "Content-Type: application/json" -d @/tmp/fraude.json
```

```bash
curl -sk "https://$IP:8443/v1/transactions/demo-001/explanation" -H "X-API-Key: $CLIENT_API_KEY"
```

Mostrar además: `scorer: "sagemaker"` en la respuesta, la tarjeta enmascarada, el reporte
SHAP en el dashboard (pestañas *Probar transacción* y *Explicaciones*) y un `401` sin API key.
Para Postman: importar `https://<IP>:8443/openapi.json` y desactivar *SSL certificate verification*.

## 6. Pruebas de carga en AWS (IE10, 1.2.c)

El SLA de 100 ms aplica **dentro** del flujo de pago (en la misma región), por eso Locust se
ejecuta en la instancia y no desde el computador (la latencia Chile ↔ us-east-1 por sí sola
es de ~120–150 ms). Conectarse por SSH:

```bash
ssh -i infra/aws/pagoseguro-fraud-key.pem ec2-user@$IP
```

En la instancia:

```bash
cd /opt/pagoseguro && sudo mkdir -p tests/load/reports
```

```bash
sudo docker compose -f docker-compose.yml -f infra/aws/docker-compose.aws.yml --profile load \
  run --rm --user root -e LOAD_SHAPE=steady -e LOAD_SCALE=0.5 locust \
  -f /mnt/locust/locustfile.py --host http://api:8000 --headless --only-summary \
  --csv /mnt/locust/reports/aws_steady --html /mnt/locust/reports/aws_steady.html
```

Repetir con `LOAD_SHAPE=spike` para provocar el auto-scaling del endpoint. En otra terminal,
observar las instancias del endpoint:

```bash
watch -n 15 "aws sagemaker describe-endpoint --endpoint-name pagoseguro-fraud \
  --query 'ProductionVariants[0].[CurrentInstanceCount,DesiredInstanceCount]'"
```

Copiar los reportes al computador y generar tablas y gráficos:

```bash
scp -i infra/aws/pagoseguro-fraud-key.pem "ec2-user@$IP:/opt/pagoseguro/tests/load/reports/aws_*" tests/load/reports/
```

```bash
python tests/load/summarize.py tests/load/reports/aws_steady
```

> Una `t3.large` (2 vCPU) comparte CPU entre Locust, API, worker, RabbitMQ y PostgreSQL:
> reportar la carga con que se cumple el SLA y explicar que en producción la API escala
> horizontalmente (réplicas en nodos `c6i`) y el endpoint con auto-scaling.

## 7. Evidencias de CloudWatch (1.2.d)

```bash
python -m sagemaker_deploy.cloudwatch_metrics --hours 2
```

Genera `models/cloudwatch_metrics.csv` con métricas del endpoint (Invocations, ModelLatency
promedio y p99, OverheadLatency, errores 4XX/5XX, CPU) y de la instancia EC2 (CPU, red).

```bash
python -m infra.aws.ec2 logs --service api --minutes 30
```

Capturas recomendadas en la consola:

- **CloudWatch → Metrics → SageMaker → Endpoint metrics**: `Invocations` y `ModelLatency`
  durante la prueba de carga; `CPUUtilization` del endpoint.
- **CloudWatch → Log groups → /pagoseguro/fraud**: streams `pagoseguro-fraud-api-1`,
  `-worker-1`, etc.
- **EC2 → instancia → Monitoring**: CPU durante la prueba de carga.
- **SageMaker → endpoint → auto-scaling** con el número de instancias subiendo en el pico.

## 8. Costos de la demo (IE4)

Precios on-demand **referenciales** en us-east-1 (confirmar en las páginas de precios de AWS):

| Recurso | Tipo | USD/hora aprox. |
|---|---|---|
| Endpoint SageMaker | ml.m5.large × 1 | ~0,12 |
| EC2 | t3.large | ~0,08 |
| EBS | gp3 30 GB | ~0,003 |
| S3, CloudWatch Logs/Metrics | volumen de la demo | < 0,01 |
| **Total** | | **~0,21 USD/h (~5 USD/día encendido)** |

La arquitectura de producción del caso (nodos c6i, MSK, ElastiCache, RDS Multi-AZ) se estima
en el informe dentro del presupuesto de US$ 3.000–5.000/mes.

## 9. Apagar todo al terminar

```bash
python -m infra.aws.ec2 down
```

```bash
python -m sagemaker_deploy.teardown --yes
```

El endpoint de SageMaker **sigue cobrando** aunque se cierre la sesión del lab; la instancia
EC2 se detiene sola, pero su disco se sigue cobrando hasta ejecutar `down`.

## Solución de problemas

| Síntoma | Causa probable / solución |
|---|---|
| `ec2 status` muestra la API "sin respuesta" > 15 min | SSH y `sudo tail -100 /var/log/pagoseguro-setup.log` o `sudo journalctl -u pagoseguro` |
| `InvalidParameterValue ... iamInstanceProfile` | El perfil tiene otro nombre en tu lab: ver IAM → Roles → LabRole → *Instance profile ARNs*, usar `--instance-profile` |
| `Unsupported` / `not authorized` al lanzar | El lab no permite ese tipo: `--instance-type t3.medium` |
| `scorer: "local-fallback"` en las respuestas | La API no llega a SageMaker: revisar que el endpoint esté `InService` y el nombre en `.secrets.env` (`SAGEMAKER_ENDPOINT`) |
| `toomanyrequests` de Docker Hub en el log | Límite de descargas anónimas: esperar unos minutos y `sudo systemctl restart pagoseguro` |
| Credenciales expiradas en tu computador | Copiar de nuevo *AWS Details* a `~/.aws/credentials` |

## Checklist de evidencias para el informe y la presentación

| Ítem pauta | Evidencia | Dónde se obtiene |
|---|---|---|
| IE1 / 1.1 | Modelo, hiperparámetros, métricas; endpoint InService | Notebook, consola SageMaker, `invoke_test` |
| IE2 / 2.1 | Decisión + explicación SHAP en la aplicación | Dashboard (Explicaciones), `GET /explanation` |
| IE4 / 1.2.a,c | Configuración del endpoint, auto-scaling, pruebas de carga, costos | Consola SageMaker, Locust en EC2, sección 8 |
| IE5 / 1.2.b | Servidor AWS, modelo, dataset, dashboard operativos | `ec2 status`, URLs, capturas |
| 1.2.d | Logs y métricas CloudWatch | `cloudwatch_metrics`, `ec2 logs`, consola |
| IE7 / 1.3, 2.2 | Swagger/OpenAPI, flujo entre componentes, seguridad (API key, TLS, rate limit) | `/docs`, diagrama, `401`/`429` con curl |
| IE10 / 2.3 | ≥ 10 casos de prueba, p50/p95/p99, throughput, errores, métricas del modelo, optimizaciones | `pytest -v`, reportes Locust, `tests/load/README.md` |
| IE9 | Demo en vivo del flujo completo | Sección 5 |
