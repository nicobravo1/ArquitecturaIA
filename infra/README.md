# Infraestructura

## docker compose (local / demo)

`docker-compose.yml` en la raíz levanta el sistema completo:

| Servicio | Imagen | Puerto | Rol |
|---|---|---|---|
| `api` | `api/Dockerfile` | 8000 (solo localhost) | FastAPI de scoring y consultas |
| `worker` | `worker/Dockerfile` | — (métricas :9100) | Consumidor SHAP → PostgreSQL |
| `dashboard` | `dashboard/Dockerfile` | — | Streamlit |
| `rabbitmq` | rabbitmq:3.13-management | 15672 (consola, localhost) | Broker de eventos |
| `postgres` | postgres:16 | — | Almacén de auditoría |
| `proxy` | caddy:2 | **8443** (API), **8444** (dashboard) | Terminación **TLS** |
| `prometheus` *(perfil monitoring)* | prom/prometheus | 9090 | Métricas de API y worker |
| `grafana` *(perfil monitoring)* | grafana/grafana | 3000 | Panel de latencia, decisiones, drift y worker |
| `locust` *(perfil load)* | locustio/locust | 8089 | Pruebas de carga |

```bash
cp .env.example .env            # cambiar contraseñas y API keys
```

```bash
docker compose up -d --build    # requiere haber ejecutado el notebook (models/)
```

```bash
docker compose --profile monitoring up -d
```

```bash
docker compose up -d --scale api=3 --scale worker=2
```

- **TLS:** Caddy emite certificados con su CA interna (`tls internal`); el navegador mostrará
  una advertencia salvo que se confíe en esa CA. En AWS se usa un ALB con certificado ACM.
- **Secretos:** solo en `.env` (ignorado por git). En AWS: Secrets Manager / SSM Parameter Store.
- **Puertos administrativos** (RabbitMQ, Prometheus, Grafana, API sin TLS) publicados solo en
  `127.0.0.1`.
- **Grafana** viene con la fuente Prometheus y el panel *PagoSeguro · Detección de fraude*
  ya aprovisionados (`infra/grafana/`).

## AWS (EC2 + SageMaker)

`infra/aws/ec2.py` crea una instancia EC2 (Amazon Linux 2023, `LabInstanceProfile`) que
levanta este mismo compose con el override `infra/aws/docker-compose.aws.yml`:

- la API usa el **endpoint de SageMaker** (credenciales del rol vía IMDSv2, sin claves en el servidor);
- logs de todos los contenedores a **CloudWatch Logs** (`/pagoseguro/fraud`);
- Caddy con TLS para la IP pública y **usuario/clave** en el dashboard (`infra/aws/Caddyfile`);
- servicio systemd `pagoseguro.service` que en cada arranque detecta la IP pública (cambia
  cuando AWS Academy detiene y reinicia la instancia) y levanta el stack.

Paso a paso: [`docs/GUIA_DESPLIEGUE_AWS.md`](../docs/GUIA_DESPLIEGUE_AWS.md).
