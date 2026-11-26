#!/bin/bash
# Arranque de la instancia EC2 (Amazon Linux 2023). Plantilla: infra/aws/ec2.py
# reemplaza las variables __ASI__ antes de lanzar la instancia.
# Log completo en /var/log/pagoseguro-setup.log
set -euo pipefail
exec > >(tee -a /var/log/pagoseguro-setup.log) 2>&1
echo "== PagoSeguro: inicio $(date -Is)"

REPO_URL="__REPO_URL__"
BRANCH="__BRANCH__"
MODEL_S3="__MODEL_S3__"
APP_DIR=/opt/pagoseguro
COMPOSE_VERSION=v2.29.7
BUILDX_VERSION=v0.17.1

# 1. Docker + compose + buildx
dnf install -y docker git
systemctl enable --now docker
mkdir -p /usr/local/lib/docker/cli-plugins
curl -fsSL -o /usr/local/lib/docker/cli-plugins/docker-compose \
  "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-x86_64"
curl -fsSL -o /usr/local/lib/docker/cli-plugins/docker-buildx \
  "https://github.com/docker/buildx/releases/download/${BUILDX_VERSION}/buildx-${BUILDX_VERSION}.linux-amd64"
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose /usr/local/lib/docker/cli-plugins/docker-buildx
docker compose version

# 2. Código y artefactos del modelo (misma versión que el endpoint de SageMaker)
rm -rf "$APP_DIR"
git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
cd "$APP_DIR"
aws s3 cp "$MODEL_S3" models/ --recursive
tar -xzf models/model.tar.gz -C models
ls -la models

# 3. Configuración (.env solo legible por root). Los secretos vienen del script local.
umask 077
cat > .env <<'ENVEOF'
__ENV_FILE__
ENVEOF
# Hash bcrypt de la clave del dashboard; en .env de compose el '$' se escapa como '$$'
HASH=$(docker run --rm caddy:2.8.4-alpine caddy hash-password --plaintext "$(grep '^DASHBOARD_PASSWORD=' .env | cut -d= -f2-)")
echo "DASHBOARD_PASSWORD_HASH=${HASH//\$/\$\$}" >> .env
sed -i '/^DASHBOARD_PASSWORD=/d' .env      # en el servidor solo queda el hash
umask 022

# 4. Servicio systemd: en cada arranque detecta la IP pública (cambia cuando AWS Academy
#    detiene y reinicia la instancia), la escribe en .env y levanta el stack.
cat > /usr/local/bin/pagoseguro-start <<'STARTEOF'
#!/bin/bash
set -euo pipefail
cd /opt/pagoseguro
TOKEN=$(curl -fsS -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
IP=$(curl -fsS -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4)
sed -i '/^PUBLIC_HOST=/d' .env
echo "PUBLIC_HOST=${IP}" >> .env
# El grupo de logs se crea antes de arrancar: si varios contenedores intentan crearlo a la
# vez, CloudWatch responde OperationAbortedException y el arranque falla.
REGION=$(grep '^AWS_REGION=' .env | cut -d= -f2)
aws logs create-log-group --log-group-name /pagoseguro/fraud --region "$REGION" 2>/dev/null || true
for attempt in 1 2 3; do
  if docker compose -f docker-compose.yml -f infra/aws/docker-compose.aws.yml up -d --build; then
    break
  fi
  echo "docker compose up falló (intento ${attempt}); reintentando en 15 s"
  sleep 15
done
echo "PagoSeguro en https://${IP}:8443/docs y https://${IP}:8444"
STARTEOF
chmod +x /usr/local/bin/pagoseguro-start
cat > /etc/systemd/system/pagoseguro.service <<'UNITEOF'
[Unit]
Description=PagoSeguro deteccion de fraude (docker compose)
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/bin/pagoseguro-start
TimeoutStartSec=1800

[Install]
WantedBy=multi-user.target
UNITEOF
systemctl daemon-reload
systemctl enable pagoseguro.service

# 5. Primer arranque (construye las imágenes: ~5-8 min)
systemctl start pagoseguro.service
docker compose -f docker-compose.yml -f infra/aws/docker-compose.aws.yml ps
echo "== PagoSeguro: listo $(date -Is)"
