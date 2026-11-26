"""Despliega la API, el worker y el dashboard en una instancia EC2 (AWS Academy).

    python -m infra.aws.ec2 up        # crea SG, key pair e instancia; arranca el stack
    python -m infra.aws.ec2 status    # IP, URLs y salud de la API / dashboard
    python -m infra.aws.ec2 logs      # últimos logs de CloudWatch (/pagoseguro/fraud)
    python -m infra.aws.ec2 down      # termina la instancia y borra SG y key pair

Requisitos previos: notebook ejecutado (``models/``) y endpoint creado con
``python -m sagemaker_deploy.deploy`` (``models/deployment.json``).

Arquitectura resultante::

    Internet ──HTTPS:8443──▶ Caddy ─▶ API (FastAPI) ──▶ SageMaker endpoint
             ──HTTPS:8444──▶ Caddy ─▶ Dashboard          │
                                          RabbitMQ ◀─────┘──▶ worker SHAP ──▶ PostgreSQL
    Logs de contenedores → CloudWatch Logs · métricas del endpoint → CloudWatch

La instancia usa ``LabInstanceProfile`` (rol LabRole): ninguna credencial se guarda
en el servidor. Los secretos (API keys, contraseñas) se generan localmente, se
guardan en ``infra/aws/.secrets.env`` (ignorado por git) y viajan en el user-data.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import ssl
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from sagemaker_deploy.common import (  # noqa: E402
    DEFAULT_REGION, DEPLOYMENT_FILE, MODEL_DIR, PROJECT, TAGS, default_bucket, ensure_bucket, load_json, session,
)

STATE_FILE = HERE / ".state.json"
SECRETS_FILE = HERE / ".secrets.env"
USER_DATA_TEMPLATE = HERE / "user_data.sh"
LOG_GROUP = "/pagoseguro/fraud"
DEFAULT_REPO = "https://github.com/nicobravo1/ArquitecturaIA.git"
AL2023_PARAM = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
MODEL_FILES = ("model.tar.gz", "metadata.json", "sample_payloads.json", "deployment.json")


# ───────────────────────────── utilidades ─────────────────────────────
def _tags(resource: str, extra: dict | None = None) -> list[dict]:
    tags = TAGS + [{"Key": "Name", "Value": f"{PROJECT}-{resource}"}]
    tags += [{"Key": k, "Value": v} for k, v in (extra or {}).items()]
    return tags


def repo_url() -> str:
    try:
        url = subprocess.run(["git", "config", "--get", "remote.origin.url"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return DEFAULT_REPO
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.split(":", 1)[1]
    return url if url.startswith("https://github.com/") else DEFAULT_REPO


def my_public_ip() -> str | None:
    try:
        with urllib.request.urlopen("https://checkip.amazonaws.com", timeout=5) as r:
            return r.read().decode().strip()
    except OSError:
        return None


def load_state() -> dict:
    return json.loads(STATE_FILE.read_text(encoding="utf-8")) if STATE_FILE.exists() else {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def generate_secrets(endpoint: str, region: str) -> dict[str, str]:
    """Reutiliza los secretos existentes para que un `up` repetido no invalide las keys."""
    current: dict[str, str] = {}
    if SECRETS_FILE.exists():
        for line in SECRETS_FILE.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                current[k] = v
    tok = lambda: secrets.token_urlsafe(24)  # noqa: E731
    values = {
        "RABBITMQ_USER": "fraud",
        "RABBITMQ_PASSWORD": current.get("RABBITMQ_PASSWORD") or tok(),
        "POSTGRES_PASSWORD": current.get("POSTGRES_PASSWORD") or tok(),
        "CLIENT_API_KEY": current.get("CLIENT_API_KEY") or f"ps_{tok()}",
        "DASHBOARD_API_KEY": current.get("DASHBOARD_API_KEY") or f"ps_{tok()}",
        "LOAD_API_KEY": current.get("LOAD_API_KEY") or f"ps_{tok()}",
        "DASHBOARD_USER": "analista",
        "DASHBOARD_PASSWORD": current.get("DASHBOARD_PASSWORD") or secrets.token_urlsafe(12),
        "SAGEMAKER_ENDPOINT": endpoint,
        "AWS_REGION": region,
    }
    values["API_KEYS"] = ",".join([values["CLIENT_API_KEY"], values["DASHBOARD_API_KEY"], values["LOAD_API_KEY"]])
    SECRETS_FILE.write_text("# Generado por infra/aws/ec2.py - NO versionar\n"
                            + "\n".join(f"{k}={v}" for k, v in values.items()) + "\n",
                            encoding="utf-8", newline="\n")
    SECRETS_FILE.chmod(0o600)
    return values


def env_file(values: dict[str, str]) -> str:
    keys = ["RABBITMQ_USER", "RABBITMQ_PASSWORD", "POSTGRES_PASSWORD", "API_KEYS", "DASHBOARD_API_KEY",
            "LOAD_API_KEY", "DASHBOARD_USER", "DASHBOARD_PASSWORD", "SAGEMAKER_ENDPOINT", "AWS_REGION"]
    extra = {"SCORER_BACKEND": "sagemaker", "RATE_LIMIT_PER_MINUTE": "60000", "EXPLAIN_DECISIONS": "REJECT,REVIEW"}
    return "\n".join([f"{k}={values[k]}" for k in keys] + [f"{k}={v}" for k, v in extra.items()])


def render_user_data(repo: str, branch: str, model_s3: str, values: dict[str, str]) -> str:
    # utf-8 explícito (Windows usa cp1252 por defecto); la lectura en modo texto también
    # convierte CRLF -> LF si git en Windows cambió los fines de línea del script
    text = USER_DATA_TEMPLATE.read_text(encoding="utf-8").replace("\r\n", "\n")
    for key, val in {"__REPO_URL__": repo, "__BRANCH__": branch, "__MODEL_S3__": model_s3,
                     "__ENV_FILE__": env_file(values)}.items():
        text = text.replace(key, val)
    leftover = [k for k in ("__REPO_URL__", "__BRANCH__", "__MODEL_S3__", "__ENV_FILE__") if k in text]
    if leftover:
        raise RuntimeError(f"Quedaron variables sin reemplazar en el user-data: {leftover}")
    if len(text.encode()) > 16_000:
        raise RuntimeError("El user-data supera el límite de 16 KB de EC2")
    return text


def latest_al2023_ami(sess) -> str:
    try:
        return sess.client("ssm").get_parameter(Name=AL2023_PARAM)["Parameter"]["Value"]
    except Exception:   # sin permiso SSM: buscar la imagen más reciente
        images = sess.client("ec2").describe_images(
            Owners=["amazon"], Filters=[{"Name": "name", "Values": ["al2023-ami-2023*-x86_64"]},
                                        {"Name": "state", "Values": ["available"]}])["Images"]
        if not images:
            raise SystemExit("No se encontró una AMI de Amazon Linux 2023") from None
        return max(images, key=lambda i: i["CreationDate"])["ImageId"]


# ───────────────────────────── up ─────────────────────────────
def upload_model_artifacts(sess, region: str, dep: dict) -> str:
    s3 = sess.client("s3")
    bucket = dep["model_data"][5:].split("/", 1)[0] if dep.get("model_data") else default_bucket(sess, region)
    ensure_bucket(s3, bucket, region)
    prefix = f"{PROJECT}/serving/{dep['model_version']}"
    for name in MODEL_FILES:
        path = MODEL_DIR / name
        if not path.exists():
            raise SystemExit(f"Falta {path}. Ejecuta el notebook y sagemaker_deploy.deploy primero.")
        s3.upload_file(str(path), bucket, f"{prefix}/{name}", ExtraArgs={"ServerSideEncryption": "aws:kms"})
    return f"s3://{bucket}/{prefix}/"


def ensure_key_pair(ec2, name: str) -> Path:
    pem = HERE / f"{name}.pem"
    try:
        ec2.describe_key_pairs(KeyNames=[name])
        if not pem.exists():
            print(f"  ⚠ El key pair {name} ya existe pero no está {pem.name}: no podrás usar SSH")
        return pem
    except ec2.exceptions.ClientError:
        pass
    key = ec2.create_key_pair(KeyName=name, KeyType="ed25519",
                              TagSpecifications=[{"ResourceType": "key-pair", "Tags": _tags("key")}])
    pem.write_text(key["KeyMaterial"], encoding="ascii", newline="\n")   # LF: ssh rechaza claves con CRLF
    pem.chmod(0o600)
    return pem


# Las descripciones de reglas solo aceptan ASCII: a-zA-Z0-9. _-:/()#,@[]+=&;{}!$*
_SG_DESCRIPTION_OK = re.compile(r"^[a-zA-Z0-9. _\-:/()#,@\[\]+=&;{}!$*]{0,255}$")


def _ingress_rules(public_cidr: str, ssh_cidr: str | None) -> list[dict]:
    rules = [
        {"IpProtocol": "tcp", "FromPort": 8443, "ToPort": 8443,
         "IpRanges": [{"CidrIp": public_cidr, "Description": "API HTTPS"}]},
        {"IpProtocol": "tcp", "FromPort": 8444, "ToPort": 8444,
         "IpRanges": [{"CidrIp": public_cidr, "Description": "Dashboard HTTPS"}]},
    ]
    if ssh_cidr:
        rules.append({"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
                      "IpRanges": [{"CidrIp": ssh_cidr, "Description": "SSH administracion"}]})
    for r in rules:
        assert _SG_DESCRIPTION_OK.match(r["IpRanges"][0]["Description"]), r
    return rules


def ensure_security_group(ec2, name: str, public_cidr: str, ssh_cidr: str | None) -> str:
    """Crea el security group si no existe y agrega las reglas que falten (idempotente:
    un `up` que falló a medias se puede repetir sin dejar el grupo sin reglas)."""
    vpc = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
    if not vpc:
        raise SystemExit("No hay VPC por defecto en la región")
    found = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [name]},
                                                  {"Name": "vpc-id", "Values": [vpc[0]["VpcId"]]}])
    if found["SecurityGroups"]:
        sg = found["SecurityGroups"][0]["GroupId"]
        existing = {(p.get("FromPort"), r["CidrIp"]) for p in found["SecurityGroups"][0]["IpPermissions"]
                    for r in p.get("IpRanges", [])}
    else:
        sg = ec2.create_security_group(GroupName=name, VpcId=vpc[0]["VpcId"],
                                       Description="PagoSeguro fraude: API 8443, dashboard 8444, SSH",
                                       TagSpecifications=[{"ResourceType": "security-group",
                                                           "Tags": _tags("sg")}])["GroupId"]
        existing = set()
    missing = [r for r in _ingress_rules(public_cidr, ssh_cidr)
               if (r["FromPort"], r["IpRanges"][0]["CidrIp"]) not in existing]
    if missing:
        ec2.authorize_security_group_ingress(GroupId=sg, IpPermissions=missing)
    return sg


def up(args, sess=None) -> dict:
    sess = sess or session(args.region)
    ec2 = sess.client("ec2")
    state = load_state()
    if state.get("instance_id"):
        status = _instance(ec2, state["instance_id"])
        if status and status["State"]["Name"] not in ("terminated", "shutting-down"):
            raise SystemExit(f"Ya existe la instancia {state['instance_id']} ({status['State']['Name']}). "
                             "Usa `status` o `down` primero.")

    dep = load_json(DEPLOYMENT_FILE)
    meta = load_json(MODEL_DIR / "metadata.json")
    if dep["model_version"] != meta["model_version"]:
        raise SystemExit("models/metadata.json y deployment.json tienen versiones distintas: "
                         "vuelve a ejecutar sagemaker_deploy.deploy")
    print(f"[1/5] Subiendo artefactos del modelo {dep['model_version']} a S3")
    model_s3 = upload_model_artifacts(sess, args.region, dep)

    print("[2/5] Key pair y security group")
    pem = ensure_key_pair(ec2, args.key_name)
    ssh_cidr = args.ssh_cidr if args.ssh_cidr != "auto" else (f"{ip}/32" if (ip := my_public_ip()) else None)
    sg = ensure_security_group(ec2, f"{PROJECT}-sg", args.public_cidr, ssh_cidr)

    print("[3/5] Secretos y user-data")
    values = generate_secrets(dep["endpoint_name"], args.region)
    user_data = render_user_data(args.repo_url or repo_url(), args.branch, model_s3, values)

    print(f"[4/5] Lanzando {args.instance_type} (Amazon Linux 2023, perfil {args.instance_profile})")
    inst = ec2.run_instances(
        ImageId=latest_al2023_ami(sess), InstanceType=args.instance_type, MinCount=1, MaxCount=1,
        KeyName=args.key_name, SecurityGroupIds=[sg], UserData=user_data,
        IamInstanceProfile={"Name": args.instance_profile},
        # Hop limit 2: los contenedores llegan a IMDSv2 para obtener credenciales del rol
        MetadataOptions={"HttpTokens": "required", "HttpPutResponseHopLimit": 2, "HttpEndpoint": "enabled"},
        BlockDeviceMappings=[{"DeviceName": "/dev/xvda",
                              "Ebs": {"VolumeSize": args.volume_gb, "VolumeType": "gp3", "Encrypted": True}}],
        TagSpecifications=[{"ResourceType": "instance", "Tags": _tags("app", {"model_version": dep["model_version"]})}],
    )["Instances"][0]
    iid = inst["InstanceId"]
    ec2.get_waiter("instance_running").wait(InstanceIds=[iid], WaiterConfig={"Delay": 5, "MaxAttempts": 60})
    ip = _instance(ec2, iid).get("PublicIpAddress")

    state = {"instance_id": iid, "security_group_id": sg, "key_name": args.key_name, "public_ip": ip,
             "region": args.region, "endpoint_name": dep["endpoint_name"], "model_version": dep["model_version"],
             "model_s3": model_s3, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    save_state(state)
    print(f"[5/5] Instancia {iid} en {ip}. La instalación tarda ~6-10 min (docker build).")
    _print_access(state, values, pem)
    return state


def _print_access(state: dict, values: dict, pem: Path | None = None) -> None:
    ip = state.get("public_ip")
    print(f"""
  API (Swagger) : https://{ip}:8443/docs
  Dashboard     : https://{ip}:8444   usuario {values.get('DASHBOARD_USER')} / clave en {SECRETS_FILE.name}
  API key       : CLIENT_API_KEY en {SECRETS_FILE.relative_to(ROOT)}
  Avance        : python -m infra.aws.ec2 status
  SSH           : ssh -i {pem or HERE / (state['key_name'] + '.pem')} ec2-user@{ip}
                  sudo tail -f /var/log/pagoseguro-setup.log
                  sudo journalctl -u pagoseguro -f
  (El certificado es de la CA interna de Caddy: el navegador mostrará una advertencia; usa curl -k)
""")


# ───────────────────────────── status / logs / down ─────────────────────────────
def _instance(ec2, iid: str) -> dict | None:
    try:
        res = ec2.describe_instances(InstanceIds=[iid])["Reservations"]
    except ec2.exceptions.ClientError:
        return None
    return res[0]["Instances"][0] if res else None


def _probe(url: str, headers: dict | None = None) -> str:
    ctx = ssl.create_default_context()
    ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE   # CA interna de Caddy
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5, context=ctx) as r:
            return f"{r.status} {r.read(300).decode(errors='replace')}"
    except urllib.error.HTTPError as err:
        return f"{err.code} {err.reason}"
    except OSError as err:
        return f"sin respuesta ({err.__class__.__name__})"


def status(args, sess=None) -> dict:
    state = load_state()
    if not state:
        raise SystemExit("No hay despliegue registrado (infra/aws/.state.json)")
    sess = sess or session(state["region"])
    inst = _instance(sess.client("ec2"), state["instance_id"])
    st = inst["State"]["Name"] if inst else "no encontrada"
    ip = (inst or {}).get("PublicIpAddress", state.get("public_ip"))
    if ip and ip != state.get("public_ip"):      # AWS Academy reasigna la IP al reiniciar
        state["public_ip"] = ip
        save_state(state)
    print(f"Instancia {state['instance_id']}: {st} · IP {ip} · modelo {state['model_version']}")
    if st == "stopped":
        print("  La instancia está detenida (fin de sesión del lab). Iníciala desde la consola EC2;\n"
              "  el servicio pagoseguro.service levanta todo con la nueva IP.")
    if st == "running" and ip:
        api = _probe(f"https://{ip}:8443/health")
        dash = _probe(f"https://{ip}:8444/_stcore/health")
        print("  API /health    :", api)
        print("  Dashboard      :", "401 → listo (pide usuario/clave)" if dash.startswith("401") else dash)
        if api.startswith("sin respuesta"):
            print("  Aún instalando si pasaron < 10 min desde `up`. Si no, revisa el log por SSH:\n"
                  "    sudo tail -50 /var/log/pagoseguro-setup.log")
    return {"state": st, "ip": ip}


def logs(args, sess=None) -> None:
    state = load_state()
    sess = sess or session(state.get("region", args.region))
    cw = sess.client("logs")
    start = int((datetime.now(timezone.utc) - timedelta(minutes=args.minutes)).timestamp() * 1000)
    kwargs = {"logGroupName": LOG_GROUP, "startTime": start, "limit": args.limit}
    if args.service:
        kwargs["logStreamNamePrefix"] = f"{PROJECT}-{args.service}"
    for ev in cw.filter_log_events(**kwargs).get("events", []):
        ts = datetime.fromtimestamp(ev["timestamp"] / 1000, timezone.utc).strftime("%H:%M:%S")
        print(f"{ts} [{ev['logStreamName']}] {ev['message'].rstrip()}")


def down(args, sess=None) -> list[str]:
    state = load_state()
    if not state:
        print("Nada que eliminar")
        return []
    sess = sess or session(state["region"])
    ec2 = sess.client("ec2")
    done = []
    if _instance(ec2, state["instance_id"]):
        ec2.terminate_instances(InstanceIds=[state["instance_id"]])
        ec2.get_waiter("instance_terminated").wait(InstanceIds=[state["instance_id"]],
                                                   WaiterConfig={"Delay": 5, "MaxAttempts": 60})
        done.append(f"instance/{state['instance_id']}")
    for _ in range(12):     # la ENI puede tardar en liberarse tras terminar la instancia
        try:
            ec2.delete_security_group(GroupId=state["security_group_id"])
            done.append(f"security-group/{state['security_group_id']}")
            break
        except ec2.exceptions.ClientError as err:
            if "DependencyViolation" not in str(err):
                break
            time.sleep(10)
    ec2.delete_key_pair(KeyName=state["key_name"])
    done.append(f"key-pair/{state['key_name']}")
    (HERE / f"{state['key_name']}.pem").unlink(missing_ok=True)
    STATE_FILE.unlink(missing_ok=True)
    for item in done:
        print("eliminado:", item)
    print(f"(Se conservan {SECRETS_FILE.name}, el endpoint de SageMaker y los logs. "
          "Para el endpoint: python -m sagemaker_deploy.teardown)")
    return done


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("up", help="crear la instancia y desplegar")
    u.add_argument("--region", default=DEFAULT_REGION)
    u.add_argument("--instance-type", default="t3.large")
    u.add_argument("--instance-profile", default="LabInstanceProfile")
    u.add_argument("--key-name", default=f"{PROJECT}-key")
    u.add_argument("--volume-gb", type=int, default=30)
    u.add_argument("--public-cidr", default="0.0.0.0/0", help="quién puede acceder a 8443/8444")
    u.add_argument("--ssh-cidr", default="auto", help="'auto' = tu IP pública/32; 'none' para cerrar SSH")
    u.add_argument("--repo-url", default=None)
    u.add_argument("--branch", default="main")
    s = sub.add_parser("status")
    l_ = sub.add_parser("logs")
    l_.add_argument("--region", default=DEFAULT_REGION)
    l_.add_argument("--minutes", type=int, default=30)
    l_.add_argument("--limit", type=int, default=100)
    l_.add_argument("--service", choices=["api", "worker", "dashboard", "proxy", "rabbitmq", "postgres"])
    d = sub.add_parser("down")
    for sp in (s, d):
        sp.add_argument("--region", default=DEFAULT_REGION)
    args = p.parse_args(argv)
    if args.cmd == "up" and args.ssh_cidr == "none":
        args.ssh_cidr = None
    {"up": up, "status": status, "logs": logs, "down": down}[args.cmd](args)


if __name__ == "__main__":
    main()
