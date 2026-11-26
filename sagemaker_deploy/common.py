"""Configuración común de los scripts de SageMaker (boto3, sin el SDK `sagemaker`)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "models"
DEPLOYMENT_FILE = MODEL_DIR / "deployment.json"

PROJECT = "pagoseguro-fraud"
DEFAULT_ENDPOINT = os.environ.get("SAGEMAKER_ENDPOINT", PROJECT)
DEFAULT_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
VARIANT = "AllTraffic"
TAGS = [{"Key": "project", "Value": PROJECT}, {"Key": "course", "Value": "ITY1102-EP3"}]

# Contenedor XGBoost integrado de SageMaker. Debe coincidir con la versión de
# xgboost usada al entrenar (requirements.txt fija xgboost==3.0.5).
XGBOOST_CONTAINER_VERSION = "3.0-5"
# Cuentas ECR oficiales del contenedor 3.0-5 (sagemaker SDK, image_uri_config/xgboost.json)
XGBOOST_REGISTRIES = {
    "af-south-1": "510948584623", "ap-east-1": "651117190479", "ap-northeast-1": "354813040037",
    "ap-northeast-2": "366743142698", "ap-northeast-3": "867004704886", "ap-south-1": "720646828776",
    "ap-southeast-1": "121021644041", "ap-southeast-2": "783357654285", "ca-central-1": "341280168497",
    "eu-central-1": "492215442770", "eu-north-1": "662702820516", "eu-south-1": "978288397137",
    "eu-west-1": "141502667606", "eu-west-2": "764974769150", "eu-west-3": "659782779980",
    "me-south-1": "801668240914", "sa-east-1": "737474898029", "us-east-1": "683313688378",
    "us-east-2": "257758044811", "us-west-1": "746614075791", "us-west-2": "246618743249",
}


def xgboost_image_uri(region: str, version: str = XGBOOST_CONTAINER_VERSION) -> str:
    try:
        account = XGBOOST_REGISTRIES[region]
    except KeyError:
        raise SystemExit(f"Región {region} no mapeada; usa --image-uri explícito") from None
    return f"{account}.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:{version}"


def session(region: str) -> boto3.Session:
    return boto3.Session(region_name=region)


def account_id(sess: boto3.Session) -> str:
    return sess.client("sts").get_caller_identity()["Account"]


def resolve_role(sess: boto3.Session, role: str | None) -> str:
    """Rol de ejecución de SageMaker.

    En AWS Academy no se pueden crear roles IAM: se usa ``LabRole`` (ya existe y
    tiene permisos de SageMaker, S3, CloudWatch y ECR).
    """
    role = role or os.environ.get("SAGEMAKER_ROLE_ARN") or "LabRole"
    if role.startswith("arn:"):
        return role
    return f"arn:aws:iam::{account_id(sess)}:role/{role}"


def default_bucket(sess: boto3.Session, region: str) -> str:
    return f"sagemaker-{region}-{account_id(sess)}"


def ensure_bucket(s3, bucket: str, region: str) -> None:
    try:
        s3.head_bucket(Bucket=bucket)
        return
    except s3.exceptions.ClientError as err:
        if err.response["Error"]["Code"] not in ("404", "NoSuchBucket", "NotFound"):
            raise
    kwargs = {"Bucket": bucket}
    if region != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
    s3.create_bucket(**kwargs)
    s3.put_public_access_block(Bucket=bucket, PublicAccessBlockConfiguration={
        "BlockPublicAcls": True, "IgnorePublicAcls": True,
        "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
    # Cifrado en reposo (PCI-DSS): SSE-KMS con la llave administrada de AWS
    s3.put_bucket_encryption(Bucket=bucket, ServerSideEncryptionConfiguration={
        "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "aws:kms"}}]})


def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"No existe {path}. Ejecuta primero el notebook de entrenamiento / deploy.py")
    raw = path.read_bytes()
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:          # archivo escrito en Windows con cp1252
        return json.loads(raw.decode("cp1252"))
