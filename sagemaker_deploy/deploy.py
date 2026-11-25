"""Despliega el modelo XGBoost en un endpoint de AWS SageMaker.

Pasos:
 1. Valida los artefactos de ``models/`` (generados por el notebook).
 2. Sube ``model.tar.gz`` a S3 (bucket cifrado, sin acceso público).
 3. Crea ``Model`` (contenedor XGBoost integrado), ``EndpointConfig`` (con Data
    Capture para auditoría y monitoreo de drift) y crea/actualiza el ``Endpoint``.
 4. Registra auto-scaling (target tracking sobre invocaciones por instancia).
 5. Guarda ``models/deployment.json`` con todo lo necesario para la API.

Uso (AWS Academy: exportar credenciales del Learner Lab antes)::

    python -m sagemaker_deploy.deploy                       # ml.m5.large, 1-4 instancias
    python -m sagemaker_deploy.deploy --instance-type ml.c5.large --max-instances 2
    python -m sagemaker_deploy.deploy --serverless          # más barato, con cold start
"""
from __future__ import annotations

import argparse
import json
import re
import tarfile
from datetime import datetime, timezone

from sagemaker_deploy.common import (
    DEFAULT_ENDPOINT, DEFAULT_REGION, DEPLOYMENT_FILE, MODEL_DIR, PROJECT, TAGS, VARIANT,
    XGBOOST_CONTAINER_VERSION, default_bucket, ensure_bucket, load_json, resolve_role,
    session, xgboost_image_uri,
)
from fraud_core import FEATURE_COLUMNS


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--endpoint-name", default=DEFAULT_ENDPOINT)
    p.add_argument("--bucket", help="Bucket S3 (por defecto sagemaker-<region>-<account>)")
    p.add_argument("--role", help="ARN o nombre del rol de ejecución (por defecto LabRole)")
    p.add_argument("--image-uri", help="Imagen del contenedor (por defecto XGBoost integrado "
                                       f"{XGBOOST_CONTAINER_VERSION})")
    p.add_argument("--instance-type", default="ml.m5.large")
    p.add_argument("--min-instances", type=int, default=1)
    p.add_argument("--max-instances", type=int, default=4)
    p.add_argument("--target-invocations", type=int, default=3000,
                   help="Invocaciones/minuto por instancia que dispara el scale-out")
    p.add_argument("--data-capture-percent", type=int, default=20,
                   help="%% de requests guardados en S3 (0 = desactivado)")
    p.add_argument("--serverless", action="store_true",
                   help="Serverless Inference (paga por uso; sin auto-scaling ni latencia garantizada)")
    p.add_argument("--serverless-memory", type=int, default=2048)
    p.add_argument("--serverless-concurrency", type=int, default=5)
    p.add_argument("--no-wait", action="store_true", help="No esperar a que el endpoint quede InService")
    args = p.parse_args(argv)
    if not 1 <= args.min_instances <= args.max_instances:
        p.error("Se requiere 1 <= --min-instances <= --max-instances")
    if not 0 <= args.data_capture_percent <= 100:
        p.error("--data-capture-percent debe estar entre 0 y 100")
    return args


def validate_artifacts() -> dict:
    """Comprueba que el notebook generó artefactos coherentes con este despliegue."""
    meta = load_json(MODEL_DIR / "metadata.json")
    tar_path = MODEL_DIR / "model.tar.gz"
    if not tar_path.exists():
        raise SystemExit(f"Falta {tar_path}: ejecuta el notebook de entrenamiento")
    with tarfile.open(tar_path) as tar:
        if "xgboost-model" not in tar.getnames():
            raise SystemExit("model.tar.gz debe contener 'xgboost-model' en la raíz")
    if meta["feature_columns"] != FEATURE_COLUMNS:
        raise SystemExit("Las variables del modelo no coinciden con fraud_core.FEATURE_COLUMNS")
    trained = ".".join(meta["framework"]["xgboost"].split(".")[:2])
    container = XGBOOST_CONTAINER_VERSION.split("-")[0]
    if trained != container:
        raise SystemExit(f"Modelo entrenado con xgboost {trained} pero el contenedor es {container}. "
                         "Reentrena con requirements.txt o usa --image-uri.")
    return meta


def safe_name(*parts: str, max_len: int = 63) -> str:
    name = re.sub(r"[^a-zA-Z0-9-]", "-", "-".join(parts))
    return re.sub(r"-+", "-", name).strip("-")[:max_len].rstrip("-")


def upload_model(s3, bucket: str, model_version: str) -> str:
    key = f"{PROJECT}/models/{model_version}/model.tar.gz"
    s3.upload_file(str(MODEL_DIR / "model.tar.gz"), bucket, key,
                   ExtraArgs={"ServerSideEncryption": "aws:kms"})
    return f"s3://{bucket}/{key}"


def endpoint_exists(sm, name: str) -> bool:
    try:
        sm.describe_endpoint(EndpointName=name)
        return True
    except sm.exceptions.ClientError as err:
        if "Could not find" in str(err) or err.response["Error"]["Code"] == "ValidationException":
            return False
        raise


def scaling_resource_id(endpoint: str) -> str:
    return f"endpoint/{endpoint}/variant/{VARIANT}"


def deregister_autoscaling(aas, endpoint: str) -> None:
    rid = scaling_resource_id(endpoint)
    found = aas.describe_scalable_targets(ServiceNamespace="sagemaker", ResourceIds=[rid])
    if found.get("ScalableTargets"):
        aas.deregister_scalable_target(ServiceNamespace="sagemaker", ResourceId=rid,
                                       ScalableDimension="sagemaker:variant:DesiredInstanceCount")


def register_autoscaling(aas, endpoint: str, min_c: int, max_c: int, target: int) -> None:
    rid = scaling_resource_id(endpoint)
    aas.register_scalable_target(
        ServiceNamespace="sagemaker", ResourceId=rid,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=min_c, MaxCapacity=max_c)
    aas.put_scaling_policy(
        PolicyName=f"{endpoint}-invocations-target",
        ServiceNamespace="sagemaker", ResourceId=rid,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": float(target),
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"},
            "ScaleOutCooldown": 60,    # reaccionar rápido a picos (1.000 TPS)
            "ScaleInCooldown": 300,    # bajar con calma para evitar oscilaciones
        })


def deploy(args: argparse.Namespace, sess=None) -> dict:
    meta = validate_artifacts()
    sess = sess or session(args.region)
    sm, s3 = sess.client("sagemaker"), sess.client("s3")

    bucket = args.bucket or default_bucket(sess, args.region)
    ensure_bucket(s3, bucket, args.region)
    role = resolve_role(sess, args.role)
    image = args.image_uri or xgboost_image_uri(args.region)
    model_version = meta["model_version"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    print(f"[1/4] Subiendo modelo {model_version} a s3://{bucket}")
    model_data = upload_model(s3, bucket, model_version)

    model_name = safe_name(PROJECT, model_version, stamp)
    print(f"[2/4] Creando Model {model_name} ({image})")
    sm.create_model(
        ModelName=model_name, ExecutionRoleArn=role,
        PrimaryContainer={"Image": image, "ModelDataUrl": model_data,
                          "Environment": {"MODEL_VERSION": model_version}},
        Tags=TAGS + [{"Key": "model_version", "Value": model_version}],
    )

    variant = {"VariantName": VARIANT, "ModelName": model_name}
    if args.serverless:
        variant["ServerlessConfig"] = {"MemorySizeInMB": args.serverless_memory,
                                       "MaxConcurrency": args.serverless_concurrency}
    else:
        variant.update(InstanceType=args.instance_type, InitialInstanceCount=args.min_instances,
                       InitialVariantWeight=1.0)
    config_name = safe_name(args.endpoint_name, "cfg", stamp)
    config_kwargs = {"EndpointConfigName": config_name, "ProductionVariants": [variant], "Tags": TAGS}
    if args.data_capture_percent and not args.serverless:
        config_kwargs["DataCaptureConfig"] = {
            "EnableCapture": True,
            "InitialSamplingPercentage": args.data_capture_percent,
            "DestinationS3Uri": f"s3://{bucket}/{PROJECT}/datacapture",
            "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
            "CaptureContentTypeHeader": {"CsvContentTypes": ["text/csv"],
                                         "JsonContentTypes": ["application/json"]},
        }
    print(f"[3/4] Creando EndpointConfig {config_name}")
    sm.create_endpoint_config(**config_kwargs)

    aas = sess.client("application-autoscaling")
    if endpoint_exists(sm, args.endpoint_name):
        print(f"      Actualizando endpoint existente {args.endpoint_name} (blue/green, sin downtime)")
        deregister_autoscaling(aas, args.endpoint_name)
        sm.update_endpoint(EndpointName=args.endpoint_name, EndpointConfigName=config_name)
    else:
        print(f"      Creando endpoint {args.endpoint_name}")
        sm.create_endpoint(EndpointName=args.endpoint_name, EndpointConfigName=config_name, Tags=TAGS)

    if not args.no_wait:
        print("      Esperando estado InService (5-10 min)…")
        sm.get_waiter("endpoint_in_service").wait(
            EndpointName=args.endpoint_name, WaiterConfig={"Delay": 30, "MaxAttempts": 40})

    autoscaling = None
    if args.serverless:
        print("[4/4] Serverless: el escalado lo gestiona SageMaker (MaxConcurrency)")
    elif args.no_wait:
        print("[4/4] --no-wait: registra el auto-scaling luego con `python -m sagemaker_deploy.autoscaling`")
    else:
        print(f"[4/4] Auto-scaling {args.min_instances}-{args.max_instances} instancias, "
              f"objetivo {args.target_invocations} invocaciones/min/instancia")
        try:
            register_autoscaling(aas, args.endpoint_name, args.min_instances,
                                 args.max_instances, args.target_invocations)
            autoscaling = {"min": args.min_instances, "max": args.max_instances,
                           "target_invocations_per_instance_per_min": args.target_invocations}
        except Exception as err:  # p. ej. permisos restringidos en AWS Academy
            print(f"      ⚠ No se pudo registrar el auto-scaling: {err}")

    status = sm.describe_endpoint(EndpointName=args.endpoint_name)["EndpointStatus"]
    deployment = {
        "endpoint_name": args.endpoint_name,
        "region": args.region,
        "status": status,
        "model_name": model_name,
        "endpoint_config_name": config_name,
        "model_version": model_version,
        "model_data": model_data,
        "image_uri": image,
        "execution_role": role,
        "mode": "serverless" if args.serverless else "realtime",
        "instance_type": None if args.serverless else args.instance_type,
        "autoscaling": autoscaling,
        "data_capture_percent": 0 if args.serverless else args.data_capture_percent,
        "threshold": meta["threshold"],
        "review_threshold": meta.get("review_threshold", meta["threshold"]),
        "deployed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    DEPLOYMENT_FILE.write_text(json.dumps(deployment, indent=2))
    print(f"Listo: endpoint {args.endpoint_name} [{status}] → {DEPLOYMENT_FILE.relative_to(MODEL_DIR.parent)}")
    return deployment


def main(argv=None) -> None:
    deploy(parse_args(argv))


if __name__ == "__main__":
    main()
