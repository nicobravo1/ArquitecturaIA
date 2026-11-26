"""Flujo de despliegue contra AWS simulado con moto (sin costo ni credenciales)."""
import json

import pytest

moto = pytest.importorskip("moto")
import boto3  # noqa: E402

from sagemaker_deploy import deploy as dep  # noqa: E402
from sagemaker_deploy import teardown as td  # noqa: E402
from sagemaker_deploy.common import xgboost_image_uri  # noqa: E402

REGION = "us-east-1"


@pytest.fixture
def aws(monkeypatch):
    for k, v in {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing",
                 "AWS_DEFAULT_REGION": REGION}.items():
        monkeypatch.setenv(k, v)
    with moto.mock_aws():
        yield boto3.Session(region_name=REGION)


def test_image_uri_contenedor_xgboost():
    assert xgboost_image_uri("us-east-1") == \
        "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:3.0-5"
    with pytest.raises(SystemExit):
        xgboost_image_uri("xx-nowhere-1")


def test_safe_name_cumple_reglas_de_sagemaker():
    name = dep.safe_name("pagoseguro-fraud", "xgb_2026.11.25", "x" * 80)
    assert len(name) <= 63 and all(c.isalnum() or c == "-" for c in name)


def test_validate_artifacts_rechaza_version_incompatible(model_dir):
    meta = json.loads((model_dir / "metadata.json").read_text())
    meta["framework"]["xgboost"] = "1.7.6"
    (model_dir / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="contenedor"):
        dep.validate_artifacts()


def test_deploy_crea_recursos_y_deployment_json(model_dir, aws):
    info = dep.deploy(dep.parse_args(["--region", REGION]), sess=aws)
    sm, s3 = aws.client("sagemaker"), aws.client("s3")

    assert info["endpoint_name"] == "pagoseguro-fraud"
    assert info["execution_role"].endswith(":role/LabRole")
    bucket, key = info["model_data"][5:].split("/", 1)
    assert s3.head_object(Bucket=bucket, Key=key)["ServerSideEncryption"] == "aws:kms"

    model = sm.describe_model(ModelName=info["model_name"])["PrimaryContainer"]
    assert model["Image"].endswith("sagemaker-xgboost:3.0-5")
    cfg = sm.describe_endpoint_config(EndpointConfigName=info["endpoint_config_name"])
    assert cfg["ProductionVariants"][0]["InstanceType"] == "ml.m5.large"
    assert cfg["DataCaptureConfig"]["InitialSamplingPercentage"] == 20

    targets = aws.client("application-autoscaling").describe_scalable_targets(
        ServiceNamespace="sagemaker")["ScalableTargets"]
    assert targets and targets[0]["MaxCapacity"] == 4
    saved = json.loads((model_dir / "deployment.json").read_text())
    assert saved["threshold"] == 0.6 and saved["autoscaling"]["min"] == 1


def test_redeploy_actualiza_endpoint_y_teardown_limpia(model_dir, aws, monkeypatch):
    args = dep.parse_args(["--region", REGION, "--max-instances", "2"])
    first = dep.deploy(args, sess=aws)

    # moto no implementa UpdateEndpoint: se registra la llamada
    updates = []
    sm = aws.client("sagemaker")
    sm.update_endpoint = lambda **kw: updates.append(kw)
    real_client = aws.client
    monkeypatch.setattr(aws, "client", lambda name, **kw: sm if name == "sagemaker" else real_client(name, **kw))
    monkeypatch.setattr(dep, "datetime", _FutureDatetime)   # nombres únicos en el 2.º deploy

    second = dep.deploy(args, sess=aws)
    assert second["endpoint_config_name"] != first["endpoint_config_name"]
    assert updates == [{"EndpointName": "pagoseguro-fraud",
                        "EndpointConfigName": second["endpoint_config_name"]}]

    deleted = td.teardown(aws, "pagoseguro-fraud")
    assert "endpoint/pagoseguro-fraud" in deleted
    assert not sm.list_endpoint_configs()["EndpointConfigs"]
    assert not sm.list_models()["Models"]
    assert not real_client("application-autoscaling").describe_scalable_targets(
        ServiceNamespace="sagemaker")["ScalableTargets"]


def test_deploy_serverless_sin_autoscaling(model_dir, aws):
    info = dep.deploy(dep.parse_args(["--region", REGION, "--serverless"]), sess=aws)
    cfg = aws.client("sagemaker").describe_endpoint_config(
        EndpointConfigName=info["endpoint_config_name"])
    assert cfg["ProductionVariants"][0]["ServerlessConfig"]["MaxConcurrency"] == 5
    assert info["autoscaling"] is None and info["mode"] == "serverless"


def test_parse_args_valida_capacidades():
    with pytest.raises(SystemExit):
        dep.parse_args(["--min-instances", "3", "--max-instances", "2"])


class _FutureDatetime:
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz

    @classmethod
    def now(cls, tz=None):
        return cls._dt.now(tz) + cls._td(hours=1)


def test_cloudwatch_metrics_endpoint_y_ec2(aws, monkeypatch):
    from datetime import datetime, timedelta, timezone

    from sagemaker_deploy import cloudwatch_metrics as cwm
    # moto no implementa estadísticas percentiles (p99); CloudWatch real sí
    monkeypatch.setattr(cwm, "METRICS", [m for m in cwm.METRICS if not m[2].startswith("p")])
    ts = datetime.now(timezone.utc) - timedelta(minutes=2)
    cw = aws.client("cloudwatch")
    cw.put_metric_data(Namespace="AWS/SageMaker", MetricData=[{
        "MetricName": "Invocations", "Value": 42, "Timestamp": ts,
        "Dimensions": [{"Name": "EndpointName", "Value": "ep"}, {"Name": "VariantName", "Value": "AllTraffic"}]}])
    cw.put_metric_data(Namespace="AWS/EC2", MetricData=[{
        "MetricName": "CPUUtilization", "Value": 37.5, "Timestamp": ts,
        "Dimensions": [{"Name": "InstanceId", "Value": "i-123"}]}])
    rows = cwm.collect(cw, "ep", hours=1, period=60, instance_id="i-123")
    summary = cwm.summarize(rows)
    assert summary["endpoint.Invocations_total_count"] == 42
    assert summary["ec2.CPUUtilization_promedio_%"] == 37.5
    assert summary["ec2.CPUUtilization_peor_minuto_%"] == 37.5


def test_resumen_cloudwatch_distingue_promedio_y_peor_minuto():
    from sagemaker_deploy.cloudwatch_metrics import summarize
    rows = [{"source": "endpoint", "metric": "OverheadLatency", "stat": "Average", "unit": "ms", "value": v}
            for v in (2.0, 3.0, 55.0)]
    rows += [{"source": "endpoint", "metric": "ModelLatency", "stat": "p99", "unit": "ms", "value": v}
             for v in (10.0, 20.0)]
    s = summarize(rows)
    assert s["endpoint.OverheadLatency_promedio_ms"] == 20.0
    assert s["endpoint.OverheadLatency_peor_minuto_ms"] == 55.0
    assert s["endpoint.ModelLatency_p99_peor_minuto_ms"] == 20.0
