"""Despliegue en EC2 contra AWS simulado (moto)."""
import json
import shutil
import subprocess

import pytest

moto = pytest.importorskip("moto")
import boto3  # noqa: E402

from infra.aws import ec2 as mod  # noqa: E402

REGION = "us-east-1"


@pytest.fixture
def aws(monkeypatch, model_dir, tmp_path):
    for k, v in {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing",
                 "AWS_DEFAULT_REGION": REGION}.items():
        monkeypatch.setenv(k, v)
    (model_dir / "deployment.json").write_text(json.dumps({
        "endpoint_name": "pagoseguro-fraud", "model_version": "xgb-test-0001", "region": REGION,
        "model_data": "s3://sagemaker-us-east-1-123456789012/pagoseguro-fraud/models/x/model.tar.gz"}))
    state_dir = tmp_path / "aws"
    state_dir.mkdir()
    monkeypatch.setattr(mod, "MODEL_DIR", model_dir)
    monkeypatch.setattr(mod, "DEPLOYMENT_FILE", model_dir / "deployment.json")
    monkeypatch.setattr(mod, "HERE", state_dir)
    monkeypatch.setattr(mod, "STATE_FILE", state_dir / ".state.json")
    monkeypatch.setattr(mod, "SECRETS_FILE", state_dir / ".secrets.env")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "my_public_ip", lambda: "203.0.113.7")
    with moto.mock_aws():
        sess = boto3.Session(region_name=REGION)
        sess.client("iam").create_instance_profile(InstanceProfileName="LabInstanceProfile")
        yield sess


def _args(*extra):
    import argparse  # noqa: F401
    ns = mod.argparse.Namespace(region=REGION, instance_type="t3.large", instance_profile="LabInstanceProfile",
                                key_name="pagoseguro-fraud-key", volume_gb=30, public_cidr="0.0.0.0/0",
                                ssh_cidr="auto", repo_url="https://github.com/acme/repo.git", branch="main")
    for k, v in extra:
        setattr(ns, k, v)
    return ns


def test_up_crea_instancia_segura(aws, model_dir):
    state = mod.up(_args(), sess=aws)
    ec2 = aws.client("ec2")
    inst = ec2.describe_instances(InstanceIds=[state["instance_id"]])["Reservations"][0]["Instances"][0]
    assert inst["InstanceType"] == "t3.large"
    assert inst["IamInstanceProfile"]["Arn"].endswith("instance-profile/LabInstanceProfile")
    assert inst["MetadataOptions"]["HttpTokens"] == "required"
    assert inst["MetadataOptions"]["HttpPutResponseHopLimit"] == 2

    sg = ec2.describe_security_groups(GroupIds=[state["security_group_id"]])["SecurityGroups"][0]
    rules = {(r["FromPort"], r["IpRanges"][0]["CidrIp"]) for r in sg["IpPermissions"]}
    assert rules == {(8443, "0.0.0.0/0"), (8444, "0.0.0.0/0"), (22, "203.0.113.7/32")}

    s3 = aws.client("s3")
    keys = {o["Key"] for o in s3.list_objects_v2(Bucket="sagemaker-us-east-1-123456789012")["Contents"]}
    assert {f"pagoseguro-fraud/serving/xgb-test-0001/{f}" for f in mod.MODEL_FILES} <= keys

    secrets = (mod.SECRETS_FILE.read_text())
    assert "CLIENT_API_KEY=ps_" in secrets and oct(mod.SECRETS_FILE.stat().st_mode)[-3:] == "600"
    assert (mod.HERE / "pagoseguro-fraud-key.pem").exists()


def test_user_data_valido_y_completo(aws):
    values = mod.generate_secrets("pagoseguro-fraud", REGION)
    ud = mod.render_user_data("https://github.com/acme/repo.git", "main", "s3://b/p/", values)
    assert "__" not in ud.split("ENVEOF")[1]                       # sin placeholders en el .env
    assert f"API_KEYS={values['API_KEYS']}" in ud and "SCORER_BACKEND=sagemaker" in ud
    assert "aws s3 cp \"$MODEL_S3\"" in ud and 'MODEL_S3="s3://b/p/"' in ud
    if shutil.which("bash"):
        r = subprocess.run(["bash", "-n"], input=ud, text=True, capture_output=True)
        assert r.returncode == 0, r.stderr


def test_secretos_se_reutilizan_entre_despliegues(aws):
    first = mod.generate_secrets("ep", REGION)
    second = mod.generate_secrets("ep", REGION)
    assert first["CLIENT_API_KEY"] == second["CLIENT_API_KEY"]
    assert first["POSTGRES_PASSWORD"] == second["POSTGRES_PASSWORD"]


def test_up_rechaza_versiones_inconsistentes(aws, model_dir):
    dep = json.loads((model_dir / "deployment.json").read_text())
    dep["model_version"] = "otra"
    (model_dir / "deployment.json").write_text(json.dumps(dep))
    with pytest.raises(SystemExit, match="versiones distintas"):
        mod.up(_args(), sess=aws)


def test_down_limpia_todo(aws):
    state = mod.up(_args(("ssh_cidr", None)), sess=aws)
    with pytest.raises(SystemExit, match="Ya existe"):
        mod.up(_args(), sess=aws)
    deleted = mod.down(None, sess=aws)
    ec2 = aws.client("ec2")
    assert f"instance/{state['instance_id']}" in deleted
    assert not ec2.describe_key_pairs()["KeyPairs"]
    assert not mod.STATE_FILE.exists()
    groups = [g["GroupName"] for g in ec2.describe_security_groups()["SecurityGroups"]]
    assert "pagoseguro-fraud-sg" not in groups
