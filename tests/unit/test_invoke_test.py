import io
import json

import pytest

xgb = pytest.importorskip("xgboost")
from fraud_core import FEATURE_COLUMNS  # noqa: E402
from sagemaker_deploy import invoke_test  # noqa: E402


class LocalModelRuntime:
    """Simula el contenedor XGBoost de SageMaker usando el modelo local."""

    def __init__(self, model_path, offset=0.0):
        self.booster = xgb.Booster()
        self.booster.load_model(model_path)
        self.offset = offset

    def invoke_endpoint(self, Body, **_):
        rows = [[float(v) for v in line.split(",")] for line in Body.decode().splitlines()]
        preds = self.booster.predict(xgb.DMatrix(rows, feature_names=FEATURE_COLUMNS))
        body = json.dumps({"predictions": [{"score": min(1.0, float(p) + self.offset)} for p in preds]})
        return {"Body": io.BytesIO(body.encode()), "ContentType": "application/json"}


def test_invoke_test_coincide_con_modelo_local(model_dir):
    stats = invoke_test.run("ep", "us-east-1", 20, runtime_client=LocalModelRuntime(model_dir / "xgboost-model"))
    assert stats["requests"] == 20 and stats["max_abs_diff_vs_local"] < 1e-6
    report = json.loads((model_dir / "endpoint_test.json").read_text())
    assert {r["decision"] for r in report["results"]} <= {"APPROVE", "REVIEW", "REJECT"}


def test_invoke_test_detecta_modelo_distinto(model_dir):
    with pytest.raises(SystemExit, match="difiere"):
        invoke_test.run("ep", "us-east-1", 1,
                        runtime_client=LocalModelRuntime(model_dir / "xgboost-model", offset=0.05))
