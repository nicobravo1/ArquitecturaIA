import json
import tarfile

import numpy as np
import pandas as pd
import pytest

from fraud_core import FEATURE_COLUMNS, RAW_COLUMNS


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    """Artefactos mínimos (como los del notebook) en un directorio temporal."""
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = (X["V1"] + rng.normal(scale=0.5, size=400) > 1.2).astype(int)
    booster = xgb.train({"objective": "binary:logistic", "max_depth": 2},
                        xgb.DMatrix(X, label=y), num_boost_round=5)
    booster.save_model(tmp_path / "xgboost-model.json")
    (tmp_path / "xgboost-model.json").rename(tmp_path / "xgboost-model")
    with tarfile.open(tmp_path / "model.tar.gz", "w:gz") as tar:
        tar.add(tmp_path / "xgboost-model", arcname="xgboost-model")
    meta = {"model_version": "xgb-test-0001", "feature_columns": FEATURE_COLUMNS,
            "threshold": 0.6, "review_threshold": 0.3,
            "framework": {"xgboost": xgb.__version__}}
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    tx = {c: 0.0 for c in RAW_COLUMNS} | {"Amount": 10.0}
    (tmp_path / "sample_payloads.json").write_text(json.dumps(
        {"legit": [tx | {"V1": -2.0}], "fraud": [tx | {"V1": 3.0}]}))

    import sagemaker_deploy.common as common
    for mod in ("sagemaker_deploy.deploy", "sagemaker_deploy.invoke_test",
                "sagemaker_deploy.teardown", "sagemaker_deploy.common"):
        m = __import__(mod, fromlist=["_"])
        monkeypatch.setattr(m, "MODEL_DIR", tmp_path, raising=False)
        monkeypatch.setattr(m, "DEPLOYMENT_FILE", tmp_path / "deployment.json", raising=False)
    return tmp_path


@pytest.fixture(params=["sqlite", "postgresql"])
def audit_store(request, tmp_path):
    """El almacén de auditoría sobre SQLite y, si DATABASE_URL apunta a
    PostgreSQL, también sobre PostgreSQL (esquema temporal aislado)."""
    import os
    import uuid

    from sqlalchemy import text

    from fraud_core import audit_store as mod

    if request.param == "sqlite":
        store = mod.AuditStore(f"sqlite:///{tmp_path / 'audit.db'}")
        store.create_schema()
        yield store
        return
    url = os.environ.get("DATABASE_URL", "")
    if not url.startswith("postgresql"):
        pytest.skip("DATABASE_URL de PostgreSQL no definido")
    schema = f"test_{uuid.uuid4().hex[:8]}"
    store = mod.AuditStore(url, connect_args={"options": f"-csearch_path={schema}"})
    with store.engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    store.create_schema()
    yield store
    with store.engine.begin() as conn:
        conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
    store.engine.dispose()


@pytest.fixture
def scored_event(model_dir):
    """Evento tal como lo publica la API."""
    def make(kind="fraud", **overrides):
        samples = json.loads((model_dir / "sample_payloads.json").read_text())
        ev = {"event_type": "transaction.scored", "transaction_id": f"tx-{kind}",
              "scored_at": "2026-11-25T12:00:00+00:00", "decision": "REJECT" if kind == "fraud" else "APPROVE",
              "fraud_probability": 0.9 if kind == "fraud" else 0.01, "threshold": 0.6,
              "review_threshold": 0.3, "model_version": "xgb-test-0001", "scorer": "local",
              "client": "abc", "card_masked": "************1234", "features": samples[kind][0]}
        ev.update(overrides)
        return ev
    return make
