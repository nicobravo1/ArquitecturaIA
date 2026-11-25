"""Flujo completo con servicios reales:

    API (score) → RabbitMQ → worker SHAP → PostgreSQL → API (explanation)

    RABBITMQ_URL=amqp://guest:guest@localhost:5672/ \\
    DATABASE_URL=postgresql+psycopg://fraud:fraud@localhost:5432/fraud_audit \\
    pytest -m integration
"""
import asyncio
import json
import os
import threading
import time
import uuid

import pytest

pytestmark = pytest.mark.integration
RABBIT, DB = os.environ.get("RABBITMQ_URL"), os.environ.get("DATABASE_URL")
if not (RABBIT and DB):
    pytest.skip("Requiere RABBITMQ_URL y DATABASE_URL", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from api.app.config import Settings  # noqa: E402
from api.app.main import create_app  # noqa: E402
from fraud_core.audit_store import AuditStore  # noqa: E402
from worker.app.config import WorkerSettings  # noqa: E402
from worker.app.main import Worker  # noqa: E402


@pytest.fixture
def running_worker(model_dir):
    store = AuditStore(DB)
    store.create_schema()
    worker = Worker(WorkerSettings(rabbitmq_url=RABBIT, database_url=DB, model_dir=model_dir), store=store)
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_until_complete, args=(worker.run(),), daemon=True)
    t.start()
    time.sleep(1.0)
    yield store
    loop.call_soon_threadsafe(worker.stop)
    t.join(timeout=10)


def test_score_explicacion_en_postgres(model_dir, running_worker):
    samples = json.loads((model_dir / "sample_payloads.json").read_text())
    run = uuid.uuid4().hex[:8]
    settings = Settings(model_dir=model_dir, scorer_backend="local", api_keys=frozenset({"k"}),
                        rate_limit_per_minute=100, rabbitmq_url=RABBIT, database_url=DB,
                        rabbitmq_publish_timeout_ms=1000)
    h = {"X-API-Key": "k"}
    with TestClient(create_app(settings)) as c:
        assert c.get("/health").json()["audit_store"] == "connected"
        fraud = c.post("/v1/transactions/score", headers=h,
                       json=samples["fraud"][0] | {"transaction_id": f"e2e-f-{run}"}).json()
        legit = c.post("/v1/transactions/score", headers=h,
                       json=samples["legit"][0] | {"transaction_id": f"e2e-l-{run}"}).json()
        assert fraud["decision"] == "REJECT" and fraud["explanation_status"] == "queued"

        def wait_for(tx_id, timeout=10):
            deadline = time.time() + timeout
            while time.time() < deadline:
                r = c.get(f"/v1/transactions/{tx_id}/explanation", headers=h)
                if r.status_code == 200:
                    return r.json()
                time.sleep(0.2)
            pytest.fail(f"el worker no generó la explicación de {tx_id} a tiempo")

        exp = wait_for(f"e2e-f-{run}")
        approved = wait_for(f"e2e-l-{run}")
        stats = c.get("/v1/stats", headers=h).json()

    assert exp["fraud_probability"] == pytest.approx(fraud["fraud_probability"], abs=1e-6)
    assert exp["top_reasons"][0]["feature"] == "V1" and exp["explanation_text"].startswith("Rechazada")
    assert len(exp["shap_values"]) == 30
    # APPROVE se registra sin SHAP (EXPLAIN_DECISIONS=REJECT,REVIEW por defecto)
    assert approved["decision"] == "APPROVE" and approved["shap_values"] is None
    assert stats["total"] >= 2
