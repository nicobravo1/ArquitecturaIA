"""Pruebas funcionales de la API FastAPI (backend local, sin AWS ni RabbitMQ)."""
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.app.config import Settings  # noqa: E402
from api.app.main import create_app  # noqa: E402
from api.app.scorers import LocalBackend, ScoringService  # noqa: E402
from fraud_core.scoring import Thresholds as Thresholds_  # noqa: E402

KEY = "test-key-123"
H = {"X-API-Key": KEY}


class FakePublisher:
    def __init__(self, ok=True):
        self.ok, self.events, self.connected = ok, [], ok

    async def start(self):
        pass

    async def publish(self, event):
        self.events.append(event)
        return self.ok

    async def close(self):
        pass


class BrokenBackend:
    name = "sagemaker"

    def predict_proba(self, features):
        raise TimeoutError("endpoint no responde")


@pytest.fixture
def samples(model_dir):
    return json.loads((model_dir / "sample_payloads.json").read_text())


def make_client(model_dir, publisher=None, scoring=None, audit_store=None, **overrides):
    cfg = dict(model_dir=model_dir, scorer_backend="local", api_keys=frozenset({KEY}),
               rate_limit_per_minute=1000, rabbitmq_url=None, database_url=None)
    cfg.update(overrides)
    app = create_app(Settings(**cfg), scoring=scoring, publisher=publisher or FakePublisher(),
                     audit_store=audit_store)
    return TestClient(app)


@pytest.fixture
def client(model_dir):
    with make_client(model_dir) as c:
        yield c


# 1
def test_health_sin_autenticacion(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "xgb-test-0001", "scorer": "local",
                        "rabbitmq": "disabled", "audit_store": "disabled"}


# 2
@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "otra"}])
def test_score_rechaza_sin_api_key_valida(client, samples, headers):
    r = client.post("/v1/transactions/score", json=samples["legit"][0], headers=headers)
    assert r.status_code == 401 and r.headers["WWW-Authenticate"] == "ApiKey"


# 3
def test_score_legitima_aprobada_y_fraude_rechazado(client, samples):
    legit = client.post("/v1/transactions/score", json=samples["legit"][0], headers=H).json()
    fraud = client.post("/v1/transactions/score", json=samples["fraud"][0], headers=H).json()
    assert legit["decision"] == "APPROVE" and fraud["decision"] == "REJECT"
    assert fraud["fraud_probability"] > fraud["threshold"] > legit["fraud_probability"]
    assert fraud["model_version"] == "xgb-test-0001" and fraud["scorer"] == "local"


# 4
def test_score_coincide_con_modelo_local(client, samples, model_dir):
    from fraud_core import build_features
    expected = LocalBackend(model_dir / "xgboost-model").predict_proba(build_features(samples["fraud"][0]))[0]
    got = client.post("/v1/transactions/score", json=samples["fraud"][0], headers=H).json()
    assert got["fraud_probability"] == pytest.approx(expected, abs=1e-6)


# 5
@pytest.mark.parametrize("mutate,field", [
    (lambda t: t.pop("V5"), "V5"),
    (lambda t: t.update(Amount=-10), "Amount"),
    (lambda t: t.update(Amount="mucho"), "Amount"),
    (lambda t: t.update(extra_field=1), "extra_field"),
    (lambda t: t.update(card_number="1234"), "card_number"),
])
def test_score_valida_payload(client, samples, mutate, field):
    tx = dict(samples["legit"][0]); mutate(tx)
    r = client.post("/v1/transactions/score", json=tx, headers=H)
    assert r.status_code == 422
    assert any(field in map(str, e["loc"]) for e in r.json()["detail"])


# 6
def test_pan_enmascarado_y_nunca_publicado(model_dir, samples):
    pub = FakePublisher()
    with make_client(model_dir, publisher=pub) as c:
        tx = samples["fraud"][0] | {"card_number": "4111 1111 1111 1234", "transaction_id": "tx-1"}
        r = c.post("/v1/transactions/score", json=tx, headers=H).json()
    assert r["card_masked"] == "************1234"
    raw_event = json.dumps(pub.events)
    assert "4111111111111234" not in raw_event and "************1234" in raw_event


# 7
def test_evento_publicado_para_worker_shap(model_dir, samples):
    pub = FakePublisher()
    with make_client(model_dir, publisher=pub) as c:
        r = c.post("/v1/transactions/score", json=samples["fraud"][0] | {"transaction_id": "tx-42"}, headers=H).json()
    assert r["explanation_status"] == "queued" and r["transaction_id"] == "tx-42"
    (ev,) = pub.events
    assert ev["transaction_id"] == "tx-42" and ev["decision"] == "REJECT"
    assert set(ev["features"]) >= {"Time", "Amount", "V1", "V28"}


# 8
def test_broker_caido_no_bloquea_el_pago(model_dir, samples):
    with make_client(model_dir, publisher=FakePublisher(ok=False)) as c:
        r = c.post("/v1/transactions/score", json=samples["legit"][0], headers=H)
    assert r.status_code == 200 and r.json()["explanation_status"] == "unavailable"


# 9
def test_rate_limit_devuelve_429(model_dir, samples):
    with make_client(model_dir, rate_limit_per_minute=3) as c:
        codes = [c.post("/v1/transactions/score", json=samples["legit"][0], headers=H).status_code
                 for _ in range(4)]
        last = c.post("/v1/transactions/score", json=samples["legit"][0], headers=H)
    assert codes == [200, 200, 200, 429]
    assert int(last.headers["Retry-After"]) >= 1


# 10
def test_batch_y_limite_de_tamano(model_dir, samples):
    txs = (samples["legit"] + samples["fraud"]) * 2
    with make_client(model_dir, max_batch_size=2) as c:
        ok = c.post("/v1/transactions/score/batch", json={"transactions": txs[:2]}, headers=H)
        too_big = c.post("/v1/transactions/score/batch", json={"transactions": txs[:3]}, headers=H)
        empty = c.post("/v1/transactions/score/batch", json={"transactions": []}, headers=H)
    assert ok.status_code == 200 and ok.json()["count"] == 2
    assert too_big.status_code == 413 and empty.status_code == 422


# 11
def test_fallback_local_si_sagemaker_falla(model_dir, samples):
    scoring = ScoringService(BrokenBackend(), LocalBackend(model_dir / "xgboost-model"))
    with make_client(model_dir, scoring=scoring) as c:
        r = c.post("/v1/transactions/score", json=samples["fraud"][0], headers=H)
    assert r.status_code == 200 and r.json()["scorer"] == "local-fallback"


# 12
def test_503_si_sagemaker_falla_sin_fallback(model_dir, samples):
    with make_client(model_dir, scoring=ScoringService(BrokenBackend())) as c:
        r = c.post("/v1/transactions/score", json=samples["fraud"][0], headers=H)
    assert r.status_code == 503


# 13
def test_model_info_y_cabeceras(client):
    r = client.get("/v1/model", headers=H | {"X-Request-ID": "abc"})
    body = r.json()
    assert body["threshold"] == 0.6 and body["review_threshold"] == 0.3 and len(body["feature_columns"]) == 30
    assert r.headers["X-Request-ID"] == "abc" and "X-Process-Time-Ms" in r.headers
    assert r.headers["X-RateLimit-Limit"] == "1000"


# 14
def test_metricas_prometheus(client, samples):
    client.post("/v1/transactions/score", json=samples["fraud"][0], headers=H)
    text = client.get("/metrics").text
    assert 'fraud_api_decisions_total{decision="REJECT",scorer="local"} 1.0' in text
    assert "fraud_api_request_latency_seconds_bucket" in text


# 15
def test_openapi_documenta_endpoints_y_seguridad(client):
    spec = client.get("/openapi.json").json()
    assert {"/v1/transactions/score", "/v1/transactions/score/batch", "/v1/model", "/health"} <= set(spec["paths"])
    assert "APIKeyHeader" in spec["components"]["securitySchemes"]
    assert client.get("/docs").status_code == 200


# 16
def test_sin_api_keys_limita_por_ip(model_dir, samples):
    with make_client(model_dir, api_keys=frozenset(), rate_limit_per_minute=1) as c:
        first = c.post("/v1/transactions/score", json=samples["legit"][0])
        second = c.post("/v1/transactions/score", json=samples["legit"][0])
    assert first.status_code == 200 and second.status_code == 429


# 17
def test_consultas_de_auditoria(model_dir, audit_store, samples):
    from datetime import datetime, timezone
    from fraud_core import RAW_COLUMNS
    now = datetime.now(timezone.utc)
    base = {"scored_at": now, "explained_at": now, "threshold": 0.6, "review_threshold": 0.3,
            "model_version": "xgb-test-0001", "features": {c: 1.0 for c in RAW_COLUMNS}, "amount": 1.0}
    audit_store.save(base | {"transaction_id": "tx-r", "decision": "REJECT", "fraud_probability": 0.95,
                             "shap_values": {"V1": 2.0}, "shap_base_value": -5.0,
                             "top_reasons": [{"feature": "V1", "label": "V1", "value": 1.0, "shap": 2.0,
                                              "direction": "aumenta riesgo"}],
                             "explanation_text": "Rechazada: ..."})
    audit_store.save(base | {"transaction_id": "tx-a", "decision": "APPROVE", "fraud_probability": 0.01})
    with make_client(model_dir, audit_store=audit_store) as c:
        exp = c.get("/v1/transactions/tx-r/explanation", headers=H)
        missing = c.get("/v1/transactions/nope/explanation", headers=H)
        rejected = c.get("/v1/transactions", params={"decision": "REJECT"}, headers=H).json()
        stats = c.get("/v1/stats", params={"hours": 1}, headers=H).json()
        health = c.get("/health").json()
    assert exp.status_code == 200 and exp.json()["top_reasons"][0]["feature"] == "V1"
    assert exp.json()["shap_values"] == {"V1": 2.0}
    assert missing.status_code == 404
    assert [t["transaction_id"] for t in rejected] == ["tx-r"]
    assert stats["total"] == 2 and stats["reject_rate"] == 0.5
    assert health["audit_store"] == "connected"


# 18
def test_auditoria_no_configurada_responde_503(client):
    assert client.get("/v1/transactions/x/explanation", headers=H).status_code == 503
    assert client.get("/v1/transactions?limit=0", headers=H).status_code == 422


# 19
def test_inplace_predict_igual_a_dmatrix(model_dir, samples):
    import xgboost as xgb

    from fraud_core import FEATURE_COLUMNS, build_features, features_matrix
    txs = samples["legit"] + samples["fraud"]
    backend = LocalBackend(model_dir / "xgboost-model")
    fast = backend.predict_proba(features_matrix(txs))
    slow = backend.booster.predict(xgb.DMatrix(build_features(txs), feature_names=FEATURE_COLUMNS))
    assert fast == pytest.approx(slow.tolist(), abs=1e-7)


# 20
def test_regla_monto_alto_en_la_api(model_dir, samples):
    meta = json.loads((model_dir / "metadata.json").read_text())
    (model_dir / "metadata.json").write_text(json.dumps(meta | {"high_amount_review": 500.0}))
    pub = FakePublisher()
    with make_client(model_dir, publisher=pub) as c:
        normal = c.post("/v1/transactions/score", json=samples["legit"][0], headers=H).json()
        alto = c.post("/v1/transactions/score", json=samples["legit"][0] | {"Amount": 800.0}, headers=H).json()
        info = c.get("/v1/model", headers=H).json()
    assert normal["decision"] == "APPROVE" and normal["rule"] is None
    assert alto["decision"] == "REVIEW" and alto["rule"] == "high_amount"
    assert pub.events[1]["rule"] == "high_amount" and info["high_amount_review"] == 500.0


# 21
def test_pools_de_io_configurables(model_dir):
    import asyncio
    with make_client(model_dir, io_threads=40) as c:
        executor = c.portal.call(lambda: asyncio.get_running_loop()._default_executor)
    assert executor._max_workers == 40


def test_sagemaker_scorer_pool_de_conexiones():
    from fraud_core.scoring import SageMakerScorer
    sc = SageMakerScorer("ep", Thresholds_(0.6, 0.3), region_name="us-east-1", max_connections=77)
    assert sc.runtime.meta.config.max_pool_connections == 77
