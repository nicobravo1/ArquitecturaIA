import asyncio
import json

import pytest

pytest.importorskip("xgboost")
from worker.app.config import WorkerSettings  # noqa: E402
from worker.app.explainer import ShapExplainer, explanation_text  # noqa: E402
from worker.app.main import Worker  # noqa: E402
from worker.app.processor import InvalidEvent, build_record  # noqa: E402

ALL = frozenset({"APPROVE", "REVIEW", "REJECT"})


@pytest.fixture
def explainer(model_dir):
    return ShapExplainer(model_dir)


def test_shap_aditivo_reproduce_la_probabilidad(explainer, scored_event):
    from fraud_core import build_features
    feats = scored_event("fraud")["features"]
    exp = explainer.explain(feats, top_k=3)
    assert exp["shap_probability"] == pytest.approx(explainer.probability(build_features(feats)), abs=1e-5)
    assert len(exp["top_reasons"]) == 3 and len(exp["shap_values"]) == 30
    mags = [abs(r["shap"]) for r in exp["top_reasons"]]
    assert mags == sorted(mags, reverse=True)
    assert exp["top_reasons"][0]["feature"] == "V1"   # única variable informativa del modelo de prueba


def test_texto_de_explicacion():
    reasons = [{"label": "V14", "value": -4.2, "shap": 2.1}, {"label": "Monto", "value": 5.0, "shap": -0.3}]
    t = explanation_text("REJECT", 0.93, 0.6, reasons)
    assert t.startswith("Rechazada: probabilidad de fraude 93.0%")
    assert "aumentan el riesgo: V14=-4.20 (+2.10)" in t and "lo reducen: Monto=5.00 (-0.30)" in t


def test_approve_se_guarda_sin_shap_por_defecto(explainer, scored_event):
    r = build_record(scored_event("legit"), explainer, frozenset({"REJECT", "REVIEW"}))
    assert r["decision"] == "APPROVE" and r["shap_values"] is None and r["amount"] == 10.0
    r2 = build_record(scored_event("fraud"), explainer, frozenset({"REJECT", "REVIEW"}))
    assert r2["shap_values"] and r2["explanation_text"].startswith("Rechazada")


@pytest.mark.parametrize("mutate,msg", [
    (lambda e: e.pop("features"), "Faltan campos"),
    (lambda e: e.update(decision="MAYBE"), "Decisión inválida"),
    (lambda e: e["features"].pop("V3"), "incompletas"),
    (lambda e: e["features"].update(card_number="4111111111111111"), "PAN"),
    (lambda e: e.update(model_version="otro"), "Versión de modelo"),
])
def test_eventos_invalidos(explainer, scored_event, mutate, msg):
    ev = scored_event("fraud"); mutate(ev)
    with pytest.raises(InvalidEvent, match=msg):
        build_record(ev, explainer, ALL)


class FakeMessage:
    def __init__(self, body, headers=None):
        self.body = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.headers, self.message_id, self.content_type = headers or {}, "m1", "application/json"
        self.acked = self.rejected = False

    async def ack(self):
        self.acked = True

    async def reject(self, requeue=False):
        self.rejected = True


class FakeChannel:
    def __init__(self):
        self.published = []
        ch = self

        class _Ex:
            async def publish(self, msg, routing_key):
                ch.published.append((msg, routing_key))
        self.default_exchange = _Ex()


class FailingStore:
    def save(self, record):
        raise ConnectionError("BD caída")


def _worker(model_dir, store, explainer, **kw):
    return Worker(WorkerSettings(model_dir=model_dir, database_url="sqlite://", **kw), store=store, explainer=explainer)


def test_on_message_guarda_y_hace_ack_idempotente(model_dir, audit_store, explainer, scored_event):
    w = _worker(model_dir, audit_store, explainer)
    m1, m2 = FakeMessage(scored_event("fraud")), FakeMessage(scored_event("fraud"))
    assert asyncio.run(w.on_message(m1)) == "stored" and m1.acked
    assert asyncio.run(w.on_message(m2)) == "duplicate" and m2.acked
    saved = audit_store.get("tx-fraud")
    assert saved["top_reasons"][0]["feature"] == "V1" and saved["card_masked"] == "************1234"


@pytest.mark.parametrize("body", [b"{no es json", {"foo": 1}])
def test_mensaje_invalido_va_a_dlq(model_dir, audit_store, explainer, body):
    msg = FakeMessage(body)
    assert asyncio.run(_worker(model_dir, audit_store, explainer).on_message(msg)) == "invalid"
    assert msg.rejected and not msg.acked


def test_error_transitorio_reintenta_y_luego_dlq(model_dir, explainer, scored_event, monkeypatch):
    async def no_sleep(_):
        pass
    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    w = _worker(model_dir, FailingStore(), explainer, max_retries=2)
    ch = FakeChannel()
    first = FakeMessage(scored_event("fraud"))
    assert asyncio.run(w.on_message(first, ch)) == "retried" and first.acked
    (retry, key), = ch.published
    assert key == "shap.explanations" and retry.headers["x-retries"] == 1

    last = FakeMessage(scored_event("fraud"), headers={"x-retries": 2})
    assert asyncio.run(w.on_message(last, ch)) == "failed" and last.rejected
