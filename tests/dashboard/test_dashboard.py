"""Dashboard: gráficos, cliente HTTP y la app Streamlit completa contra una API real
(en proceso, backend local + auditoría SQLite)."""
import json
import socket
import threading
import time
from datetime import datetime, timedelta, timezone

import httpx
import pytest

pytest.importorskip("streamlit")
from dashboard import charts  # noqa: E402
from dashboard.client import ApiError, FraudApiClient  # noqa: E402
from fraud_core import FEATURE_COLUMNS, RAW_COLUMNS  # noqa: E402

from pathlib import Path  # noqa: E402

APP = Path(__file__).resolve().parents[2] / "dashboard" / "app.py"


def _tx(i, decision, minutes_ago, p):
    return {"transaction_id": f"t{i}", "decision": decision, "fraud_probability": p,
            "scored_at": (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()}


# ── gráficos ──
def test_pick_bucket_adapta_el_intervalo():
    import pandas as pd
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    assert charts.pick_bucket(pd.Series([t0, t0 + pd.Timedelta(seconds=90)])) == "10s"
    assert charts.pick_bucket(pd.Series([t0, t0 + pd.Timedelta(hours=6)])) == "15min"
    assert charts.pick_bucket(pd.Series([t0, t0 + pd.Timedelta(days=60)])) == "1D"


def test_decisiones_en_el_tiempo_usa_colores_de_estado_y_rellena_ceros():
    txs = [_tx(0, "APPROVE", 30, .01), _tx(1, "REJECT", 30, .9), _tx(2, "APPROVE", 0, .02)]
    fig, freq = charts.decisions_over_time(txs)
    assert freq == "1min"
    by_name = {t.name: t for t in fig.data}
    assert by_name["⛔ Rechazada"].line.color == charts.STATUS["REJECT"]
    assert len(by_name["✅ Aprobada"].y) == 31 and sum(by_name["✅ Aprobada"].y) == 2


def test_shap_ordena_filtra_y_colorea_por_polaridad():
    exp = {"shap_values": {f: 0.0 for f in FEATURE_COLUMNS} | {"V14": 2.0, "V4": -1.0, "Amount": 0.001},
           "features": {c: 1.0 for c in RAW_COLUMNS} | {"Time": 7200.0}}
    bar = charts.shap_contributions(exp).data[0]
    assert list(bar.y) == ["V4 = 1.00", "V14 = 1.00"]         # sin despreciables, mayor arriba
    assert list(bar.marker.color) == [charts.RISK_DOWN, charts.RISK_UP]


# ── cliente ──
def test_cliente_mapea_errores():
    def handler(req):
        if req.url.path.endswith("/explanation"):
            return httpx.Response(404, json={"detail": "no"})
        return httpx.Response(401, json={"detail": "API key inválida"})
    c = FraudApiClient("http://api", "k", transport=httpx.MockTransport(handler))
    assert c.explanation("x") is None
    with pytest.raises(ApiError) as err:
        c.stats(24)
    assert err.value.status == 401 and "inválida" in err.value.detail


def test_cliente_sin_conexion():
    with pytest.raises(ApiError) as err:
        FraudApiClient("http://127.0.0.1:9", timeout=0.5).health()
    assert err.value.status == 0


# ── app completa ──
@pytest.fixture
def live_api(model_dir, audit_store):
    if audit_store.engine.dialect.name != "sqlite":
        pytest.skip("basta con SQLite para el dashboard")
    import uvicorn

    from api.app.config import Settings
    from api.app.main import create_app

    samples = json.loads((model_dir / "sample_payloads.json").read_text())
    now = datetime.now(timezone.utc)
    audit_store.save({"transaction_id": "tx-rej", "scored_at": now, "explained_at": now, "decision": "REJECT",
                      "fraud_probability": 0.93, "threshold": 0.6, "review_threshold": 0.3,
                      "model_version": "xgb-test-0001", "amount": 12.5,
                      "features": samples["fraud"][0], "shap_base_value": -3.0,
                      "shap_values": {f: 0.0 for f in FEATURE_COLUMNS} | {"V1": 2.5},
                      "top_reasons": [{"feature": "V1", "label": "V1", "value": 3.0, "shap": 2.5,
                                       "direction": "aumenta riesgo"}],
                      "explanation_text": "Rechazada: prueba", "explain_ms": 3.0})
    app = create_app(Settings(model_dir=model_dir, scorer_backend="local", api_keys=frozenset({"k"}),
                              rate_limit_per_minute=1000, rabbitmq_url=None, database_url=None),
                     audit_store=audit_store)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True); t.start()
    while not server.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    t.join(5)


def test_app_streamlit_renderiza_todas_las_pestanas(live_api, model_dir, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("API_URL", live_api)
    monkeypatch.setenv("API_KEY", "k")
    monkeypatch.setenv("SAMPLES_PATH", str(model_dir / "sample_payloads.json"))
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Transacciones"] == "1" and metrics["⛔ Rechazada · 1 tx"] == "100.00%"
    assert any("Rechazada: prueba" in i.value for i in at.info)            # pestaña Explicaciones
    assert metrics["Umbral de rechazo"] in ("60.00%", "0.600")             # pestaña Modelo

    # Probar transacción: fraude del set de test → REJECT
    next(s for s in at.selectbox if s.label == "Partir de").select("Fraude del set de test #1").run()
    at.button[0].click().run()
    assert not at.exception
    assert {m.label: m.value for m in at.metric}["Decisión"] == "⛔ Rechazada"


def test_app_muestra_error_si_la_api_no_responde(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert any("API no disponible" in e.value for e in at.error)
