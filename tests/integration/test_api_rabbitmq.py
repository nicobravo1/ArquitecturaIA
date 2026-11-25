"""API → RabbitMQ real → cola del worker SHAP.

    RABBITMQ_URL=amqp://guest:guest@localhost:5672/ pytest -m integration
"""
import asyncio
import json
import os

import pytest

pytestmark = pytest.mark.integration
URL = os.environ.get("RABBITMQ_URL")
if not URL:
    pytest.skip("RABBITMQ_URL no definido", allow_module_level=True)

aio_pika = pytest.importorskip("aio_pika")
from fastapi.testclient import TestClient  # noqa: E402

from api.app.config import Settings  # noqa: E402
from api.app.events import EXPLANATION_QUEUE  # noqa: E402
from api.app.main import create_app  # noqa: E402


async def _purge():
    conn = await aio_pika.connect(URL)
    async with conn:
        ch = await conn.channel()
        try:
            q = await ch.declare_queue(EXPLANATION_QUEUE, passive=True)
            await q.purge()
        except aio_pika.exceptions.ChannelClosed:
            pass   # la cola aún no existe: la crea la API al arrancar


async def _drain(n, timeout=5.0):
    conn = await aio_pika.connect(URL)
    async with conn:
        ch = await conn.channel()
        q = await ch.declare_queue(EXPLANATION_QUEUE, passive=True)
        out, deadline = [], asyncio.get_running_loop().time() + timeout
        while len(out) < n and asyncio.get_running_loop().time() < deadline:
            msg = await q.get(fail=False)
            if msg is None:
                await asyncio.sleep(0.05)
                continue
            async with msg.process():
                out.append((msg, json.loads(msg.body)))
        return out


def test_eventos_llegan_a_la_cola_del_worker(model_dir):
    asyncio.run(_purge())
    samples = json.loads((model_dir / "sample_payloads.json").read_text())
    settings = Settings(model_dir=model_dir, scorer_backend="local", api_keys=frozenset(),
                        rate_limit_per_minute=100, rabbitmq_url=URL, rabbitmq_publish_timeout_ms=1000)
    with TestClient(create_app(settings)) as c:
        assert c.get("/health").json()["rabbitmq"] == "connected"
        r1 = c.post("/v1/transactions/score", json=samples["fraud"][0] | {"transaction_id": "it-fraud",
                                                                          "card_number": "4111111111111234"})
        r2 = c.post("/v1/transactions/score", json=samples["legit"][0] | {"transaction_id": "it-legit"})
    assert r1.json()["explanation_status"] == r2.json()["explanation_status"] == "queued"

    msgs = asyncio.run(_drain(2))
    by_id = {body["transaction_id"]: (msg, body) for msg, body in msgs}
    assert set(by_id) == {"it-fraud", "it-legit"}
    msg, body = by_id["it-fraud"]
    assert msg.message_id == "it-fraud" and msg.delivery_mode == aio_pika.DeliveryMode.PERSISTENT
    assert msg.headers["decision"] == "REJECT" and body["card_masked"].endswith("1234")
    assert "4111111111111234" not in msg.body.decode()
