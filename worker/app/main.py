"""Consumidor RabbitMQ → SHAP → PostgreSQL.

    python -m worker.app.main

Garantías:
* **At-least-once**: ack manual solo después de guardar en la base de datos.
* **Idempotencia**: ``transaction_id`` es clave primaria (re-entregas no duplican).
* **Errores permanentes** (JSON inválido, PAN sin enmascarar, versión de modelo
  distinta) → ``reject`` → DLQ ``shap.explanations.dlq`` para revisión.
* **Errores transitorios** (BD caída) → reintento con backoff exponencial
  (header ``x-retries``) hasta ``WORKER_MAX_RETRIES``; luego DLQ.
"""
from __future__ import annotations

import asyncio
import json
import logging
import signal

from prometheus_client import Counter, Histogram, start_http_server

from fraud_core.audit_store import AuditStore
from fraud_core.messaging import EXPLANATION_QUEUE, declare_topology

from .config import WorkerSettings
from .explainer import ShapExplainer
from .processor import InvalidEvent, build_record

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("worker")

PROCESSED = Counter("fraud_worker_messages_total", "Mensajes procesados", ["result"])
SHAP_LATENCY = Histogram("fraud_worker_shap_seconds", "Tiempo de cálculo SHAP",
                         buckets=(.001, .0025, .005, .01, .025, .05, .1, .25))
LAG = Histogram("fraud_worker_lag_seconds", "Tiempo desde el scoring hasta el registro de auditoría",
                buckets=(.05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60))


class Worker:
    def __init__(self, settings: WorkerSettings, store: AuditStore | None = None,
                 explainer: ShapExplainer | None = None):
        self.settings = settings
        self.store = store or AuditStore(settings.database_url)
        self.explainer = explainer or ShapExplainer(settings.model_dir)
        self._stop = asyncio.Event()

    def handle_body(self, body: bytes) -> tuple[str, dict | None]:
        """Lógica síncrona (se ejecuta en un hilo). Devuelve (resultado, registro)."""
        try:
            event = json.loads(body)
        except json.JSONDecodeError as err:
            raise InvalidEvent(f"JSON inválido: {err}") from None
        record = build_record(event, self.explainer, self.settings.explain_decisions, self.settings.top_k)
        if record["explain_ms"] is not None:
            SHAP_LATENCY.observe(record["explain_ms"] / 1000)
        inserted = self.store.save(record)
        return ("stored" if inserted else "duplicate"), record

    async def on_message(self, message, channel=None) -> str:
        import aio_pika

        try:
            result, record = await asyncio.to_thread(self.handle_body, message.body)
            await message.ack()
            if record is not None:
                LAG.observe(max(0.0, (record["explained_at"] - _dt(record["scored_at"])).total_seconds()))
            log.info("%s %s %s", result, message.message_id, record["decision"] if record else "")
        except InvalidEvent as err:
            log.error("Mensaje %s inválido → DLQ: %s", message.message_id, err)
            await message.reject(requeue=False)
            result = "invalid"
        except Exception as err:   # transitorio: BD no disponible, etc.
            retries = int((message.headers or {}).get("x-retries", 0))
            if retries >= self.settings.max_retries or channel is None:
                log.exception("Mensaje %s falló %d veces → DLQ", message.message_id, retries)
                await message.reject(requeue=False)
                result = "failed"
            else:
                delay = 0.5 * 2 ** retries
                log.warning("Error transitorio en %s (%s); reintento %d en %.1fs",
                            message.message_id, err, retries + 1, delay)
                await asyncio.sleep(delay)
                retry = aio_pika.Message(
                    body=message.body, content_type=message.content_type, message_id=message.message_id,
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    headers={**(message.headers or {}), "x-retries": retries + 1})
                await channel.default_exchange.publish(retry, routing_key=EXPLANATION_QUEUE)
                await message.ack()
                result = "retried"
        PROCESSED.labels(result).inc()
        return result

    async def run(self) -> None:
        import aio_pika

        self.store.create_schema()
        conn = await aio_pika.connect_robust(self.settings.rabbitmq_url)
        async with conn:
            channel = await conn.channel()
            await channel.set_qos(prefetch_count=self.settings.prefetch)
            _, queue = await declare_topology(channel, self.settings.rabbitmq_exchange)
            async def handler(message):
                await self.on_message(message, channel)

            await queue.consume(handler)
            log.info("Worker escuchando %s | modelo %s | explica %s",
                     EXPLANATION_QUEUE, self.explainer.model_version, sorted(self.settings.explain_decisions))
            await self._stop.wait()
        log.info("Worker detenido")

    def stop(self) -> None:
        self._stop.set()


def _dt(v):
    from datetime import datetime, timezone
    d = v if isinstance(v, datetime) else datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


async def _main() -> None:
    settings = WorkerSettings()
    start_http_server(settings.metrics_port)
    worker = Worker(settings)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, worker.stop)   # apagado ordenado (Kubernetes/ECS)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(_main())
