"""Publicación de eventos de scoring en RabbitMQ (comunicación asíncrona).

Topología::

    exchange  fraud.events (topic, durable)
      ├─ routing key transaction.scored  → cola shap.explanations (worker SHAP)
      └─ (otras colas pueden suscribirse: dashboard, data lake, alertas)

Mensajes persistentes, JSON, con ``message_id`` = transaction_id (idempotencia
en el consumidor) y *publisher confirms*. Si RabbitMQ no está disponible, la
decisión de pago **no se bloquea**: se informa ``explanation_status=unavailable``.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Protocol

log = logging.getLogger("api.events")

from fraud_core.messaging import EXPLANATION_QUEUE, ROUTING_KEY, declare_topology  # noqa: F401


class Publisher(Protocol):
    connected: bool

    async def start(self) -> None: ...
    async def publish(self, event: dict[str, Any]) -> bool: ...
    async def close(self) -> None: ...


class NullPublisher:
    """Sin broker configurado (desarrollo)."""
    connected = False

    async def start(self) -> None:
        log.warning("RABBITMQ_URL no configurado: los eventos no se publican")

    async def publish(self, event: dict[str, Any]) -> bool:
        return False

    async def close(self) -> None:
        pass


class RabbitPublisher:
    def __init__(self, url: str, exchange: str = "fraud.events", publish_timeout: float = 0.05):
        # publish_timeout acotado: el flujo de pago tiene presupuesto total de 100 ms
        self.url, self.exchange_name, self.timeout = url, exchange, publish_timeout
        self._conn = self._channel = self._exchange = None
        self._reconnect_task: asyncio.Task | None = None

    @property
    def connected(self) -> bool:
        return self._conn is not None and not self._conn.is_closed

    async def start(self) -> None:
        import aio_pika

        try:
            # connect_robust reconecta automáticamente si el broker se reinicia
            self._conn = await aio_pika.connect_robust(self.url, timeout=5)
            self._channel = await self._conn.channel(publisher_confirms=True)
            self._exchange, _ = await declare_topology(self._channel, self.exchange_name)
            log.info("Conectado a RabbitMQ, exchange=%s", self.exchange_name)
        except Exception:
            log.exception("No se pudo conectar a RabbitMQ; se reintentará al publicar")
            self._conn = None

    async def publish(self, event: dict[str, Any]) -> bool:
        import aio_pika

        if not self.connected:
            # Nunca reconectar dentro del request: se agenda en segundo plano
            if self._reconnect_task is None or self._reconnect_task.done():
                self._reconnect_task = asyncio.create_task(self.start())
            return False
        msg = aio_pika.Message(
            body=json.dumps(event, default=str).encode(),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            message_id=event["transaction_id"],
            type=ROUTING_KEY,
            headers={"model_version": event["model_version"], "decision": event["decision"]},
        )
        try:
            await asyncio.wait_for(self._exchange.publish(msg, routing_key=ROUTING_KEY), self.timeout)
            return True
        except Exception:
            log.exception("Error publicando evento %s", event["transaction_id"])
            return False

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()


def build_publisher(settings) -> Publisher:
    if not settings.rabbitmq_url:
        return NullPublisher()
    return RabbitPublisher(settings.rabbitmq_url, settings.rabbitmq_exchange,
                           settings.rabbitmq_publish_timeout_ms / 1000)
