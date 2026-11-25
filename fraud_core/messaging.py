"""Topología de RabbitMQ compartida por la API (productor) y el worker (consumidor).

    exchange  fraud.events      (topic, durable)
      └─ transaction.scored ──▶ cola shap.explanations  (durable, DLX)
    exchange  fraud.events.dlx  (fanout, durable)
      └─────────────────────▶ cola shap.explanations.dlq (mensajes inválidos / fallidos)
"""
from __future__ import annotations

DEFAULT_EXCHANGE = "fraud.events"
ROUTING_KEY = "transaction.scored"
EXPLANATION_QUEUE = "shap.explanations"


async def declare_topology(channel, exchange_name: str = DEFAULT_EXCHANGE):
    """Declara (idempotente) exchanges y colas. Devuelve (exchange, cola)."""
    import aio_pika

    exchange = await channel.declare_exchange(exchange_name, aio_pika.ExchangeType.TOPIC, durable=True)
    dlx = await channel.declare_exchange(f"{exchange_name}.dlx", aio_pika.ExchangeType.FANOUT, durable=True)
    dlq = await channel.declare_queue(f"{EXPLANATION_QUEUE}.dlq", durable=True)
    await dlq.bind(dlx)
    queue = await channel.declare_queue(EXPLANATION_QUEUE, durable=True,
                                        arguments={"x-dead-letter-exchange": f"{exchange_name}.dlx"})
    await queue.bind(exchange, routing_key=ROUTING_KEY)
    return exchange, queue
