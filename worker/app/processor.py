"""Procesa un evento ``transaction.scored``: valida, explica y guarda (sin I/O de red)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fraud_core import RAW_COLUMNS

REQUIRED = ("transaction_id", "scored_at", "decision", "fraud_probability", "threshold", "model_version", "features")
DECISIONS = {"APPROVE", "REVIEW", "REJECT"}


class InvalidEvent(ValueError):
    """Mensaje que nunca podrá procesarse (va directo a la DLQ)."""


def validate(event: Any) -> dict[str, Any]:
    if not isinstance(event, dict):
        raise InvalidEvent("El evento no es un objeto JSON")
    missing = [k for k in REQUIRED if k not in event]
    if missing:
        raise InvalidEvent(f"Faltan campos: {missing}")
    if event["decision"] not in DECISIONS:
        raise InvalidEvent(f"Decisión inválida: {event['decision']}")
    feats = event["features"]
    if not isinstance(feats, dict) or any(c not in feats for c in RAW_COLUMNS):
        raise InvalidEvent("features incompletas")
    if "card_number" in feats or "card_number" in event:
        raise InvalidEvent("El evento contiene un PAN sin enmascarar (violación PCI-DSS)")
    return event


def build_record(event: dict[str, Any], explainer, explain_decisions: frozenset[str],
                 top_k: int = 5) -> dict[str, Any]:
    from .explainer import explanation_text

    event = validate(event)
    if event["model_version"] != explainer.model_version:
        raise InvalidEvent(f"Versión de modelo {event['model_version']} distinta a la del worker "
                           f"({explainer.model_version}); la explicación no sería auditable")
    record = {
        "transaction_id": event["transaction_id"],
        "scored_at": event["scored_at"],
        "explained_at": datetime.now(timezone.utc),
        "decision": event["decision"],
        "fraud_probability": float(event["fraud_probability"]),
        "threshold": float(event["threshold"]),
        "review_threshold": float(event.get("review_threshold", event["threshold"])),
        "model_version": event["model_version"],
        "scorer": event.get("scorer"),
        "client": event.get("client"),
        "card_masked": event.get("card_masked"),
        "amount": float(event["features"]["Amount"]),
        "features": event["features"],
        "shap_base_value": None, "shap_values": None, "top_reasons": None,
        "explanation_text": None, "explain_ms": None,
    }
    if event["decision"] in explain_decisions:
        exp = explainer.explain(event["features"], top_k=top_k)
        record.update(
            shap_base_value=exp["shap_base_value"], shap_values=exp["shap_values"],
            top_reasons=exp["top_reasons"], explain_ms=round(exp["explain_ms"], 3),
            explanation_text=explanation_text(event["decision"], record["fraud_probability"],
                                              record["threshold"], exp["top_reasons"],
                                              rule=event.get("rule"), amount=record["amount"]))
    return record
