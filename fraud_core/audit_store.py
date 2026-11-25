"""Almacén de auditoría (lado *query* del patrón CQRS).

El worker SHAP escribe un registro inmutable por transacción; la API y el
dashboard solo leen. Implementado con SQLAlchemy Core para funcionar con
PostgreSQL (producción / Amazon RDS) y SQLite (desarrollo y tests) sin cambios.

``DATABASE_URL`` de ejemplo:
  postgresql+psycopg://fraud:***@localhost:5432/fraud_audit
  sqlite:///audit.db
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON, Column, DateTime, Float, Index, MetaData, String, Table, create_engine, func, select, text,
)
from sqlalchemy.dialects import postgresql, sqlite

# none_as_null: None se guarda como NULL de SQL (no como el JSON `null`)
JSONType = JSON(none_as_null=True).with_variant(postgresql.JSONB(none_as_null=True), "postgresql")
metadata = MetaData()

audit = Table(
    "fraud_audit", metadata,
    Column("transaction_id", String(64), primary_key=True),
    Column("scored_at", DateTime(timezone=True), nullable=False),
    Column("explained_at", DateTime(timezone=True), nullable=False),
    Column("decision", String(10), nullable=False),
    Column("fraud_probability", Float, nullable=False),
    Column("threshold", Float, nullable=False),
    Column("review_threshold", Float, nullable=False),
    Column("model_version", String(64), nullable=False),
    Column("scorer", String(32)),
    Column("client", String(64)),
    Column("card_masked", String(32)),
    Column("amount", Float),
    Column("features", JSONType, nullable=False),
    Column("shap_base_value", Float),
    Column("shap_values", JSONType),
    Column("top_reasons", JSONType),
    Column("explanation_text", String(2000)),
    Column("explain_ms", Float),
)
Index("ix_fraud_audit_scored_at", audit.c.scored_at)
Index("ix_fraud_audit_decision_scored_at", audit.c.decision, audit.c.scored_at)

SUMMARY_COLUMNS = [c for c in audit.c if c.name not in ("features", "shap_values")]


def _parse_dt(v: Any) -> datetime:
    if isinstance(v, datetime):
        dt = v
    else:
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _row(r) -> dict[str, Any]:
    d = dict(r._mapping)
    for k in ("scored_at", "explained_at"):
        if d.get(k) is not None:
            d[k] = _parse_dt(d[k])          # SQLite devuelve datetimes sin zona
    return d


class AuditStore:
    def __init__(self, url: str, **engine_kwargs):
        if url.startswith("postgresql"):
            engine_kwargs.setdefault("pool_size", 5)
            engine_kwargs.setdefault("pool_pre_ping", True)
        self.engine = create_engine(url, **engine_kwargs)
        self._insert = postgresql.insert if self.engine.dialect.name == "postgresql" else sqlite.insert

    def create_schema(self) -> None:
        metadata.create_all(self.engine)

    def ping(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def save(self, record: dict[str, Any]) -> bool:
        """Inserta el registro. Idempotente: si el ``transaction_id`` ya existe
        (mensaje reentregado por RabbitMQ) no hace nada y devuelve False."""
        rec = dict(record)
        for k in ("scored_at", "explained_at"):
            rec[k] = _parse_dt(rec[k])
        # RETURNING en vez de rowcount: psycopg 3 devuelve rowcount=-1 con ON CONFLICT
        stmt = (self._insert(audit).values(**rec)
                .on_conflict_do_nothing(index_elements=["transaction_id"])
                .returning(audit.c.transaction_id))
        with self.engine.begin() as conn:
            return conn.execute(stmt).first() is not None

    def get(self, transaction_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as conn:
            r = conn.execute(select(audit).where(audit.c.transaction_id == transaction_id)).first()
        return _row(r) if r else None

    def list_recent(self, limit: int = 50, decision: str | None = None,
                    since: datetime | None = None) -> list[dict[str, Any]]:
        q = select(*SUMMARY_COLUMNS).order_by(audit.c.scored_at.desc()).limit(limit)
        if decision:
            q = q.where(audit.c.decision == decision)
        if since:
            q = q.where(audit.c.scored_at >= since)
        with self.engine.connect() as conn:
            return [_row(r) for r in conn.execute(q)]

    def stats(self, since: datetime | None = None) -> dict[str, Any]:
        q = select(audit.c.decision, func.count(), func.avg(audit.c.fraud_probability),
                   func.sum(audit.c.amount), func.avg(audit.c.explain_ms)).group_by(audit.c.decision)
        if since:
            q = q.where(audit.c.scored_at >= since)
        by_decision, total, amount, explain = {}, 0, {}, []
        with self.engine.connect() as conn:
            for decision, n, avg_p, amt, avg_ms in conn.execute(q):
                by_decision[decision] = {"count": n, "avg_probability": float(avg_p or 0),
                                         "amount": float(amt or 0)}
                total += n
                if avg_ms is not None:
                    explain.append((avg_ms, n))
        return {
            "total": total,
            "by_decision": by_decision,
            "reject_rate": by_decision.get("REJECT", {}).get("count", 0) / total if total else 0.0,
            "avg_explain_ms": (sum(m * n for m, n in explain) / sum(n for _, n in explain)) if explain else None,
        }
