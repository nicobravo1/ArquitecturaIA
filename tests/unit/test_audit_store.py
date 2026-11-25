from datetime import datetime, timedelta, timezone

from fraud_core import RAW_COLUMNS


def rec(tx, decision="REJECT", p=0.9, minutes_ago=0, amount=100.0):
    ts = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return {"transaction_id": tx, "scored_at": ts.isoformat(), "explained_at": ts, "decision": decision,
            "fraud_probability": p, "threshold": 0.6, "review_threshold": 0.3, "model_version": "v1",
            "amount": amount, "features": {c: 0.0 for c in RAW_COLUMNS},
            "top_reasons": [{"feature": "V14", "shap": 1.2}], "explain_ms": 2.0}


def test_save_es_idempotente(audit_store):
    assert audit_store.save(rec("a")) is True
    assert audit_store.save(rec("a", p=0.1)) is False          # re-entrega: no sobrescribe
    got = audit_store.get("a")
    assert got["fraud_probability"] == 0.9 and got["scored_at"].tzinfo is not None
    assert audit_store.get("no-existe") is None


def test_list_recent_filtra_y_ordena(audit_store):
    for i, d in enumerate(["REJECT", "APPROVE", "REJECT"]):
        audit_store.save(rec(f"t{i}", decision=d, minutes_ago=i * 90))
    assert [r["transaction_id"] for r in audit_store.list_recent()] == ["t0", "t1", "t2"]
    assert [r["transaction_id"] for r in audit_store.list_recent(decision="REJECT")] == ["t0", "t2"]
    since = datetime.now(timezone.utc) - timedelta(hours=2)
    assert len(audit_store.list_recent(since=since)) == 2
    assert "shap_values" not in audit_store.list_recent()[0]   # listado liviano


def test_stats(audit_store):
    audit_store.save(rec("r1", "REJECT", 0.9, amount=50))
    audit_store.save(rec("a1", "APPROVE", 0.1, amount=10))
    audit_store.save(rec("a2", "APPROVE", 0.3, amount=20))
    s = audit_store.stats()
    assert s["total"] == 3 and abs(s["reject_rate"] - 1 / 3) < 1e-9
    assert s["by_decision"]["APPROVE"]["count"] == 2 and s["by_decision"]["APPROVE"]["amount"] == 30
    assert abs(s["by_decision"]["APPROVE"]["avg_probability"] - 0.2) < 1e-9
    assert audit_store.ping()


def test_none_se_guarda_como_null_sql(audit_store):
    from sqlalchemy import text
    r = rec("sin-shap")
    r.update(top_reasons=None, shap_values=None)
    assert audit_store.save(r) is True
    with audit_store.engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM fraud_audit WHERE shap_values IS NULL "
                              "AND top_reasons IS NULL")).scalar()
    assert n == 1 and audit_store.get("sin-shap")["shap_values"] is None
