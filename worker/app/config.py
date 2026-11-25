from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WorkerSettings:
    rabbitmq_url: str = field(default_factory=lambda: os.environ.get(
        "RABBITMQ_URL", "amqp://guest:guest@localhost:5672/"))
    rabbitmq_exchange: str = field(default_factory=lambda: os.environ.get("RABBITMQ_EXCHANGE", "fraud.events"))
    database_url: str = field(default_factory=lambda: os.environ.get(
        "DATABASE_URL", f"sqlite:///{ROOT / 'audit.db'}"))
    model_dir: Path = field(default_factory=lambda: Path(os.environ.get("MODEL_DIR", ROOT / "models")))
    # Decisiones con cálculo SHAP completo. APPROVE se guarda sin SHAP por defecto:
    # con 1M tx/día, explicar solo rechazos/revisiones reduce >99 % el costo de CPU.
    explain_decisions: frozenset[str] = field(default_factory=lambda: frozenset(
        d.strip().upper() for d in os.environ.get("EXPLAIN_DECISIONS", "REJECT,REVIEW").split(",") if d.strip()))
    top_k: int = field(default_factory=lambda: int(os.environ.get("SHAP_TOP_K", "5")))
    prefetch: int = field(default_factory=lambda: int(os.environ.get("WORKER_PREFETCH", "32")))
    max_retries: int = field(default_factory=lambda: int(os.environ.get("WORKER_MAX_RETRIES", "3")))
    metrics_port: int = field(default_factory=lambda: int(os.environ.get("WORKER_METRICS_PORT", "9100")))
