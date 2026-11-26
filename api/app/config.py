"""Configuración por variables de entorno (12-factor)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "si", "sí"}


def _keys() -> frozenset[str]:
    raw = os.environ.get("API_KEYS", "")
    return frozenset(k.strip() for k in raw.split(",") if k.strip())


@dataclass(frozen=True)
class Settings:
    model_dir: Path = field(default_factory=lambda: Path(os.environ.get("MODEL_DIR", ROOT / "models")))
    # sagemaker = endpoint en AWS (producción) | local = modelo xgboost en proceso (dev/tests)
    scorer_backend: str = field(default_factory=lambda: os.environ.get("SCORER_BACKEND", "sagemaker"))
    sagemaker_endpoint: str | None = field(default_factory=lambda: os.environ.get("SAGEMAKER_ENDPOINT"))
    aws_region: str | None = field(default_factory=lambda: os.environ.get("AWS_REGION"))
    # Si SageMaker falla, puntuar con el modelo local (alta disponibilidad)
    local_fallback: bool = field(default_factory=lambda: _bool("LOCAL_FALLBACK", True))
    api_keys: frozenset[str] = field(default_factory=_keys)
    rate_limit_per_minute: int = field(default_factory=lambda: int(os.environ.get("RATE_LIMIT_PER_MINUTE", "600")))
    rabbitmq_url: str | None = field(default_factory=lambda: os.environ.get("RABBITMQ_URL"))
    rabbitmq_exchange: str = field(default_factory=lambda: os.environ.get("RABBITMQ_EXCHANGE", "fraud.events"))
    rabbitmq_publish_timeout_ms: int = field(
        default_factory=lambda: int(os.environ.get("RABBITMQ_PUBLISH_TIMEOUT_MS", "50")))
    # Almacén de auditoría escrito por el worker SHAP (lado query de CQRS)
    database_url: str | None = field(default_factory=lambda: os.environ.get("DATABASE_URL"))
    # Hilos para llamadas bloqueantes (SageMaker vía boto3, BD de auditoría). El pool por
    # defecto de asyncio es min(32, CPUs + 4): solo 6 en una instancia de 2 vCPU.
    io_threads: int = field(default_factory=lambda: int(os.environ.get("API_IO_THREADS", "64")))
    sagemaker_max_connections: int = field(
        default_factory=lambda: int(os.environ.get("SAGEMAKER_MAX_CONNECTIONS", "64")))
    max_batch_size: int = field(default_factory=lambda: int(os.environ.get("MAX_BATCH_SIZE", "100")))
    cors_origins: tuple[str, ...] = field(default_factory=lambda: tuple(
        o for o in os.environ.get("CORS_ORIGINS", "").split(",") if o))

    def metadata(self) -> dict:
        return json.loads((self.model_dir / "metadata.json").read_text())

    def deployment(self) -> dict:
        path = self.model_dir / "deployment.json"
        return json.loads(path.read_text()) if path.exists() else {}
