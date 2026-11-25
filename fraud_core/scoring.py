"""Cliente de inferencia contra el endpoint de SageMaker y reglas de decisión.

Se comparte entre ``sagemaker_deploy/invoke_test.py`` y la API FastAPI, de modo
que la serialización (``text/csv``), el parseo de la respuesta del contenedor
XGBoost y la aplicación de umbrales sean idénticos en todos los componentes.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .features import build_features, to_sagemaker_csv

APPROVE, REVIEW, REJECT = "APPROVE", "REVIEW", "REJECT"


@dataclass(frozen=True)
class Thresholds:
    reject: float
    review: float

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "Thresholds":
        reject = float(metadata["threshold"])
        review = float(metadata.get("review_threshold", reject))
        if not 0 <= review <= reject <= 1:
            raise ValueError(f"Umbrales inválidos: review={review}, reject={reject}")
        return cls(reject=reject, review=review)


def decide(probability: float, thresholds: Thresholds) -> str:
    """APPROVE / REVIEW (step-up auth o revisión manual) / REJECT."""
    if probability >= thresholds.reject:
        return REJECT
    if probability >= thresholds.review:
        return REVIEW
    return APPROVE


_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_predictions(body: bytes | str, content_type: str = "") -> list[float]:
    """Parsea la respuesta del contenedor XGBoost de SageMaker.

    Soporta ``application/json`` (``{"predictions": [{"score": p}, ...]}``) y
    ``text/csv`` (valores separados por comas o saltos de línea).
    """
    text = body.decode("utf-8") if isinstance(body, bytes) else body
    text = text.strip()
    if "json" in content_type or text.startswith("{") or text.startswith("["):
        data = json.loads(text)
        preds = data["predictions"] if isinstance(data, dict) else data
        out = [float(p["score"]) if isinstance(p, dict) else float(p) for p in preds]
    else:
        out = [float(x) for x in _NUMBER.findall(text)]
    for p in out:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Probabilidad fuera de rango: {p}")
    return out


def load_metadata(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


class SageMakerScorer:
    """Invoca el endpoint y devuelve probabilidad, decisión y latencia."""

    def __init__(self, endpoint_name: str, thresholds: Thresholds,
                 runtime_client: Any = None, region_name: str | None = None):
        if runtime_client is None:
            import boto3
            from botocore.config import Config

            runtime_client = boto3.client(
                "sagemaker-runtime", region_name=region_name,
                # Fallar rápido: el flujo de pago tiene presupuesto de 100 ms
                config=Config(connect_timeout=1, read_timeout=2,
                              retries={"max_attempts": 2, "mode": "standard"}),
            )
        self.endpoint_name = endpoint_name
        self.thresholds = thresholds
        self.runtime = runtime_client

    def predict_proba(self, features: pd.DataFrame) -> list[float]:
        resp = self.runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="text/csv",
            Accept="application/json",
            Body=to_sagemaker_csv(features).encode("utf-8"),
        )
        preds = parse_predictions(resp["Body"].read(), resp.get("ContentType", ""))
        if len(preds) != len(features):
            raise ValueError(f"El endpoint devolvió {len(preds)} predicciones para {len(features)} filas")
        return preds

    def score(self, transactions) -> list[dict[str, Any]]:
        """Recibe transacciones crudas (dict / lista / DataFrame) y devuelve decisiones."""
        features = build_features(transactions)
        t0 = time.perf_counter()
        probs = self.predict_proba(features)
        latency_ms = (time.perf_counter() - t0) * 1000
        return [
            {"fraud_probability": p, "decision": decide(p, self.thresholds),
             "model_latency_ms": latency_ms}
            for p in probs
        ]
