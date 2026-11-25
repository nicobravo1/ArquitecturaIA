"""Backends de inferencia.

* ``SageMakerBackend``: invoca el endpoint desplegado (producción).
* ``LocalBackend``: carga ``models/xgboost-model`` en proceso (desarrollo,
  tests y *fallback* si SageMaker no responde).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from fraud_core import FEATURE_COLUMNS
from fraud_core.scoring import SageMakerScorer, Thresholds

log = logging.getLogger("api.scorers")


class Backend(Protocol):
    name: str

    def predict_proba(self, features: pd.DataFrame | np.ndarray) -> list[float]: ...


class LocalBackend:
    name = "local"

    def __init__(self, model_path: Path):
        import xgboost as xgb

        self._xgb = xgb
        self.booster = xgb.Booster()
        self.booster.load_model(model_path)
        self.booster.set_param({"nthread": 1})   # menor latencia por request individual

    def predict_proba(self, features: pd.DataFrame | np.ndarray) -> list[float]:
        # inplace_predict sobre NumPy evita construir un DMatrix (~2,6 ms → ~0,1 ms)
        X = features[FEATURE_COLUMNS].to_numpy() if isinstance(features, pd.DataFrame) else features
        return [float(p) for p in self.booster.inplace_predict(X)]


class SageMakerBackend:
    name = "sagemaker"

    def __init__(self, endpoint: str, thresholds: Thresholds, region: str | None = None, runtime_client=None):
        self.scorer = SageMakerScorer(endpoint, thresholds, runtime_client=runtime_client, region_name=region)

    def predict_proba(self, features: pd.DataFrame | np.ndarray) -> list[float]:
        return self.scorer.predict_proba(features)


class ScoringService:
    """Orquesta backend principal + fallback local."""

    def __init__(self, primary: Backend, fallback: Backend | None = None):
        self.primary, self.fallback = primary, fallback

    @property
    def name(self) -> str:
        return self.primary.name

    def predict_proba(self, features: pd.DataFrame | np.ndarray) -> tuple[list[float], str]:
        try:
            return self.primary.predict_proba(features), self.primary.name
        except Exception:
            if self.fallback is None:
                raise
            log.exception("Backend %s falló; usando fallback local", self.primary.name)
            return self.fallback.predict_proba(features), "local-fallback"


def build_scoring_service(settings, thresholds: Thresholds) -> ScoringService:
    model_path = settings.model_dir / "xgboost-model"
    if settings.scorer_backend == "local":
        return ScoringService(LocalBackend(model_path))
    if settings.scorer_backend != "sagemaker":
        raise ValueError(f"SCORER_BACKEND desconocido: {settings.scorer_backend}")

    dep = settings.deployment()
    endpoint = settings.sagemaker_endpoint or dep.get("endpoint_name")
    if not endpoint:
        raise RuntimeError("Falta SAGEMAKER_ENDPOINT (o models/deployment.json)")
    region = settings.aws_region or dep.get("region")
    fallback = LocalBackend(model_path) if settings.local_fallback and model_path.exists() else None
    return ScoringService(SageMakerBackend(endpoint, thresholds, region), fallback)
