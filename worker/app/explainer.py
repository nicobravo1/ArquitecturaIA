"""Cálculo de valores SHAP y redacción del reporte de auditoría."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_core import FEATURE_COLUMNS, build_features

FEATURE_LABELS = {"Amount": "Monto", "hour": "Hora del día"}


class ShapExplainer:
    """Valores SHAP exactos (TreeSHAP) sobre el mismo booster desplegado en SageMaker.

    Se usa la implementación nativa de XGBoost (``pred_contribs=True``, algoritmo
    TreeSHAP de Lundberg et al.) en vez de ``shap.TreeExplainer``: con
    shap 0.54 + xgboost 3.0 este último calcula mal ``expected_value`` y la
    explicación no suma la predicción del modelo, inaceptable en una auditoría.
    Los valores están en log-odds: ``base_value + Σ shap = logit(prob)``.
    """

    def __init__(self, model_dir: Path):
        import xgboost as xgb

        meta = json.loads((model_dir / "metadata.json").read_text())
        self.model_version: str = meta["model_version"]
        if meta["feature_columns"] != FEATURE_COLUMNS:
            raise ValueError("El modelo no usa fraud_core.FEATURE_COLUMNS")
        self.booster = xgb.Booster()
        self.booster.load_model(model_dir / "xgboost-model")
        self.booster.set_param({"nthread": 1})
        self._xgb = xgb

    def probability(self, features: pd.DataFrame) -> float:
        return float(self.booster.predict(self._xgb.DMatrix(features, feature_names=FEATURE_COLUMNS))[0])

    def explain(self, raw_features: dict[str, float], top_k: int = 5) -> dict[str, Any]:
        t0 = time.perf_counter()
        x = build_features(raw_features)
        contribs = self.booster.predict(self._xgb.DMatrix(x, feature_names=FEATURE_COLUMNS),
                                        pred_contribs=True)[0]
        values, base_value = contribs[:-1].astype(float), float(contribs[-1])
        shap_map = {f: float(v) for f, v in zip(FEATURE_COLUMNS, values)}
        order = np.argsort(-np.abs(values))[:top_k]
        reasons = [{
            "feature": FEATURE_COLUMNS[i],
            "label": FEATURE_LABELS.get(FEATURE_COLUMNS[i], FEATURE_COLUMNS[i]),
            "value": float(x.iloc[0, i]),
            "shap": float(values[i]),
            "direction": "aumenta riesgo" if values[i] > 0 else "reduce riesgo",
        } for i in order]
        logit = base_value + float(values.sum())
        return {
            "shap_base_value": base_value,
            "shap_values": shap_map,
            "top_reasons": reasons,
            "shap_probability": float(1 / (1 + np.exp(-logit))),
            "explain_ms": (time.perf_counter() - t0) * 1000,
        }


def explanation_text(decision: str, probability: float, threshold: float, reasons: list[dict]) -> str:
    """Resumen legible para el analista / auditor (CMF)."""
    verbo = {"REJECT": "Rechazada", "REVIEW": "Enviada a revisión", "APPROVE": "Aprobada"}[decision]
    up = [r for r in reasons if r["shap"] > 0]
    down = [r for r in reasons if r["shap"] <= 0]
    fmt = lambda rs: "; ".join(f"{r['label']}={r['value']:.2f} ({r['shap']:+.2f})" for r in rs)  # noqa: E731
    parts = [f"{verbo}: probabilidad de fraude {probability:.1%} (umbral de rechazo {threshold:.1%})."]
    if up:
        parts.append(f"Factores que aumentan el riesgo: {fmt(up)}.")
    if down:
        parts.append(f"Factores que lo reducen: {fmt(down)}.")
    return " ".join(parts)
