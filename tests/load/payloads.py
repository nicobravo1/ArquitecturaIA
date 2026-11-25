"""Generación de transacciones y etapas de carga (sin dependencias de Locust,
para poder probarlas con pytest sin el monkey-patching de gevent)."""
from __future__ import annotations

import json
import os
import random
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCALE = float(os.environ.get("LOAD_SCALE", "1.0"))

PCA = [f"V{i}" for i in range(1, 29)]
_FALLBACK = {  # fila legítima real del dataset si no existe models/sample_payloads.json
    "legit": [{"Time": 0.0, "Amount": 149.62, **dict(zip(PCA, [
        -1.359807, -0.072781, 2.536347, 1.378155, -0.338321, 0.462388, 0.239599, 0.098698, 0.363787,
        0.090794, -0.551600, -0.617801, -0.991390, -0.311169, 1.468177, -0.470401, 0.207971, 0.025791,
        0.403993, 0.251412, -0.018307, 0.277838, -0.110474, 0.066928, 0.128539, -0.189115, 0.133558,
        -0.021053]))}],
    "fraud": [],
}


def _load_samples() -> tuple[list[dict], list[dict]]:
    path = Path(os.environ.get("SAMPLES_PATH", ROOT / "models" / "sample_payloads.json"))
    data = json.loads(path.read_text()) if path.exists() else _FALLBACK
    return data["legit"], data["fraud"] or data["legit"]


LEGIT, FRAUD = _load_samples()
FRAUD_RATIO = float(os.environ.get("FRAUD_RATIO", "0.02"))   # algo más que el 0,17 % real, para ejercitar SHAP


def make_transaction() -> dict:
    """Transacción realista con ruido para no repetir exactamente el mismo payload."""
    base = random.choice(FRAUD if random.random() < FRAUD_RATIO else LEGIT)
    tx = {k: (v + random.gauss(0, 0.05) if k in PCA else v) for k, v in base.items()}
    tx["Amount"] = round(max(0.0, base["Amount"] * random.uniform(0.5, 1.5)), 2)
    tx["Time"] = float(random.randint(0, 172_799))
    tx["transaction_id"] = f"load-{uuid.uuid4().hex[:20]}"
    if random.random() < 0.3:
        tx["card_number"] = "4" + "".join(random.choices("0123456789", k=15))
    return tx


# (duración en s, TPS objetivo) — cifras del caso a escala 1.0
STAGES: dict[str, list[tuple[int, float]]] = {
    "steady": [(30, 75), (150, 150)],                                   # promedio: 150 TPS
    "spike": [(30, 150), (30, 400), (60, 1000), (60, 150)],             # pico: 1.000 TPS
    "stress": [(30, tps) for tps in (100, 250, 500, 750, 1000, 1500, 2000)],
}


def users_for(tps: float, scale: float | None = None) -> int:
    """Usuarios PaymentGateway (~1 req/s cada uno) para un TPS objetivo."""
    return max(1, round(tps * (SCALE if scale is None else scale)))
