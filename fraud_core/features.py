"""Ingeniería de variables para el dataset *Credit Card Fraud Detection* (mlg-ulb).

Columnas originales: ``Time``, ``V1``..``V28`` (componentes PCA anonimizadas),
``Amount`` y ``Class`` (1 = fraude).

Variables del modelo (en este orden exacto, que es el que espera el endpoint
de SageMaker en formato ``text/csv`` sin cabecera):

* ``V1``..``V28``
* ``Amount``  -> monto de la transacción
* ``hour``    -> hora del día derivada de ``Time`` (segundos desde la primera
  transacción del dataset). Se descarta ``Time`` en bruto porque es un
  contador monotónico que no generaliza a producción.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

PCA_COLUMNS: list[str] = [f"V{i}" for i in range(1, 29)]
RAW_COLUMNS: list[str] = ["Time", *PCA_COLUMNS, "Amount"]
TARGET_COLUMN = "Class"
FEATURE_COLUMNS: list[str] = [*PCA_COLUMNS, "Amount", "hour"]

SECONDS_PER_DAY = 86_400


def build_features(data: pd.DataFrame | Mapping | Iterable[Mapping]) -> pd.DataFrame:
    """Transforma transacciones crudas en la matriz de variables del modelo.

    Acepta un ``DataFrame``, un dict (una transacción) o una lista de dicts.
    Devuelve un ``DataFrame`` con ``FEATURE_COLUMNS`` en orden y tipo float.
    """
    if isinstance(data, Mapping):
        df = pd.DataFrame([data])
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.DataFrame(list(data))

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    out = df[PCA_COLUMNS + ["Amount"]].astype("float64").copy()
    if (out["Amount"] < 0).any():
        raise ValueError("Amount no puede ser negativo")
    out["hour"] = (df["Time"].astype("float64") % SECONDS_PER_DAY) // 3600
    return out[FEATURE_COLUMNS]


def to_sagemaker_csv(features: pd.DataFrame) -> str:
    """Serializa variables al formato ``text/csv`` (sin cabecera) del contenedor
    XGBoost de SageMaker."""
    return "\n".join(
        ",".join(repr(float(v)) for v in row)
        for row in features[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    )
