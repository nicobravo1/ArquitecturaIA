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


_PCA_AMOUNT = PCA_COLUMNS + ["Amount"]


def features_matrix(records: Iterable[Mapping]) -> np.ndarray:
    """Igual que :func:`build_features` pero devuelve un ``ndarray`` (n, 30) sin pasar
    por pandas. Es la ruta caliente de la API: ~0,02 ms por transacción frente a
    ~2 ms de construir un DataFrame."""
    records = list(records)
    out = np.empty((len(records), len(FEATURE_COLUMNS)), dtype=np.float64)
    for i, rec in enumerate(records):
        try:
            out[i, :29] = [rec[c] for c in _PCA_AMOUNT]
            t = float(rec["Time"])
        except KeyError as err:
            raise ValueError(f"Faltan columnas requeridas: [{err.args[0]!r}]") from None
        out[i, 29] = (t % SECONDS_PER_DAY) // 3600
    if (out[:, 28] < 0).any():
        raise ValueError("Amount no puede ser negativo")
    return out


def to_sagemaker_csv(features: pd.DataFrame | np.ndarray) -> str:
    """Serializa variables al formato ``text/csv`` (sin cabecera) del contenedor
    XGBoost de SageMaker."""
    arr = features[FEATURE_COLUMNS].to_numpy(dtype=np.float64) if isinstance(features, pd.DataFrame) \
        else np.asarray(features, dtype=np.float64)
    return "\n".join(",".join(repr(float(v)) for v in row) for row in arr)
