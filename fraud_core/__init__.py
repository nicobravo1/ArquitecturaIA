"""Código compartido entre entrenamiento, API de inferencia y worker SHAP.

Mantener la ingeniería de variables en un solo lugar evita el
*training/serving skew*: el notebook, FastAPI y el worker usan exactamente
las mismas funciones y el mismo orden de columnas.
"""
from .features import (  # noqa: F401
    RAW_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_features,
    to_sagemaker_csv,
)
