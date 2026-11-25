"""Contratos de entrada/salida (generan el esquema OpenAPI / Swagger)."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

Decision = Literal["APPROVE", "REVIEW", "REJECT"]

_PCA_FIELDS = {f"V{i}": (float, Field(..., description=f"Componente PCA {i} (anonimizada)"))
               for i in range(1, 29)}

_TransactionBase = create_model(  # V1..V28 generados dinámicamente
    "_TransactionBase", __config__=ConfigDict(extra="forbid"), **_PCA_FIELDS)


class Transaction(_TransactionBase):
    transaction_id: str | None = Field(
        None, max_length=64, pattern=r"^[A-Za-z0-9_\-:.]+$",
        description="ID de la transacción en el sistema de pagos (se genera si no viene)")
    Time: float = Field(..., ge=0, description="Segundos desde la primera transacción del período")
    Amount: float = Field(..., ge=0, le=1_000_000, description="Monto de la transacción")
    card_number: str | None = Field(
        None, description="PAN opcional. **Nunca se almacena ni se publica**: solo se conserva "
                          "enmascarado (PCI-DSS)", json_schema_extra={"writeOnly": True})

    @field_validator("card_number")
    @classmethod
    def _valid_pan(cls, v: str | None) -> str | None:
        if v is None:
            return v
        digits = re.sub(r"[\s-]", "", v)
        if not re.fullmatch(r"\d{12,19}", digits):
            raise ValueError("card_number debe tener entre 12 y 19 dígitos")
        return digits

    def masked_pan(self) -> str | None:
        return None if self.card_number is None else "*" * (len(self.card_number) - 4) + self.card_number[-4:]

    def raw_features(self) -> dict[str, float]:
        return self.model_dump(exclude={"transaction_id", "card_number"})

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "transaction_id": "tx-000123", "Time": 406.0, "Amount": 0.0,
        **{f"V{i}": v for i, v in enumerate([
            -2.312, 1.952, -1.610, 3.998, -0.522, -1.427, -2.537, 1.392, -2.770, -2.772,
            3.202, -2.900, -0.595, -4.289, 0.390, -1.141, -2.830, -0.017, 0.417, 0.127,
            0.517, -0.035, -0.465, 0.320, 0.045, 0.178, 0.261, -0.143], start=1)},
    }})


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1)


class ScoreResponse(BaseModel):
    transaction_id: str
    decision: Decision
    fraud_probability: float = Field(..., ge=0, le=1)
    threshold: float
    review_threshold: float
    model_version: str
    scorer: Literal["sagemaker", "local", "local-fallback"]
    latency_ms: float = Field(..., description="Tiempo de procesamiento en la API")
    explanation_status: Literal["queued", "unavailable"] = Field(
        ..., description="queued: el worker SHAP generará el reporte de auditoría")
    card_masked: str | None = None
    scored_at: datetime


class BatchResponse(BaseModel):
    results: list[ScoreResponse]
    count: int


class ModelInfo(BaseModel):
    model_version: str
    threshold: float
    review_threshold: float
    threshold_policy: str | None = None
    feature_columns: list[str]
    metrics_test: dict[str, float] = {}
    scorer_backend: str
    sagemaker_endpoint: str | None = None
    trained_at: str | None = None


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_version: str
    scorer: str
    rabbitmq: Literal["connected", "disconnected", "disabled"]
