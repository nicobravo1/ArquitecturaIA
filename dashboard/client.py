"""Cliente HTTP de la API de fraude. El dashboard solo consume la API REST
(no accede a la base de datos ni al modelo): es un cliente más del sistema."""
from __future__ import annotations

from typing import Any

import httpx


class ApiError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


class FraudApiClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 5.0,
                 transport: httpx.BaseTransport | None = None):
        headers = {"X-API-Key": api_key} if api_key else {}
        self._http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout,
                                  transport=transport)

    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            r = self._http.request(method, path, **kwargs)
        except httpx.HTTPError as err:
            raise ApiError(0, f"No se pudo conectar con la API ({err.__class__.__name__})") from None
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise ApiError(r.status_code, detail if isinstance(detail, str) else str(detail))
        return r.json()

    def health(self) -> dict:
        return self._request("GET", "/health")

    def model(self) -> dict:
        return self._request("GET", "/v1/model")

    def stats(self, hours: float) -> dict:
        return self._request("GET", "/v1/stats", params={"hours": hours})

    def transactions(self, limit: int = 200, decision: str | None = None, hours: float | None = None) -> list[dict]:
        params: dict[str, Any] = {"limit": limit}
        if decision:
            params["decision"] = decision
        if hours:
            params["hours"] = hours
        return self._request("GET", "/v1/transactions", params=params)

    def explanation(self, transaction_id: str) -> dict | None:
        try:
            return self._request("GET", f"/v1/transactions/{transaction_id}/explanation")
        except ApiError as err:
            if err.status == 404:
                return None
            raise

    def score(self, transaction: dict) -> dict:
        return self._request("POST", "/v1/transactions/score", json=transaction)
