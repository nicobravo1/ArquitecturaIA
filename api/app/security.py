"""Autenticación por API key y rate limiting por cliente."""
from __future__ import annotations

import hashlib
import secrets
import threading
import time
from collections import deque

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False,
                              description="API key del cliente (header X-API-Key)")


def client_id(api_key: str) -> str:
    """Identificador no reversible de la key (para logs, métricas y rate limit)."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:12]


class SlidingWindowRateLimiter:
    """Límite de N requests por ventana de 60 s por cliente (en memoria, thread-safe).

    Con varias réplicas detrás de un balanceador se reemplaza por Redis
    (ElastiCache) o por el throttling de API Gateway / WAF.
    """

    def __init__(self, limit: int, window_s: float = 60.0):
        self.limit, self.window = limit, window_s
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int, float]:
        """Devuelve (permitido, restantes, segundos hasta liberar un cupo)."""
        now = time.monotonic()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False, 0, self.window - (now - q[0])
            q.append(now)
            return True, self.limit - len(q), 0.0


def require_api_key(request: Request, api_key: str | None = Security(api_key_header)) -> str:
    settings = request.app.state.settings
    if not settings.api_keys:
        # Modo desarrollo sin keys configuradas: se limita por IP de origen
        cid = f"ip:{request.client.host if request.client else 'unknown'}"
    elif api_key and any(secrets.compare_digest(api_key, k) for k in settings.api_keys):
        cid = client_id(api_key)
    else:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "API key inválida o ausente",
                            headers={"WWW-Authenticate": "ApiKey"})
    allowed, remaining, retry = request.app.state.rate_limiter.check(cid)
    request.state.rate_limit_remaining = remaining
    if not allowed:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Límite de solicitudes excedido",
                            headers={"Retry-After": str(max(1, int(retry) + 1))})
    return cid
