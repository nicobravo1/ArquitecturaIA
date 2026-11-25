"""API REST de detección de fraude – PagoSeguro.

Flujo del *command side* (CQRS)::

    POST /v1/transactions/score
      → validación (Pydantic) + API key + rate limit
      → fraud_core.build_features
      → SageMaker endpoint (fallback: modelo local)
      → decisión APPROVE / REVIEW / REJECT según umbrales del modelo
      → evento a RabbitMQ (worker SHAP genera el reporte de auditoría)
      → respuesta al sistema de pagos

Ejecutar::

    uvicorn api.app.main:create_app --factory --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest

from fraud_core import build_features
from fraud_core.scoring import Thresholds, decide

from .config import Settings
from .events import Publisher, build_publisher
from .schemas import (
    BatchRequest, BatchResponse, Decision, Explanation, Health, ModelInfo, ScoreResponse, Stats,
    Transaction, TransactionSummary,
)
from .scorers import ScoringService, build_scoring_service
from .security import SlidingWindowRateLimiter, require_api_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")

TAGS = [
    {"name": "scoring", "description": "Evaluación de transacciones en tiempo real"},
    {"name": "auditoría", "description": "Consultas sobre los reportes SHAP generados por el worker (CQRS)"},
    {"name": "modelo", "description": "Información del modelo desplegado"},
    {"name": "operación", "description": "Salud y métricas (Prometheus)"},
]


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.latency = Histogram("fraud_api_request_latency_seconds", "Latencia de scoring en la API",
                                 ["endpoint"], registry=self.registry,
                                 buckets=(.005, .01, .025, .05, .075, .1, .15, .25, .5, 1))
        self.decisions = Counter("fraud_api_decisions_total", "Decisiones emitidas",
                                 ["decision", "scorer"], registry=self.registry)
        self.events = Counter("fraud_api_events_total", "Eventos a RabbitMQ", ["status"], registry=self.registry)
        self.errors = Counter("fraud_api_scoring_errors_total", "Errores del backend de scoring",
                              registry=self.registry)
        self.probability = Histogram("fraud_api_fraud_probability", "Distribución de probabilidades "
                                     "(monitoreo de drift)", registry=self.registry,
                                     buckets=(.001, .01, .05, .1, .2, .3, .5, .7, .9, 1))


def create_app(settings: Settings | None = None, scoring: ScoringService | None = None,
               publisher: Publisher | None = None, audit_store=None) -> FastAPI:
    settings = settings or Settings()
    metadata = settings.metadata()
    thresholds = Thresholds.from_metadata(metadata)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.scoring = scoring or build_scoring_service(settings, thresholds)
        app.state.publisher = publisher or build_publisher(settings)
        app.state.audit = audit_store
        if app.state.audit is None and settings.database_url:
            from fraud_core.audit_store import AuditStore
            app.state.audit = AuditStore(settings.database_url)
        await app.state.publisher.start()
        log.info("API lista | modelo %s | backend %s", metadata["model_version"], app.state.scoring.name)
        yield
        await app.state.publisher.close()

    app = FastAPI(
        title="PagoSeguro · API de Detección de Fraude",
        version="1.0.0",
        description=(
            "Scoring de transacciones en tiempo real con XGBoost desplegado en AWS SageMaker. "
            "Cada decisión se publica en RabbitMQ para que el worker SHAP genere el reporte "
            "de explicabilidad exigido por la CMF.\n\n"
            "**Autenticación:** header `X-API-Key`. **Rate limit:** por cliente, por minuto "
            "(cabeceras `X-RateLimit-*`, respuesta 429 con `Retry-After`)."
        ),
        openapi_tags=TAGS,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.metadata = metadata
    app.state.thresholds = thresholds
    app.state.rate_limiter = SlidingWindowRateLimiter(settings.rate_limit_per_minute)
    app.state.metrics = metrics = Metrics()

    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                           allow_methods=["GET", "POST"], allow_headers=["X-API-Key", "Content-Type"])

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - t0) * 1000:.2f}"
        remaining = getattr(request.state, "rate_limit_remaining", None)
        if remaining is not None:
            response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
        # Cabeceras de seguridad básicas
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    async def score_many(request: Request, txs: list[Transaction], client: str) -> list[ScoreResponse]:
        t0 = time.perf_counter()
        features = build_features([t.raw_features() for t in txs])
        try:
            probs, used = await asyncio.to_thread(request.app.state.scoring.predict_proba, features)
        except Exception:
            metrics.errors.inc()
            log.exception("Fallo de scoring")
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                                "Servicio de scoring no disponible") from None
        now = datetime.now(timezone.utc)
        latency_ms = (time.perf_counter() - t0) * 1000
        responses, events = [], []
        for tx, p in zip(txs, probs):
            decision = decide(p, thresholds)
            tx_id = tx.transaction_id or f"tx-{uuid.uuid4().hex[:16]}"
            responses.append(dict(
                transaction_id=tx_id, decision=decision, fraud_probability=round(p, 6),
                threshold=thresholds.reject, review_threshold=thresholds.review,
                model_version=metadata["model_version"], scorer=used, latency_ms=round(latency_ms, 2),
                card_masked=tx.masked_pan(), scored_at=now))
            events.append({
                "event_type": "transaction.scored", "transaction_id": tx_id, "scored_at": now.isoformat(),
                "decision": decision, "fraud_probability": p, "threshold": thresholds.reject,
                "review_threshold": thresholds.review, "model_version": metadata["model_version"],
                "scorer": used, "client": client, "card_masked": tx.masked_pan(),
                "features": tx.raw_features(),   # sin PAN
            })
            metrics.decisions.labels(decision, used).inc()
            metrics.probability.observe(p)
        published = await asyncio.gather(*(request.app.state.publisher.publish(e) for e in events))
        for r, ok in zip(responses, published):
            r["explanation_status"] = "queued" if ok else "unavailable"
            metrics.events.labels("published" if ok else "failed").inc()
        return [ScoreResponse(**r) for r in responses]

    @app.post("/v1/transactions/score", response_model=ScoreResponse, tags=["scoring"],
              summary="Evalúa una transacción",
              responses={401: {"description": "API key inválida"}, 422: {"description": "Payload inválido"},
                         429: {"description": "Rate limit"}, 503: {"description": "Scoring no disponible"}})
    async def score(tx: Transaction, request: Request, client: str = Depends(require_api_key)):
        with metrics.latency.labels("score").time():
            return (await score_many(request, [tx], client))[0]

    @app.post("/v1/transactions/score/batch", response_model=BatchResponse, tags=["scoring"],
              summary="Evalúa un lote de transacciones (máx. MAX_BATCH_SIZE)")
    async def score_batch(body: BatchRequest, request: Request, client: str = Depends(require_api_key)):
        if len(body.transactions) > settings.max_batch_size:
            raise HTTPException(413,
                                f"Máximo {settings.max_batch_size} transacciones por lote")
        with metrics.latency.labels("batch").time():
            results = await score_many(request, body.transactions, client)
        return BatchResponse(results=results, count=len(results))

    def audit_store_or_503(request: Request):
        store = request.app.state.audit
        if store is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Almacén de auditoría no configurado")
        return store

    async def audit_call(fn, *args, **kwargs):
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        except Exception:
            log.exception("Error consultando el almacén de auditoría")
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Almacén de auditoría no disponible") from None

    @app.get("/v1/transactions/{transaction_id}/explanation", response_model=Explanation, tags=["auditoría"],
             summary="Reporte SHAP de una transacción",
             responses={404: {"description": "Aún no procesada por el worker o inexistente"}})
    async def get_explanation(transaction_id: str, request: Request, _: str = Depends(require_api_key)):
        store = audit_store_or_503(request)
        rec = await audit_call(store.get, transaction_id)
        if rec is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND,
                                "Explicación no disponible (inexistente o aún en cola del worker)")
        return rec

    @app.get("/v1/transactions", response_model=list[TransactionSummary], tags=["auditoría"],
             summary="Transacciones auditadas recientes")
    async def list_transactions(request: Request, decision: Decision | None = None,
                                limit: int = Query(50, ge=1, le=500),
                                hours: float | None = Query(None, gt=0, le=24 * 90),
                                _: str = Depends(require_api_key)):
        store = audit_store_or_503(request)
        since = datetime.now(timezone.utc) - timedelta(hours=hours) if hours else None
        return await audit_call(store.list_recent, limit=limit, decision=decision, since=since)

    @app.get("/v1/stats", response_model=Stats, tags=["auditoría"], summary="Resumen de decisiones")
    async def stats(request: Request, hours: float = Query(24, gt=0, le=24 * 90),
                    _: str = Depends(require_api_key)):
        store = audit_store_or_503(request)
        data = await audit_call(store.stats, since=datetime.now(timezone.utc) - timedelta(hours=hours))
        return Stats(window_hours=hours, **data)

    @app.get("/v1/model", response_model=ModelInfo, tags=["modelo"], summary="Modelo y umbrales en uso")
    async def model_info(request: Request, _: str = Depends(require_api_key)):
        return ModelInfo(
            model_version=metadata["model_version"], threshold=thresholds.reject,
            review_threshold=thresholds.review, threshold_policy=metadata.get("threshold_policy"),
            feature_columns=metadata["feature_columns"], metrics_test=metadata.get("metrics_test", {}),
            scorer_backend=request.app.state.scoring.name, trained_at=metadata.get("trained_at"),
            sagemaker_endpoint=settings.sagemaker_endpoint or settings.deployment().get("endpoint_name")
            if settings.scorer_backend == "sagemaker" else None)

    @app.get("/health", response_model=Health, tags=["operación"], summary="Liveness/readiness")
    async def health(request: Request):
        pub = request.app.state.publisher
        rabbit = "disabled" if not settings.rabbitmq_url else ("connected" if pub.connected else "disconnected")
        store = request.app.state.audit
        audit = "disabled" if store is None else (
            "connected" if await asyncio.to_thread(store.ping) else "disconnected")
        return Health(status="degraded" if "disconnected" in (rabbit, audit) else "ok",
                      model_version=metadata["model_version"], scorer=request.app.state.scoring.name,
                      rabbitmq=rabbit, audit_store=audit)

    @app.get("/metrics", tags=["operación"], summary="Métricas Prometheus", include_in_schema=True)
    async def prometheus_metrics():
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app
