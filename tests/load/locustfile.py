"""Pruebas de carga de la API de fraude (Locust).

Perfiles de usuario (peso relativo):
* ``PaymentGateway`` (95 %): el sistema de pagos evaluando transacciones una a una
  (``POST /v1/transactions/score``), ~1 request/s por usuario simulado.
* ``BatchClient``   (3 %): reprocesos en lote (``/score/batch``, 20 transacciones).
* ``RiskAnalyst``   (2 %): analistas usando el dashboard (stats, listados, explicaciones).

Formas de carga (``LOAD_SHAPE``), escalables con ``LOAD_SCALE`` para que el mismo
guion sirva en local y en AWS (escala 1.0 = cifras del caso):

* ``steady``: rampa y meseta en el promedio del caso (150 TPS).
* ``spike``  : promedio → pico de 1.000 TPS → recuperación (valida auto-scaling).
* ``stress`` : escalones crecientes hasta encontrar el punto de quiebre.
* (sin LOAD_SHAPE): usuarios/tasa por línea de comandos (-u / -r).

``TIME_SCALE`` (< 1) acorta las etapas. Para más de ~500 TPS desde una sola
máquina, usar ``--processes -1`` (un proceso generador por CPU).

Al terminar se evalúa el SLA del caso: p99 ≤ ``SLA_P99_MS`` (90 ms) y tasa de error
≤ ``SLA_MAX_ERROR_RATE`` (1 %); si no se cumple, Locust termina con código 1.

    API_KEY=demo-key LOAD_SHAPE=steady LOAD_SCALE=0.2 \\
      locust -f tests/load/locustfile.py --host http://localhost:8000 --headless \\
      --csv tests/load/reports/steady --html tests/load/reports/steady.html
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

from locust import FastHttpUser, LoadTestShape, between, events, task

sys.path.insert(0, str(Path(__file__).resolve().parent))
API_KEY = os.environ.get("API_KEY", "")
TIME_SCALE = float(os.environ.get("TIME_SCALE", "1.0"))     # <1 acorta las etapas (pruebas rápidas)
SLA_P99_MS = float(os.environ.get("SLA_P99_MS", "90"))
SLA_MAX_ERROR_RATE = float(os.environ.get("SLA_MAX_ERROR_RATE", "0.01"))
SCORE_NAME = "POST /v1/transactions/score"

from payloads import STAGES, make_transaction, users_for  # noqa: E402  (mismo directorio)


class _ApiUser(FastHttpUser):
    """FastHttpUser (geventhttpclient): varias veces menos CPU por request que
    HttpUser/requests, para que el generador de carga no distorsione la latencia."""
    abstract = True
    default_headers = {"X-API-Key": API_KEY} if API_KEY else {}


def poisson_arrivals(rate_per_user: float = 1.0):
    """Llegadas de Poisson (esperas exponenciales): el tráfico de pagos real no llega en
    ráfagas sincronizadas. Con esperas fijas (constant_throughput) todos los usuarios
    disparan en el mismo instante cada segundo y la prueba mide colas artificiales."""
    return lambda user: random.expovariate(rate_per_user)


class PaymentGateway(_ApiUser):
    weight = 95
    wait_time = poisson_arrivals(1.0)    # ~1 req/s por usuario → usuarios ≈ TPS

    @task
    def score(self):
        with self.client.post("/v1/transactions/score", json=make_transaction(), name=SCORE_NAME,
                              catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
                return
            body = r.json()
            if body.get("decision") not in ("APPROVE", "REVIEW", "REJECT"):
                r.failure("respuesta sin decisión válida")


class BatchClient(_ApiUser):
    weight = 3
    wait_time = between(2, 5)

    @task
    def batch(self):
        with self.client.post("/v1/transactions/score/batch", name="POST /v1/transactions/score/batch",
                              json={"transactions": [make_transaction() for _ in range(20)]},
                              catch_response=True) as r:
            if r.status_code != 200 or r.json().get("count") != 20:
                r.failure(f"HTTP {r.status_code}")


class RiskAnalyst(_ApiUser):
    weight = 2
    wait_time = between(3, 8)

    @task(3)
    def stats(self):
        self.client.get("/v1/stats?hours=1", name="GET /v1/stats")

    @task(3)
    def recent_rejections(self):
        with self.client.get("/v1/transactions?decision=REJECT&limit=20", name="GET /v1/transactions",
                             catch_response=True) as r:
            if r.status_code == 200:
                self.rejected = [t["transaction_id"] for t in r.json()]

    @task(2)
    def explanation(self):
        ids = getattr(self, "rejected", None)
        if not ids:
            return
        with self.client.get(f"/v1/transactions/{random.choice(ids)}/explanation",
                             name="GET /v1/transactions/{id}/explanation", catch_response=True) as r:
            if r.status_code == 404:
                r.success()     # aún en cola del worker: no es un error

    @task(1)
    def health(self):
        self.client.get("/health", name="GET /health")


# ───────────────────────────── Formas de carga ─────────────────────────────
class _Stages(LoadTestShape):
    abstract = True
    stages: list[tuple[int, float]] = []   # (duración s, TPS objetivo)

    def tick(self):
        t, elapsed = self.get_run_time(), 0
        for duration, tps in self.stages:
            elapsed += duration * TIME_SCALE
            if t < elapsed:
                users = users_for(tps)
                return users, max(1, users // 5)
        return None


_SHAPE = os.environ.get("LOAD_SHAPE", "").lower()
if _SHAPE in STAGES:
    # Se define una sola clase de forma: Locust usa la que encuentre en el módulo
    LoadShape = type(f"{_SHAPE.capitalize()}Shape", (_Stages,), {"stages": STAGES[_SHAPE]})
elif _SHAPE:
    raise SystemExit(f"LOAD_SHAPE desconocido: {_SHAPE} (opciones: {', '.join(STAGES)})")


# ───────────────────────────── Evaluación del SLA ─────────────────────────────
@events.quitting.add_listener
def check_sla(environment, **_):
    stats = environment.stats.get(SCORE_NAME, "POST")
    total = environment.stats.total
    if not stats or stats.num_requests == 0:
        return
    p50, p95, p99 = (stats.get_response_time_percentile(q) for q in (0.5, 0.95, 0.99))
    err_rate = total.num_failures / max(total.num_requests, 1)
    ok = p99 <= SLA_P99_MS and err_rate <= SLA_MAX_ERROR_RATE
    print(f"\n=== SLA scoring: p50={p50:.0f} ms p95={p95:.0f} ms p99={p99:.0f} ms | "
          f"{stats.total_rps:.1f} req/s | errores {err_rate:.2%} → {'CUMPLE' if ok else 'NO CUMPLE'} "
          f"(p99 ≤ {SLA_P99_MS:.0f} ms, errores ≤ {SLA_MAX_ERROR_RATE:.0%}) ===")
    if not ok:
        environment.process_exit_code = 1
