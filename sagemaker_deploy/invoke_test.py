"""Prueba funcional y de latencia del endpoint desplegado.

* Envía las transacciones reales de ``models/sample_payloads.json`` (legítimas y fraudes).
* Si existe ``models/xgboost-model`` compara cada probabilidad con la predicción
  local (verifica que el contenedor carga el mismo modelo).
* Mide latencia cliente → endpoint (p50/p95/p99) con ``--requests`` llamadas.

    python -m sagemaker_deploy.invoke_test --requests 200
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from sagemaker_deploy.common import DEPLOYMENT_FILE, MODEL_DIR, load_json, session
from fraud_core import FEATURE_COLUMNS, build_features
from fraud_core.scoring import SageMakerScorer, Thresholds


def local_probabilities(features):
    path = MODEL_DIR / "xgboost-model"
    if not path.exists():
        return None
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(path)
    return booster.predict(xgb.DMatrix(features, feature_names=FEATURE_COLUMNS)).tolist()


def run(endpoint: str, region: str, n_requests: int, runtime_client=None) -> dict:
    dep = load_json(DEPLOYMENT_FILE) if DEPLOYMENT_FILE.exists() else {}
    meta = load_json(MODEL_DIR / "metadata.json")
    samples = load_json(MODEL_DIR / "sample_payloads.json")
    runtime_client = runtime_client or session(region).client("sagemaker-runtime")
    scorer = SageMakerScorer(endpoint, Thresholds.from_metadata(meta), runtime_client=runtime_client)

    txs = [(label, tx) for label in ("legit", "fraud") for tx in samples[label]]
    results = scorer.score([tx for _, tx in txs])
    local = local_probabilities(build_features([tx for _, tx in txs]))

    print(f"Endpoint: {endpoint} | modelo {dep.get('model_version', meta['model_version'])}")
    print(f"{'real':6s} {'prob. endpoint':>15s} {'prob. local':>12s}  decisión")
    max_diff = 0.0
    for i, ((label, _), r) in enumerate(zip(txs, results)):
        loc = local[i] if local else None
        if loc is not None:
            max_diff = max(max_diff, abs(loc - r["fraud_probability"]))
        print(f"{label:6s} {r['fraud_probability']:15.6f} {loc if loc is not None else float('nan'):12.6f}  {r['decision']}")
    if local is not None and max_diff > 1e-4:
        raise SystemExit(f"✗ El endpoint difiere del modelo local (máx. dif. {max_diff:.2e})")

    lat = []
    one = [samples["legit"][0]]
    for i in range(n_requests):
        t = time.perf_counter()
        scorer.score(one if i % 2 else [samples["fraud"][0]])
        lat.append((time.perf_counter() - t) * 1000)
    stats = {f"p{q}_ms": round(float(np.percentile(lat, q)), 2) for q in (50, 95, 99)}
    stats.update(requests=n_requests, max_abs_diff_vs_local=max_diff)
    print("Latencia cliente→endpoint:", stats)
    (MODEL_DIR / "endpoint_test.json").write_text(json.dumps(
        {"endpoint": endpoint, "results": results, "latency": stats}, indent=2))
    return stats


def main(argv=None) -> None:
    dep = load_json(DEPLOYMENT_FILE) if DEPLOYMENT_FILE.exists() else {}
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint-name", default=dep.get("endpoint_name", "pagoseguro-fraud"))
    p.add_argument("--region", default=dep.get("region", "us-east-1"))
    p.add_argument("--requests", type=int, default=100)
    a = p.parse_args(argv)
    run(a.endpoint_name, a.region, a.requests)


if __name__ == "__main__":
    main()
