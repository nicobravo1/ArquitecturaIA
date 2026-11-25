"""Extrae métricas de CloudWatch del endpoint (evidencia para el informe, ítem 1.2.d).

Métricas AWS/SageMaker: Invocations, ModelLatency y OverheadLatency (µs),
Invocation4XXErrors, Invocation5XXErrors y CPUUtilization (/aws/sagemaker/Endpoints).

    python -m sagemaker_deploy.cloudwatch_metrics --hours 3
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone

from sagemaker_deploy.common import DEPLOYMENT_FILE, MODEL_DIR, VARIANT, load_json, session

METRICS = [
    # (namespace, métrica, estadística, factor -> unidad mostrada)
    ("AWS/SageMaker", "Invocations", "Sum", 1, "count"),
    ("AWS/SageMaker", "ModelLatency", "Average", 1 / 1000, "ms"),
    ("AWS/SageMaker", "ModelLatency", "p99", 1 / 1000, "ms"),
    ("AWS/SageMaker", "OverheadLatency", "Average", 1 / 1000, "ms"),
    ("AWS/SageMaker", "Invocation4XXErrors", "Sum", 1, "count"),
    ("AWS/SageMaker", "Invocation5XXErrors", "Sum", 1, "count"),
    ("/aws/sagemaker/Endpoints", "CPUUtilization", "Average", 1, "%"),
]


def collect(cw, endpoint: str, hours: float, period: int) -> list[dict]:
    end = datetime.now(timezone.utc)
    queries = []
    for i, (ns, name, stat, _, _) in enumerate(METRICS):
        queries.append({
            "Id": f"m{i}",
            "Label": f"{name}:{stat}",
            "MetricStat": {
                "Metric": {"Namespace": ns, "MetricName": name, "Dimensions": [
                    {"Name": "EndpointName", "Value": endpoint},
                    {"Name": "VariantName", "Value": VARIANT}]},
                "Period": period, "Stat": stat},
        })
    resp = cw.get_metric_data(MetricDataQueries=queries, StartTime=end - timedelta(hours=hours), EndTime=end)
    rows = []
    for (ns, name, stat, factor, unit), res in zip(METRICS, resp["MetricDataResults"]):
        for ts, val in zip(res["Timestamps"], res["Values"]):
            rows.append({"timestamp": ts.isoformat(), "metric": name, "stat": stat,
                         "value": round(val * factor, 4), "unit": unit})
    return sorted(rows, key=lambda r: (r["metric"], r["stat"], r["timestamp"]))


def summarize(rows: list[dict]) -> dict:
    out: dict[str, float] = {}
    for r in rows:
        key = f"{r['metric']}_{r['stat']}_{r['unit']}"
        if r["stat"] == "Sum":
            out[key] = out.get(key, 0) + r["value"]
        else:
            out[key] = max(out.get(key, 0), r["value"])
    return out


def main(argv=None) -> None:
    dep = load_json(DEPLOYMENT_FILE) if DEPLOYMENT_FILE.exists() else {}
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint-name", default=dep.get("endpoint_name", "pagoseguro-fraud"))
    p.add_argument("--region", default=dep.get("region", "us-east-1"))
    p.add_argument("--hours", type=float, default=3)
    p.add_argument("--period", type=int, default=60)
    a = p.parse_args(argv)
    rows = collect(session(a.region).client("cloudwatch"), a.endpoint_name, a.hours, a.period)
    out = MODEL_DIR / "cloudwatch_metrics.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "metric", "stat", "value", "unit"])
        w.writeheader(); w.writerows(rows)
    print(json.dumps(summarize(rows), indent=2))
    print(f"{len(rows)} puntos → {out}")


if __name__ == "__main__":
    main()
