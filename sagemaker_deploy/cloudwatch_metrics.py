"""Extrae métricas de CloudWatch del endpoint (evidencia para el informe, ítem 1.2.d).

Métricas AWS/SageMaker: Invocations, ModelLatency y OverheadLatency (µs),
Invocation4XXErrors, Invocation5XXErrors y CPUUtilization (/aws/sagemaker/Endpoints);
y de la instancia EC2 de la aplicación: CPUUtilization, NetworkIn/Out.

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


# Métricas de la instancia EC2 de la aplicación (infra/aws/ec2.py)
EC2_METRICS = [
    ("AWS/EC2", "CPUUtilization", "Average", 1, "%"),
    ("AWS/EC2", "NetworkIn", "Sum", 1 / 1e6, "MB"),
    ("AWS/EC2", "NetworkOut", "Sum", 1 / 1e6, "MB"),
]


def collect(cw, endpoint: str, hours: float, period: int, instance_id: str | None = None) -> list[dict]:
    end = datetime.now(timezone.utc)
    specs, queries = [], []
    endpoint_dims = [{"Name": "EndpointName", "Value": endpoint}, {"Name": "VariantName", "Value": VARIANT}]
    for spec in METRICS:
        specs.append(("endpoint", spec, endpoint_dims))
    if instance_id:
        for spec in EC2_METRICS:
            specs.append(("ec2", spec, [{"Name": "InstanceId", "Value": instance_id}]))
    for i, (_, (ns, name, stat, _, _), dims) in enumerate(specs):
        queries.append({"Id": f"m{i}", "Label": f"{name}:{stat}", "MetricStat": {
            "Metric": {"Namespace": ns, "MetricName": name, "Dimensions": dims},
            "Period": period, "Stat": stat}})
    resp = cw.get_metric_data(MetricDataQueries=queries, StartTime=end - timedelta(hours=hours), EndTime=end)
    rows = []
    for (source, (ns, name, stat, factor, unit), _), res in zip(specs, resp["MetricDataResults"]):
        for ts, val in zip(res["Timestamps"], res["Values"]):
            rows.append({"timestamp": ts.isoformat(), "source": source, "metric": name, "stat": stat,
                         "value": round(val * factor, 4), "unit": unit})
    return sorted(rows, key=lambda r: (r["metric"], r["stat"], r["timestamp"]))


def summarize(rows: list[dict]) -> dict:
    out: dict[str, float] = {}
    for r in rows:
        key = f"{r['source']}.{r['metric']}_{r['stat']}_{r['unit']}"
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
    state_file = MODEL_DIR.parent / "infra" / "aws" / ".state.json"
    default_iid = json.loads(state_file.read_text()).get("instance_id") if state_file.exists() else None
    p.add_argument("--instance-id", default=default_iid, help="instancia EC2 de la app (por defecto la de infra/aws)")
    a = p.parse_args(argv)
    rows = collect(session(a.region).client("cloudwatch"), a.endpoint_name, a.hours, a.period, a.instance_id)
    out = MODEL_DIR / "cloudwatch_metrics.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "source", "metric", "stat", "value", "unit"])
        w.writeheader(); w.writerows(rows)
    print(json.dumps(summarize(rows), indent=2))
    print(f"{len(rows)} puntos → {out}")


if __name__ == "__main__":
    main()
