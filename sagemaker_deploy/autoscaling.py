"""Registra (o actualiza) el auto-scaling de un endpoint ya InService.

    python -m sagemaker_deploy.autoscaling --min 1 --max 4 --target 3000
"""
from __future__ import annotations

import argparse

from sagemaker_deploy.common import DEFAULT_ENDPOINT, DEFAULT_REGION, session
from sagemaker_deploy.deploy import register_autoscaling


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--endpoint-name", default=DEFAULT_ENDPOINT)
    p.add_argument("--min", type=int, default=1)
    p.add_argument("--max", type=int, default=4)
    p.add_argument("--target", type=int, default=3000, help="invocaciones/min por instancia")
    a = p.parse_args(argv)
    register_autoscaling(session(a.region).client("application-autoscaling"),
                         a.endpoint_name, a.min, a.max, a.target)
    print(f"Auto-scaling registrado: {a.min}-{a.max} instancias, objetivo {a.target} inv/min/instancia")


if __name__ == "__main__":
    main()
