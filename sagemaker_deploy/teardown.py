"""Elimina endpoint, endpoint configs, modelos y auto-scaling del proyecto.

Importante en AWS Academy: un endpoint en tiempo real cobra por hora aunque no
reciba tráfico. Ejecutar al terminar cada sesión de trabajo / demo.

    python -m sagemaker_deploy.teardown            # pide confirmación
    python -m sagemaker_deploy.teardown --yes
"""
from __future__ import annotations

import argparse

from sagemaker_deploy.common import DEFAULT_ENDPOINT, DEFAULT_REGION, DEPLOYMENT_FILE, PROJECT, session
from sagemaker_deploy.deploy import deregister_autoscaling, endpoint_exists


def teardown(sess, endpoint: str, keep_models: bool = False) -> list[str]:
    sm, aas = sess.client("sagemaker"), sess.client("application-autoscaling")
    deleted = []
    if endpoint_exists(sm, endpoint):
        deregister_autoscaling(aas, endpoint)
        sm.delete_endpoint(EndpointName=endpoint)
        deleted.append(f"endpoint/{endpoint}")
    for cfg in sm.list_endpoint_configs(NameContains=endpoint, MaxResults=100)["EndpointConfigs"]:
        sm.delete_endpoint_config(EndpointConfigName=cfg["EndpointConfigName"])
        deleted.append(f"endpoint-config/{cfg['EndpointConfigName']}")
    if not keep_models:
        for m in sm.list_models(NameContains=PROJECT, MaxResults=100)["Models"]:
            sm.delete_model(ModelName=m["ModelName"])
            deleted.append(f"model/{m['ModelName']}")
    return deleted


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--endpoint-name", default=DEFAULT_ENDPOINT)
    p.add_argument("--keep-models", action="store_true", help="Conserva los Model (no cobran)")
    p.add_argument("--yes", action="store_true")
    a = p.parse_args(argv)
    if not a.yes and input(f"¿Eliminar el endpoint '{a.endpoint_name}' en {a.region}? [s/N] ").lower() != "s":
        return
    for item in teardown(session(a.region), a.endpoint_name, a.keep_models) or ["(nada que eliminar)"]:
        print("eliminado:", item)
    DEPLOYMENT_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
