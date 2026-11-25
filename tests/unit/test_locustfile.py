"""Payloads y etapas de carga de Locust (sin importar Locust: su monkey-patching de
gevent interfiere con el resto de la suite)."""
import importlib
import sys
from pathlib import Path

import pytest

LOAD_DIR = Path(__file__).resolve().parents[1] / "load"


@pytest.fixture
def payloads(monkeypatch):
    monkeypatch.syspath_prepend(str(LOAD_DIR))
    sys.modules.pop("payloads", None)
    return importlib.import_module("payloads")


def test_transacciones_generadas_pasan_la_validacion_de_la_api(payloads):
    from api.app.schemas import Transaction
    for _ in range(200):
        tx = Transaction(**payloads.make_transaction())
        assert tx.Amount >= 0 and tx.transaction_id.startswith("load-")


@pytest.mark.parametrize("shape,peak", [("steady", 150), ("spike", 1000), ("stress", 2000)])
def test_etapas_y_escala(payloads, shape, peak):
    assert max(tps for _, tps in payloads.STAGES[shape]) == peak
    assert payloads.users_for(peak, scale=0.5) == peak // 2
    assert payloads.users_for(1, scale=0.01) == 1


def test_locustfile_carga_con_locust():
    """Importa el locustfile en un proceso aparte (como lo hace Locust)."""
    import os
    import subprocess
    import importlib.util
    if importlib.util.find_spec("locust") is None:      # no importar: aplica monkey-patching
        pytest.skip("locust no instalado")
    env = {**os.environ, "LOAD_SHAPE": "spike"}
    code = ("import sys; sys.path.insert(0, 'tests/load'); import locustfile as lf; "
            "assert lf.LoadShape.stages[2] == (60, 1000); print('ok')")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         cwd=LOAD_DIR.parents[1], timeout=60)
    assert out.stdout.strip() == "ok", out.stderr[-2000:]
