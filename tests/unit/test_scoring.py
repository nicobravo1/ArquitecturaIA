import io
import json

import pytest

from fraud_core import RAW_COLUMNS
from fraud_core.scoring import (
    APPROVE, REJECT, REVIEW, SageMakerScorer, Thresholds, decide, parse_predictions,
)

TH = Thresholds(reject=0.8, review=0.4)


@pytest.mark.parametrize("p,expected", [(0.0, APPROVE), (0.39, APPROVE), (0.4, REVIEW),
                                        (0.79, REVIEW), (0.8, REJECT), (1.0, REJECT)])
def test_decide_bordes_de_umbral(p, expected):
    assert decide(p, TH) == expected


def test_thresholds_desde_metadata_y_validacion():
    assert Thresholds.from_metadata({"threshold": 0.7}) == Thresholds(0.7, 0.7)
    with pytest.raises(ValueError):
        Thresholds.from_metadata({"threshold": 0.3, "review_threshold": 0.5})


@pytest.mark.parametrize("body,ctype", [
    (b'{"predictions": [{"score": 0.1}, {"score": 0.9}]}', "application/json"),
    (b"0.1,0.9", "text/csv"),
    (b"0.1\n0.9\n", "text/csv; charset=utf-8"),
    (b"[0.1, 0.9]", ""),
])
def test_parse_predictions_formatos_del_contenedor(body, ctype):
    assert parse_predictions(body, ctype) == pytest.approx([0.1, 0.9])


def test_parse_predictions_notacion_cientifica_y_rango():
    assert parse_predictions(b"1.5e-05", "text/csv") == pytest.approx([1.5e-05])
    with pytest.raises(ValueError):
        parse_predictions(b"1.7", "text/csv")


class FakeRuntime:
    def __init__(self, scores):
        self.scores, self.calls = scores, []

    def invoke_endpoint(self, **kwargs):
        self.calls.append(kwargs)
        body = json.dumps({"predictions": [{"score": s} for s in self.scores]}).encode()
        return {"Body": io.BytesIO(body), "ContentType": "application/json"}


def _tx(**kw):
    return {c: 0.0 for c in RAW_COLUMNS} | {"Amount": 5.0} | kw


def test_scorer_serializa_csv_y_decide():
    rt = FakeRuntime([0.05, 0.95])
    out = SageMakerScorer("ep", TH, runtime_client=rt).score([_tx(), _tx(Time=7200)])
    assert [r["decision"] for r in out] == [APPROVE, REJECT]
    call = rt.calls[0]
    assert call["EndpointName"] == "ep" and call["ContentType"] == "text/csv"
    rows = call["Body"].decode().splitlines()
    assert len(rows) == 2 and rows[1].endswith(",2.0")      # hour = 2


def test_scorer_detecta_respuesta_incompleta():
    with pytest.raises(ValueError, match="1 predicciones para 2"):
        SageMakerScorer("ep", TH, runtime_client=FakeRuntime([0.1])).score([_tx(), _tx()])
