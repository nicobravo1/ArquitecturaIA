import pandas as pd
import pytest

from fraud_core import FEATURE_COLUMNS, RAW_COLUMNS, build_features, to_sagemaker_csv


def _tx(**overrides):
    tx = {c: 0.0 for c in RAW_COLUMNS}
    tx.update(Time=3600 * 26 + 5, Amount=149.62)
    tx.update(overrides)
    return tx


def test_build_features_orden_y_hora():
    feats = build_features(_tx())
    assert list(feats.columns) == FEATURE_COLUMNS
    assert feats.loc[0, "hour"] == 2  # 26 h -> 02:00 del día siguiente
    assert feats.loc[0, "Amount"] == pytest.approx(149.62)


def test_build_features_lista_y_dataframe_equivalentes():
    txs = [_tx(), _tx(Amount=1.0)]
    pd.testing.assert_frame_equal(build_features(txs), build_features(pd.DataFrame(txs)))


def test_build_features_columna_faltante():
    tx = _tx()
    del tx["V7"]
    with pytest.raises(ValueError, match="V7"):
        build_features(tx)


def test_build_features_monto_negativo():
    with pytest.raises(ValueError):
        build_features(_tx(Amount=-1))


def test_to_sagemaker_csv_sin_cabecera():
    csv = to_sagemaker_csv(build_features([_tx(), _tx()]))
    lines = csv.splitlines()
    assert len(lines) == 2
    assert len(lines[0].split(",")) == len(FEATURE_COLUMNS)
