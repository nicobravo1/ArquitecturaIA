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


def test_features_matrix_equivale_a_build_features():
    import numpy as np
    from fraud_core import features_matrix
    txs = [_tx(), _tx(Amount=3.5, Time=90_000, V3=-1.25)]
    np.testing.assert_array_equal(features_matrix(txs), build_features(txs).to_numpy())
    assert to_sagemaker_csv(features_matrix(txs)) == to_sagemaker_csv(build_features(txs))


def test_features_matrix_valida():
    from fraud_core import features_matrix
    tx = _tx(); del tx["Time"]
    with pytest.raises(ValueError, match="Time"):
        features_matrix([tx])
    with pytest.raises(ValueError, match="negativo"):
        features_matrix([_tx(Amount=-2)])
