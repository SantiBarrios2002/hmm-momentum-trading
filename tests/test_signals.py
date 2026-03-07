import numpy as np
import pytest

from src.strategy.signals import predictions_to_signal


def test_predictions_to_signal_sign_mode():
    predictions = np.array([0.01, -0.02, 0.0, 0.005, -0.001], dtype=float)

    signals = predictions_to_signal(predictions, transfer_fn="sign")

    np.testing.assert_array_equal(signals, np.sign(predictions))


def test_predictions_to_signal_linear_mode():
    predictions = np.array([0.5, -1.5, 0.0, 2.0], dtype=float)

    signals = predictions_to_signal(predictions, transfer_fn="linear", scale=1.0)

    np.testing.assert_allclose(signals, np.array([0.5, -1.0, 0.0, 1.0], dtype=float))


def test_predictions_to_signal_output_range():
    rng = np.random.default_rng(42)
    predictions = rng.normal(loc=0.0, scale=1.0, size=1000)

    signals_sign = predictions_to_signal(predictions, transfer_fn="sign")
    signals_linear = predictions_to_signal(predictions, transfer_fn="linear", scale=1.0)

    assert np.all(signals_sign >= -1.0)
    assert np.all(signals_sign <= 1.0)
    assert np.all(signals_linear >= -1.0)
    assert np.all(signals_linear <= 1.0)


def test_predictions_to_signal_rejects_empty():
    with pytest.raises(ValueError):
        predictions_to_signal(np.array([], dtype=float))


def test_predictions_to_signal_rejects_unknown_transfer_fn():
    with pytest.raises(ValueError):
        predictions_to_signal(np.array([0.01, -0.01], dtype=float), transfer_fn="unknown")
