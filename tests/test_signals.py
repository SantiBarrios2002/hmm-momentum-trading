import numpy as np
import pytest

from src.strategy.signals import predictions_to_signal, states_to_signal


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


def test_states_to_signal_dominated_state():
    state_probs = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    mu = np.array([-0.01, 0.0, 0.01], dtype=float)

    signals = states_to_signal(state_probs, mu)

    np.testing.assert_array_equal(signals, np.array([1.0, 1.0], dtype=float))


def test_states_to_signal_weighted_vote():
    state_probs = np.array([[0.7, 0.1, 0.2]], dtype=float)
    mu = np.array([-0.01, 0.0, 0.01], dtype=float)

    signals = states_to_signal(state_probs, mu)

    assert np.isclose(signals[0], -0.5)


def test_states_to_signal_output_range():
    rng = np.random.default_rng(123)
    raw = rng.uniform(size=(1000, 3))
    state_probs = raw / raw.sum(axis=1, keepdims=True)
    mu = np.array([-0.2, 0.0, 0.2], dtype=float)

    signals = states_to_signal(state_probs, mu)

    assert np.all(signals >= -1.0)
    assert np.all(signals <= 1.0)


def test_states_to_signal_rejects_shape_mismatch():
    state_probs = np.array([[0.6, 0.4, 0.0]], dtype=float)
    mu = np.array([-0.01, 0.01], dtype=float)

    with pytest.raises(ValueError):
        states_to_signal(state_probs, mu)


def test_states_to_signal_rejects_empty():
    with pytest.raises(ValueError):
        states_to_signal(np.empty((0, 3), dtype=float), np.array([-0.01, 0.0, 0.01]))
