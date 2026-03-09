import numpy as np
import pytest

from src.strategy.signals import (
    apply_no_trade_zone,
    predictions_to_signal,
    smooth_signal,
    states_to_signal,
)


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


# --- apply_no_trade_zone tests ---


def test_no_trade_zone_zeros_signal_above_threshold():
    """When neutral posterior exceeds threshold, signal should be zeroed."""
    signals = np.array([0.8, -0.5, 0.3, -0.9, 0.6])
    state_probs = np.array([
        [0.1, 0.7, 0.2],  # neutral=0.7, above 0.5 -> zero
        [0.6, 0.2, 0.2],  # neutral=0.2, below 0.5 -> keep
        [0.1, 0.5, 0.4],  # neutral=0.5, not < 0.5 -> zero
        [0.2, 0.1, 0.7],  # neutral=0.1, below 0.5 -> keep
        [0.0, 0.9, 0.1],  # neutral=0.9, above 0.5 -> zero
    ])

    result = apply_no_trade_zone(signals, state_probs, neutral_idx=1, threshold=0.5)

    expected = np.array([0.0, -0.5, 0.0, -0.9, 0.0])
    np.testing.assert_array_equal(result, expected)


def test_no_trade_zone_threshold_one_keeps_all():
    """Threshold=1.0 means only exact 1.0 neutral posterior is zeroed."""
    signals = np.array([0.5, -0.3])
    state_probs = np.array([
        [0.2, 0.6, 0.2],
        [0.3, 0.4, 0.3],
    ])

    result = apply_no_trade_zone(signals, state_probs, neutral_idx=1, threshold=1.0)

    np.testing.assert_array_equal(result, signals)


def test_no_trade_zone_output_range():
    """Output must stay in [-1, 1]."""
    rng = np.random.default_rng(99)
    signals = rng.uniform(-1.0, 1.0, size=500)
    raw = rng.uniform(size=(500, 3))
    state_probs = raw / raw.sum(axis=1, keepdims=True)

    result = apply_no_trade_zone(signals, state_probs, neutral_idx=1, threshold=0.5)

    assert np.all(result >= -1.0)
    assert np.all(result <= 1.0)


def test_no_trade_zone_rejects_length_mismatch():
    with pytest.raises(ValueError):
        apply_no_trade_zone(
            np.array([0.5, -0.3]),
            np.array([[0.2, 0.6, 0.2]]),
            neutral_idx=1,
            threshold=0.5,
        )


def test_no_trade_zone_rejects_invalid_threshold():
    signals = np.array([0.5])
    state_probs = np.array([[0.2, 0.6, 0.2]])
    with pytest.raises(ValueError):
        apply_no_trade_zone(signals, state_probs, neutral_idx=1, threshold=0.0)
    with pytest.raises(ValueError):
        apply_no_trade_zone(signals, state_probs, neutral_idx=1, threshold=1.5)


def test_no_trade_zone_rejects_invalid_neutral_idx():
    signals = np.array([0.5])
    state_probs = np.array([[0.2, 0.6, 0.2]])
    with pytest.raises(ValueError):
        apply_no_trade_zone(signals, state_probs, neutral_idx=3, threshold=0.5)


# --- smooth_signal tests ---


def test_smooth_signal_alpha_one_returns_original():
    """alpha=1.0 means no smoothing — output equals input."""
    signals = np.array([0.5, -0.3, 0.8, -0.1, 0.0])

    result = smooth_signal(signals, alpha=1.0)

    np.testing.assert_array_equal(result, signals)


def test_smooth_signal_ema_recursion():
    """Verify the EMA recursion manually for alpha=0.5."""
    signals = np.array([1.0, -1.0, 1.0, -1.0])
    alpha = 0.5

    # s'_0 = 1.0
    # s'_1 = 0.5*(-1.0) + 0.5*1.0 = 0.0
    # s'_2 = 0.5*(1.0) + 0.5*0.0 = 0.5
    # s'_3 = 0.5*(-1.0) + 0.5*0.5 = -0.25
    expected = np.array([1.0, 0.0, 0.5, -0.25])

    result = smooth_signal(signals, alpha=alpha)

    np.testing.assert_allclose(result, expected)


def test_smooth_signal_reduces_variance():
    """Smoothing should reduce signal variance compared to the raw signal."""
    rng = np.random.default_rng(7)
    signals = rng.choice([-1.0, 0.0, 1.0], size=500)

    smoothed = smooth_signal(signals, alpha=0.2)

    assert np.var(smoothed) < np.var(signals)


def test_smooth_signal_output_range():
    """Output must stay in [-1, 1]."""
    rng = np.random.default_rng(42)
    signals = rng.uniform(-1.0, 1.0, size=500)

    result = smooth_signal(signals, alpha=0.3)

    assert np.all(result >= -1.0)
    assert np.all(result <= 1.0)


def test_smooth_signal_rejects_empty():
    with pytest.raises(ValueError):
        smooth_signal(np.array([], dtype=float), alpha=0.5)


def test_smooth_signal_rejects_invalid_alpha():
    signals = np.array([0.5, -0.3])
    with pytest.raises(ValueError):
        smooth_signal(signals, alpha=0.0)
    with pytest.raises(ValueError):
        smooth_signal(signals, alpha=1.5)
