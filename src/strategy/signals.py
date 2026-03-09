"""Signal generation utilities for HMM-based trading strategies."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def predictions_to_signal(
    predictions: NDArray[np.floating],
    transfer_fn: str = "sign",
    *,
    scale: float | None = None,
) -> NDArray[np.floating]:
    """
    Convert return predictions to trading signals (Paper §7, strategy evaluation).

    Maps one-step-ahead return predictions from online inference to positions:
        sign mode:   signal_t = sign(prediction_t)
        linear mode: signal_t = clip(prediction_t / scale, -1, 1)

    Parameters:
        predictions: np.ndarray, shape (T,)
            One-step-ahead return predictions from run_inference().
        transfer_fn: str, default "sign"
            Transfer function, one of {"sign", "linear"}.
        scale: float or None, keyword-only
            Required for "linear" mode. Normalization factor used in
            prediction_t / scale before clipping to [-1, 1].

    Returns:
        signals: np.ndarray, shape (T,)
            Trading signals. In {-1, 0, +1} for "sign", in [-1, 1] for "linear".

    Raises:
        ValueError: If predictions is empty/non-1D, transfer_fn is unknown, or
            linear mode is selected with invalid scale.
    """
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 1 or predictions.size == 0:
        raise ValueError("predictions must be a non-empty 1D array")

    if transfer_fn == "sign":
        return np.sign(predictions)

    if transfer_fn == "linear":
        if scale is None or not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("linear transfer requires a positive finite scale")
        return np.clip(predictions / float(scale), -1.0, 1.0)

    raise ValueError("transfer_fn must be either 'sign' or 'linear'")


def states_to_signal(
    state_probs: NDArray[np.floating],
    mu: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Convert state posterior probabilities to trading signals via weighted vote.

    Implements:
        signal_t = sum_k omega_{t,k} * sign(mu_k)

    where omega_{t,k} = p(m_t = k | y_{1:t}) are filtered state probabilities
    from online inference and mu_k are emission means (Paper §7).

    Parameters:
        state_probs: np.ndarray, shape (T, K)
            Filtered state probabilities from run_inference().
        mu: np.ndarray, shape (K,)
            Emission means for each hidden state.

    Returns:
        signals: np.ndarray, shape (T,)
            Continuous trading signals in [-1, 1].

    Raises:
        ValueError: If inputs are empty, shapes are inconsistent, or state_probs
            rows are not valid probability vectors.
    """
    state_probs = np.asarray(state_probs, dtype=float)
    mu = np.asarray(mu, dtype=float)

    if state_probs.ndim != 2 or state_probs.size == 0:
        raise ValueError("state_probs must be a non-empty 2D array")
    if mu.ndim != 1 or mu.size == 0:
        raise ValueError("mu must be a non-empty 1D array")
    if state_probs.shape[1] != mu.size:
        raise ValueError("state_probs and mu have inconsistent shapes")
    if np.any(state_probs < 0.0):
        raise ValueError("state_probs entries must be non-negative")
    if not np.allclose(state_probs.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("each row of state_probs must sum to 1")

    signals = state_probs @ np.sign(mu)
    return np.clip(signals, -1.0, 1.0)


def apply_no_trade_zone(
    signals: NDArray[np.floating],
    state_probs: NDArray[np.floating],
    neutral_idx: int,
    threshold: float,
) -> NDArray[np.floating]:
    """
    Zero out trading signals when the neutral-state posterior exceeds a threshold.

    Implements:
        s'_t = s_t * I(omega_{t, neutral} < threshold)

    where omega_{t, neutral} = p(m_t = neutral | y_{1:t}) is the filtered
    posterior probability of the neutral state, and I(.) is the indicator
    function. When the model is highly uncertain (neutral posterior above
    threshold), the strategy goes flat instead of trading on a weak signal.

    Parameters:
        signals: np.ndarray, shape (T,)
            Raw trading signals in [-1, 1] from states_to_signal or
            predictions_to_signal.
        state_probs: np.ndarray, shape (T, K)
            Filtered state probabilities from run_inference().
        neutral_idx: int
            Column index of the neutral state in state_probs (typically 1
            for a 3-state model sorted by ascending mu).
        threshold: float
            Neutral posterior probability above which the signal is zeroed.
            Must be in (0, 1].

    Returns:
        filtered_signals: np.ndarray, shape (T,)
            Trading signals with no-trade zone applied. Same as input where
            the neutral posterior is below threshold, zero otherwise.

    Raises:
        ValueError: If inputs are empty, shapes are inconsistent, threshold
            is out of range, or neutral_idx is out of bounds.
    """
    signals = np.asarray(signals, dtype=float)
    state_probs = np.asarray(state_probs, dtype=float)

    if signals.ndim != 1 or signals.size == 0:
        raise ValueError("signals must be a non-empty 1D array")
    if state_probs.ndim != 2 or state_probs.size == 0:
        raise ValueError("state_probs must be a non-empty 2D array")
    if signals.shape[0] != state_probs.shape[0]:
        raise ValueError("signals and state_probs must have the same length")
    if not (0.0 < threshold <= 1.0):
        raise ValueError("threshold must be in (0, 1]")
    if not (0 <= neutral_idx < state_probs.shape[1]):
        raise ValueError("neutral_idx is out of bounds for state_probs")

    neutral_posterior = state_probs[:, neutral_idx]
    mask = neutral_posterior < threshold
    return signals * mask


def smooth_signal(signals: NDArray[np.floating], alpha: float) -> NDArray[np.floating]:
    """
    Apply exponential moving average (EMA) smoothing to trading signals.

    Implements the recursion:
        s'_1     = s_1
        s'_t     = alpha * s_t + (1 - alpha) * s'_{t-1}     for t >= 2

    where alpha in (0, 1] controls the smoothing strength. Small alpha
    produces heavy smoothing (slow to react), alpha = 1 returns the
    original signal (no smoothing).

    The output is clipped to [-1, 1] to maintain valid signal bounds.

    Parameters:
        signals: np.ndarray, shape (T,)
            Raw trading signals in [-1, 1].
        alpha: float
            Smoothing factor in (0, 1]. alpha=1 means no smoothing,
            alpha close to 0 means heavy smoothing.

    Returns:
        smoothed: np.ndarray, shape (T,)
            EMA-smoothed trading signals, clipped to [-1, 1].

    Raises:
        ValueError: If signals is empty/non-1D, or alpha is out of range.
    """
    signals = np.asarray(signals, dtype=float)

    if signals.ndim != 1 or signals.size == 0:
        raise ValueError("signals must be a non-empty 1D array")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")

    smoothed = np.empty_like(signals)
    smoothed[0] = signals[0]
    for t in range(1, signals.size):
        smoothed[t] = alpha * signals[t] + (1.0 - alpha) * smoothed[t - 1]

    return np.clip(smoothed, -1.0, 1.0)
