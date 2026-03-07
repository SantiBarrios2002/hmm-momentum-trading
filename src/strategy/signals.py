"""Signal generation utilities for HMM-based trading strategies."""

import numpy as np


def predictions_to_signal(predictions, transfer_fn="sign", *, scale=None):
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


def states_to_signal(state_probs, mu):
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
