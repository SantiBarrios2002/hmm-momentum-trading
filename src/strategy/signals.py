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
