"""Plotting helpers for regime analysis and strategy evaluation."""

import matplotlib.pyplot as plt
import numpy as np


def plot_regime_colored_prices(prices, regimes, ax=None, title="Price by Regime"):
    """
    Plot a price series with points colored by inferred regime.

    Parameters:
        prices: 1D array-like of price values.
        regimes: 1D array-like of integer regime labels.
        ax: Optional matplotlib Axes to draw on.
        title: Figure title.

    Returns:
        (fig, ax): matplotlib Figure and Axes.

    Raises:
        ValueError: If inputs are not 1D, empty, or length-mismatched.
    """
    price_values = np.asarray(prices, dtype=float)
    regime_values = np.asarray(regimes)

    if price_values.ndim != 1 or regime_values.ndim != 1:
        raise ValueError("prices and regimes must be 1D array-like inputs")
    if price_values.size == 0:
        raise ValueError("prices must contain at least 1 observation")
    if price_values.size != regime_values.size:
        raise ValueError("prices and regimes must have the same length")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    x = np.arange(price_values.size)
    ax.plot(x, price_values, color="black", linewidth=1.0, alpha=0.7, label="Price")
    scatter = ax.scatter(
        x,
        price_values,
        c=regime_values,
        cmap="tab10",
        s=20,
        alpha=0.9,
    )
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, label="Regime")
    fig.tight_layout()
    return fig, ax
