"""Plotting helpers for regime analysis and strategy evaluation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray


def plot_regime_colored_prices(
    prices: NDArray[np.floating],
    regimes: NDArray[np.integer],
    ax: Axes | None = None,
    title: str = "Price by Regime",
) -> tuple[Figure, Axes]:
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


def plot_cumulative_returns(
    strategy_cumulative: NDArray[np.floating],
    benchmark_cumulative: NDArray[np.floating] | None = None,
    ax: Axes | None = None,
    title: str = "Cumulative Returns",
) -> tuple[Figure, Axes]:
    """
    Plot cumulative return curves for strategy and optional benchmark.

    Parameters:
        strategy_cumulative: 1D array-like strategy cumulative curve.
        benchmark_cumulative: Optional 1D array-like benchmark cumulative curve.
        ax: Optional matplotlib Axes to draw on.
        title: Figure title.

    Returns:
        (fig, ax): matplotlib Figure and Axes.

    Raises:
        ValueError: If strategy is not 1D/non-empty, or benchmark length differs.
    """
    strategy = np.asarray(strategy_cumulative, dtype=float)
    if strategy.ndim != 1:
        raise ValueError("strategy_cumulative must be a 1D array-like input")
    if strategy.size == 0:
        raise ValueError("strategy_cumulative must contain at least 1 observation")

    benchmark = None
    if benchmark_cumulative is not None:
        benchmark = np.asarray(benchmark_cumulative, dtype=float)
        if benchmark.ndim != 1:
            raise ValueError("benchmark_cumulative must be a 1D array-like input")
        if benchmark.size != strategy.size:
            raise ValueError("benchmark_cumulative must match strategy length")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    x = np.arange(strategy.size)
    ax.plot(x, strategy, label="Strategy", linewidth=2)
    if benchmark is not None:
        ax.plot(x, benchmark, label="Benchmark", linewidth=2, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Value")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax
