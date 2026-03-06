import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from src.utils.plotting import plot_cumulative_returns, plot_regime_colored_prices


def test_plot_regime_colored_prices_returns_figure_and_axes():
    prices = [100.0, 101.0, 99.5, 102.0]
    regimes = [0, 1, 1, 0]

    fig, ax = plot_regime_colored_prices(prices, regimes)

    assert fig is not None
    assert ax is not None
    assert len(ax.lines) == 1
    assert len(ax.collections) >= 1
    plt.close(fig)


def test_plot_regime_colored_prices_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        plot_regime_colored_prices([100.0, 101.0], [0])


def test_plot_cumulative_returns_with_benchmark():
    strategy = [1.0, 1.02, 1.01, 1.05]
    benchmark = [1.0, 1.01, 1.00, 1.03]

    fig, ax = plot_cumulative_returns(strategy, benchmark)

    assert fig is not None
    assert ax is not None
    assert len(ax.lines) == 2
    plt.close(fig)


def test_plot_cumulative_returns_rejects_length_mismatch():
    with pytest.raises(ValueError, match="match strategy length"):
        plot_cumulative_returns([1.0, 1.1], [1.0])
