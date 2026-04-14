"""Tests for return discretization to tick-size grid (Paper §2.1)."""

import numpy as np
import pytest

from src.hmm.discretize import (
    discretize_returns,
    discretized_log_gaussian,
    tick_variance_floor,
)


# ── discretize_returns ───────────────────────────────────────────────────────


class TestDiscretizeReturns:
    """Tests for discretize_returns."""

    def test_exact_multiples_unchanged(self):
        """Returns already on the grid should not change."""
        tick = 0.25
        x = np.array([-0.50, -0.25, 0.0, 0.25, 0.50])
        result = discretize_returns(x, tick)
        np.testing.assert_array_equal(result, x)

    def test_midpoints_round_to_nearest(self):
        """Values exactly between two grid points follow banker's rounding."""
        tick = 0.25
        x = np.array([0.125, 0.375])  # midpoints
        result = discretize_returns(x, tick)
        # np.round uses banker's rounding: 0.5 -> 0 (even), 1.5 -> 2 (even)
        # 0.125/0.25 = 0.5 -> rounds to 0.0; 0.375/0.25 = 1.5 -> rounds to 2*0.25=0.5
        assert result[0] == 0.0
        assert result[1] == 0.5

    def test_off_grid_values_round_correctly(self):
        """Arbitrary values round to nearest grid point."""
        tick = 0.25
        x = np.array([0.10, 0.20, -0.10, -0.30])
        result = discretize_returns(x, tick)
        expected = np.array([0.0, 0.25, 0.0, -0.25])
        np.testing.assert_array_almost_equal(result, expected)

    def test_output_is_exact_multiples(self):
        """All output values must be exact multiples of tick_size."""
        tick = 0.25
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, size=1000)
        result = discretize_returns(x, tick)
        remainders = np.abs(result / tick - np.round(result / tick))
        # Tolerance: float64 round-trip through division/multiplication
        # introduces error of order eps * max(|x|/tick) ≈ eps * 4/0.25 = 16*eps
        np.testing.assert_allclose(remainders, 0.0, atol=np.finfo(float).eps * 100)

    def test_invalid_tick_size(self):
        """Negative or zero tick size should raise."""
        with pytest.raises(ValueError, match="tick_size must be positive"):
            discretize_returns(np.array([1.0]), 0.0)
        with pytest.raises(ValueError, match="tick_size must be positive"):
            discretize_returns(np.array([1.0]), -0.1)

    def test_empty_array(self):
        """Empty input returns empty output."""
        result = discretize_returns(np.array([]), 0.25)
        assert len(result) == 0

    def test_single_value(self):
        """Single observation works."""
        result = discretize_returns(np.array([0.13]), 0.25)
        assert result[0] == 0.25


# ── tick_variance_floor ──────────────────────────────────────────────────────


class TestTickVarianceFloor:
    """Tests for tick_variance_floor."""

    def test_es_tick_size(self):
        """ES tick = 0.25 → min_var = 0.25²/2 = 0.03125."""
        assert tick_variance_floor(0.25) == pytest.approx(0.03125)

    def test_unit_tick(self):
        """Tick = 1.0 → min_var = 0.5."""
        assert tick_variance_floor(1.0) == pytest.approx(0.5)

    def test_small_tick(self):
        """Tick = 0.01 → min_var = 0.00005."""
        assert tick_variance_floor(0.01) == pytest.approx(0.00005)

    def test_invalid_tick(self):
        """Non-positive tick size should raise."""
        with pytest.raises(ValueError):
            tick_variance_floor(0.0)
        with pytest.raises(ValueError):
            tick_variance_floor(-1.0)


# ── discretized_log_gaussian ─────────────────────────────────────────────────


class TestDiscretizedLogGaussian:
    """Tests for discretized_log_gaussian."""

    def test_pmf_sums_to_one(self):
        """The PMF over the full grid should sum to 1.0."""
        tick = 0.25
        mu, sigma2 = 0.0, 1.0
        sigma = np.sqrt(sigma2)

        # Build grid covering ±10σ
        half = 10 * sigma
        grid = np.arange(-half, half + tick / 2, tick)

        log_pmf = discretized_log_gaussian(grid, mu, sigma2, tick)
        total = np.sum(np.exp(log_pmf))
        # Grid truncation at ±10σ leaves ~2e-23 probability mass outside;
        # dominant error is float64 summation over ~80 terms: O(n * eps) ≈ 2e-14
        np.testing.assert_allclose(total, 1.0, atol=len(grid) * np.finfo(float).eps * 10)

    def test_approaches_continuous_for_small_tick(self):
        """As tick → 0, discretized log-PDF → continuous log-PDF (up to a constant)."""
        mu, sigma2 = 0.5, 0.1
        x = np.array([0.3, 0.5, 0.7])

        # Continuous log-PDF
        log_pdf_cont = -0.5 * np.log(2 * np.pi * sigma2) - 0.5 * (x - mu) ** 2 / sigma2

        # Very small tick → normalization constant ≈ 1/tick (Riemann sum)
        tick = 0.001
        log_pmf_disc = discretized_log_gaussian(x, mu, sigma2, tick)

        # The difference should be approximately log(tick) for small tick
        # because PMF ≈ PDF * tick, so log(PMF) ≈ log(PDF) + log(tick)
        # Riemann sum error is O(tick² / sigma²) ≈ 0.001² / 0.1 = 1e-5
        diff = log_pmf_disc - log_pdf_cont
        riemann_error = tick ** 2 / sigma2
        np.testing.assert_allclose(diff, np.log(tick), atol=riemann_error * 10)

    def test_mode_at_mu(self):
        """The PMF should be maximized at the grid point nearest to mu."""
        tick = 0.25
        mu, sigma2 = 0.0, 0.5
        grid = np.arange(-5, 5 + tick / 2, tick)
        log_pmf = discretized_log_gaussian(grid, mu, sigma2, tick)
        mode_idx = np.argmax(log_pmf)
        assert grid[mode_idx] == pytest.approx(0.0, abs=tick)

    def test_symmetry(self):
        """For mu=0, the PMF should be symmetric."""
        tick = 0.25
        mu, sigma2 = 0.0, 1.0
        x_pos = np.array([0.25, 0.50, 1.00])
        x_neg = np.array([-0.25, -0.50, -1.00])
        log_pmf_pos = discretized_log_gaussian(x_pos, mu, sigma2, tick)
        log_pmf_neg = discretized_log_gaussian(x_neg, mu, sigma2, tick)
        # Symmetry is exact up to float64 precision; grid is symmetric around mu=0
        np.testing.assert_allclose(log_pmf_pos, log_pmf_neg, atol=np.finfo(float).eps * 100)

    def test_invalid_sigma2(self):
        """Non-positive variance should raise."""
        with pytest.raises(ValueError, match="sigma2 must be positive"):
            discretized_log_gaussian(np.array([0.0]), 0.0, 0.0, 0.25)
        with pytest.raises(ValueError, match="sigma2 must be positive"):
            discretized_log_gaussian(np.array([0.0]), 0.0, -1.0, 0.25)

    def test_invalid_tick_size(self):
        """Non-positive tick size should raise."""
        with pytest.raises(ValueError, match="tick_size must be positive"):
            discretized_log_gaussian(np.array([0.0]), 0.0, 1.0, 0.0)

    def test_single_observation(self):
        """Single observation should produce a single log-PMF value."""
        result = discretized_log_gaussian(np.array([0.0]), 0.0, 1.0, 0.25)
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_values_are_negative(self):
        """Log-PMF values should all be negative (probabilities < 1)."""
        tick = 0.25
        x = np.array([-1.0, 0.0, 1.0])
        log_pmf = discretized_log_gaussian(x, 0.0, 1.0, tick)
        assert np.all(log_pmf < 0)
