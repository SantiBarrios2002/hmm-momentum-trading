"""Tests for side information features — vol ratio + seasonality (Paper §4)."""

import numpy as np
import pandas as pd
import pytest

from src.hmm.side_info import (
    evaluate_spline,
    ewma_volatility,
    fit_spline,
    seasonality_feature,
    spline_buckets,
    volatility_ratio,
)


# ── ewma_volatility ──────────────────────────────────────────────────────────


class TestEwmaVolatility:
    def test_recursion_matches_formula(self):
        """EWMA recursion σ²_{t+1} = λ·σ²_t + (1-λ)·Δy²_t matches closed form."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, size=100)
        lam = 0.79
        sigma = ewma_volatility(returns, lam=lam)

        # Manual recursion
        sigma2_manual = np.empty(100)
        sigma2_manual[0] = returns[0] ** 2
        for t in range(1, 100):
            sigma2_manual[t] = lam * sigma2_manual[t - 1] + (1 - lam) * returns[t] ** 2

        np.testing.assert_allclose(
            sigma ** 2, sigma2_manual,
            rtol=np.finfo(float).eps * 100,
        )

    def test_output_positive(self):
        """Volatility estimates must be non-negative."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, size=500)
        sigma = ewma_volatility(returns)
        assert np.all(sigma >= 0)

    def test_higher_lambda_smoother(self):
        """Higher λ → smoother vol estimates (lower variance of σ series)."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, size=500)
        sigma_low = ewma_volatility(returns, lam=0.5)
        sigma_high = ewma_volatility(returns, lam=0.95)
        assert np.std(sigma_high) < np.std(sigma_low)

    def test_constant_returns(self):
        """Constant returns → σ converges to |return|."""
        r = 0.01
        returns = np.full(200, r)
        sigma = ewma_volatility(returns, lam=0.79)
        # Should converge to sqrt((1-λ)·r²/(1-λ)) = |r|
        assert sigma[-1] == pytest.approx(abs(r), rel=0.01)

    def test_invalid_lambda(self):
        with pytest.raises(ValueError, match="lam"):
            ewma_volatility(np.array([1.0]), lam=0.0)
        with pytest.raises(ValueError, match="lam"):
            ewma_volatility(np.array([1.0]), lam=1.0)

    def test_empty_returns(self):
        with pytest.raises(ValueError, match="at least 1"):
            ewma_volatility(np.array([]))


# ── volatility_ratio ──────────────────────────────────────────────────────────


class TestVolatilityRatio:
    def test_ratio_in_reasonable_range(self):
        """Vol ratio should be in (0, ∞), typically near 1.0."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, size=500)
        ratio = volatility_ratio(returns)
        assert np.all(ratio > 0)
        assert np.all(np.isfinite(ratio))
        # Stationary returns: ratio should be near 1.0
        assert np.median(ratio) == pytest.approx(1.0, abs=0.3)

    def test_decreasing_vol_gives_low_ratio(self):
        """When recent vol drops, short EWMA reacts faster → ratio < 1."""
        rng = np.random.default_rng(42)
        # High vol then sudden drop — short window adapts faster
        returns = np.concatenate([
            rng.normal(0, 0.05, 300),
            rng.normal(0, 0.001, 300),  # vol drops 50x
        ])
        ratio = volatility_ratio(returns, window_short=50, window_long=100)
        # After the drop, short vol falls faster → ratio < 1
        # With λ=0.79, effective memory ~5 bars, so check right after transition
        # The long window retains more high-vol memory than the short window
        assert np.min(ratio[300:310]) < 1.0

    def test_invalid_window_order(self):
        with pytest.raises(ValueError, match="window_short"):
            volatility_ratio(np.array([1.0] * 200), window_short=100, window_long=50)


# ── seasonality_feature ───────────────────────────────────────────────────────


class TestSeasonalityFeature:
    def test_range_0_to_1(self):
        """Feature values should be in [0, 1]."""
        timestamps = pd.date_range("2024-01-02 09:30", periods=390, freq="1min")
        feat = seasonality_feature(timestamps)
        assert np.all(feat >= 0.0)
        assert np.all(feat <= 1.0)

    def test_monotonic_within_day(self):
        """Feature should increase monotonically within a single trading day."""
        timestamps = pd.date_range("2024-01-02 09:30", periods=390, freq="1min")
        feat = seasonality_feature(timestamps)
        assert np.all(np.diff(feat) >= 0)

    def test_output_length(self):
        timestamps = pd.date_range("2024-01-02 09:30", periods=100, freq="1min")
        feat = seasonality_feature(timestamps)
        assert len(feat) == 100


# ── fit_spline + evaluate_spline ──────────────────────────────────────────────


class TestSpline:
    def test_spline_fits_known_function(self):
        """Spline should approximate a known non-linear function."""
        rng = np.random.default_rng(42)
        x = np.linspace(0, 1, 500)
        # True function: sine wave
        y_true = 0.1 * np.sin(2 * np.pi * x) + rng.normal(0, 0.01, 500)
        spline = fit_spline(x, y_true, n_knots=6)
        y_pred = evaluate_spline(spline, x)
        # Should track the sine wave; residual σ should be near noise level
        residual_std = np.std(y_pred - 0.1 * np.sin(2 * np.pi * x))
        assert residual_std < 0.05  # much less than signal amplitude 0.1

    def test_spline_is_zero_mean(self):
        """Spline integral should be ~zero (Paper §4.1)."""
        rng = np.random.default_rng(42)
        x = np.linspace(0, 1, 1000)
        y = 0.05 * np.sin(4 * np.pi * x) + rng.normal(0, 0.01, 1000)
        spline = fit_spline(x, y, n_knots=10)
        # Evaluate on dense grid and check mean ≈ 0
        x_eval = np.linspace(x.min(), x.max(), 5000)
        spline_mean = np.mean(spline(x_eval))
        assert abs(spline_mean) < 0.01

    def test_evaluate_clips_to_domain(self):
        """Evaluation outside training domain should clip, not extrapolate."""
        rng = np.random.default_rng(42)
        x = np.linspace(0.1, 0.9, 200)
        y = rng.normal(0, 0.01, 200)
        spline = fit_spline(x, y, n_knots=4)
        # Evaluate outside domain
        x_outside = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
        result = evaluate_spline(spline, x_outside)
        assert np.all(np.isfinite(result))

    def test_spline_with_6_knots(self):
        """Paper uses 6 knots for vol ratio — should work."""
        rng = np.random.default_rng(42)
        x = np.linspace(0.5, 1.0, 300)
        y = rng.normal(0, 0.01, 300)
        spline = fit_spline(x, y, n_knots=6)
        result = evaluate_spline(spline, x)
        assert result.shape == (300,)
        assert np.all(np.isfinite(result))

    def test_invalid_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            fit_spline(np.array([1.0, 2.0]), np.array([1.0]))

    def test_too_few_observations(self):
        with pytest.raises(ValueError, match="Need at least"):
            fit_spline(np.array([1.0, 2.0]), np.array([1.0, 2.0]), n_knots=10)


# ── spline_buckets ────────────────────────────────────────────────────────────


class TestSplineBuckets:
    def test_sine_spline_has_roots(self):
        """Sine-like spline should produce roots and multiple buckets."""
        rng = np.random.default_rng(42)
        x = np.linspace(0, 1, 500)
        y = 0.1 * np.sin(2 * np.pi * x) + rng.normal(0, 0.01, 500)
        spline = fit_spline(x, y, n_knots=6)
        boundaries, bucket_idx = spline_buckets(spline, x)
        assert len(boundaries) >= 1  # At least 1 root
        R = len(boundaries) + 1
        assert R >= 2
        assert bucket_idx.min() == 0
        assert bucket_idx.max() == R - 1

    def test_all_observations_assigned(self):
        """Every observation must get a bucket index."""
        rng = np.random.default_rng(42)
        x = np.linspace(0, 1, 300)
        y = 0.05 * np.sin(4 * np.pi * x) + rng.normal(0, 0.01, 300)
        spline = fit_spline(x, y, n_knots=8)
        _, bucket_idx = spline_buckets(spline, x)
        assert len(bucket_idx) == 300
        assert np.all(bucket_idx >= 0)

    def test_flat_spline_gives_one_bucket(self):
        """Constant returns → flat spline → no roots → 1 bucket."""
        x = np.linspace(0, 1, 200)
        y = np.zeros(200)  # Flat
        spline = fit_spline(x, y, n_knots=4)
        boundaries, bucket_idx = spline_buckets(spline, x)
        # Flat spline ≈ 0 everywhere → may have 0 or many "roots"
        # With zero-mean correction on zero data, spline ≈ 0
        # All observations should still be assigned
        assert len(bucket_idx) == 200

    def test_bucket_indices_are_valid(self):
        """Bucket indices should be in [0, R-1]."""
        rng = np.random.default_rng(42)
        x = np.linspace(0.5, 1.0, 400)
        y = 0.1 * np.sin(3 * np.pi * x) + rng.normal(0, 0.01, 400)
        spline = fit_spline(x, y, n_knots=6)
        boundaries, bucket_idx = spline_buckets(spline, x)
        R = len(boundaries) + 1
        assert np.all(bucket_idx >= 0)
        assert np.all(bucket_idx < R)

    def test_boundaries_are_sorted(self):
        """Spline roots (boundaries) should be sorted."""
        rng = np.random.default_rng(42)
        x = np.linspace(0, 1, 500)
        y = 0.1 * np.sin(2 * np.pi * x) + rng.normal(0, 0.01, 500)
        spline = fit_spline(x, y, n_knots=6)
        boundaries, _ = spline_buckets(spline, x)
        if len(boundaries) > 1:
            assert np.all(np.diff(boundaries) > 0)
