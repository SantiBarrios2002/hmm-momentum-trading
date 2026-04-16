"""Tests for Piecewise Linear Regression — Default HMM (Paper §3.1)."""

import numpy as np
import pytest

from src.hmm.plr import (
    _fit_ols,
    durbin_watson,
    piecewise_linear_regression,
    plr_to_hmm_params,
)


# ── _fit_ols ──────────────────────────────────────────────────────────────────


class TestFitOLS:
    def test_perfect_line(self):
        """OLS on y = 2 + 3t should recover exact slope and intercept."""
        t = np.arange(100, dtype=np.float64)
        y = 2.0 + 3.0 * t
        slope, intercept, residuals = _fit_ols(y)
        assert slope == pytest.approx(3.0, abs=np.finfo(float).eps * 100)
        assert intercept == pytest.approx(2.0, abs=np.finfo(float).eps * 100)
        np.testing.assert_allclose(residuals, 0.0, atol=np.finfo(float).eps * 300)

    def test_noisy_line(self):
        """OLS on noisy data should recover slope within noise-derived tolerance."""
        rng = np.random.default_rng(42)
        n = 1000
        true_slope = 0.5
        sigma = 0.1
        t = np.arange(n, dtype=np.float64)
        y = 1.0 + true_slope * t + rng.normal(0, sigma, size=n)
        slope, intercept, residuals = _fit_ols(y)
        # Standard error of slope: sigma / sqrt(Σ(t-t̄)²) ≈ sigma / (n * sqrt(1/12))
        # For n=1000: SE ≈ 0.1 / (1000 * 0.289) ≈ 3.5e-4
        se_slope = sigma / np.sqrt(np.sum((t - t.mean()) ** 2))
        assert slope == pytest.approx(true_slope, abs=4 * se_slope)
        assert np.mean(residuals) == pytest.approx(0.0, abs=sigma / np.sqrt(n) * 4)

    def test_single_point(self):
        """Single observation: slope=0, intercept=y."""
        slope, intercept, residuals = _fit_ols(np.array([5.0]))
        assert slope == 0.0
        assert intercept == 5.0
        assert residuals[0] == 0.0

    def test_two_points(self):
        """Two points define a line exactly."""
        y = np.array([1.0, 3.0])
        slope, intercept, residuals = _fit_ols(y)
        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(1.0)
        np.testing.assert_allclose(residuals, 0.0, atol=np.finfo(float).eps * 10)


# ── piecewise_linear_regression ───────────────────────────────────────────────


class TestPiecewiseLinearRegression:
    def test_detects_known_change_point(self):
        """Two clear linear segments should yield one change point."""
        n = 200
        t = np.arange(n, dtype=np.float64)
        # Segment 1: uptrend, segment 2: downtrend
        prices = np.where(t < 100, 100.0 + 0.5 * t, 150.0 - 0.3 * (t - 100))
        result = piecewise_linear_regression(prices, min_segment_length=30, significance=0.05)
        assert result["n_segments"] >= 2
        # Change point should be near t=100
        cps = result["change_points"]
        assert len(cps) >= 1
        assert any(abs(cp - 100) < 20 for cp in cps)

    def test_recovers_segment_slopes(self):
        """Per-segment slopes should approximate the known trend gradients."""
        rng = np.random.default_rng(42)
        n1, n2 = 150, 150
        sigma = 0.05
        # Segment 1: slope +0.1, segment 2: slope -0.2
        t1 = np.arange(n1, dtype=np.float64)
        t2 = np.arange(n2, dtype=np.float64)
        seg1 = 100.0 + 0.1 * t1 + rng.normal(0, sigma, n1)
        seg2 = seg1[-1] - 0.2 * t2 + rng.normal(0, sigma, n2)
        prices = np.concatenate([seg1, seg2])

        result = piecewise_linear_regression(prices, min_segment_length=30, significance=0.05)
        # Should detect at least one change point
        assert result["n_segments"] >= 2
        slopes = [s["slope"] for s in result["segments"]]
        # At least one segment should have positive slope near 0.1
        assert any(abs(s - 0.1) < 0.05 for s in slopes)
        # At least one segment should have negative slope near -0.2
        assert any(abs(s - (-0.2)) < 0.05 for s in slopes)

    def test_residual_variance_recovery(self):
        """Residual variance should approximate known noise variance."""
        rng = np.random.default_rng(123)
        n = 500
        sigma = 0.1
        t = np.arange(n, dtype=np.float64)
        prices = 50.0 + 0.05 * t + rng.normal(0, sigma, n)

        result = piecewise_linear_regression(prices, min_segment_length=100, significance=0.01)
        # With low significance and single trend, should be 1 segment
        for seg in result["segments"]:
            # MLE variance should be near sigma²=0.01
            # Tolerance: ~2*sigma²/sqrt(n) for chi² variance
            n_seg = seg["end"] - seg["start"]
            tol = 2 * sigma ** 2 / np.sqrt(n_seg)
            assert seg["residual_variance"] == pytest.approx(sigma ** 2, abs=tol)

    def test_no_change_point_in_pure_trend(self):
        """Single linear trend with tight significance should give 1 segment."""
        rng = np.random.default_rng(42)
        n = 300
        sigma = 0.01
        t = np.arange(n, dtype=np.float64)
        prices = 100.0 + 0.01 * t + rng.normal(0, sigma, n)

        result = piecewise_linear_regression(prices, min_segment_length=50, significance=0.001)
        assert result["n_segments"] == 1
        assert result["change_points"] == []

    def test_multiple_change_points(self):
        """Three distinct trends should yield at least 2 change points."""
        n = 300
        t = np.arange(n, dtype=np.float64)
        prices = np.piecewise(
            t,
            [t < 100, (t >= 100) & (t < 200), t >= 200],
            [lambda t: 100.0 + 0.5 * t,
             lambda t: 150.0 - 0.3 * (t - 100),
             lambda t: 120.0 + 0.2 * (t - 200)],
        )
        result = piecewise_linear_regression(prices, min_segment_length=30, significance=0.05)
        assert result["n_segments"] >= 3

    def test_segments_cover_full_series(self):
        """Segments should partition the entire price series without gaps."""
        rng = np.random.default_rng(42)
        prices = np.cumsum(rng.normal(0, 1, 200)) + 100
        result = piecewise_linear_regression(prices, min_segment_length=30, significance=0.05)
        # Check no gaps
        assert result["segments"][0]["start"] == 0
        assert result["segments"][-1]["end"] == len(prices)
        for i in range(len(result["segments"]) - 1):
            assert result["segments"][i]["end"] == result["segments"][i + 1]["start"]

    def test_invalid_prices_too_short(self):
        with pytest.raises(ValueError, match="at least 2"):
            piecewise_linear_regression(np.array([1.0]))

    def test_invalid_min_segment_length(self):
        with pytest.raises(ValueError, match="min_segment_length"):
            piecewise_linear_regression(np.array([1.0, 2.0, 3.0]), min_segment_length=1)

    def test_invalid_significance(self):
        with pytest.raises(ValueError, match="significance"):
            piecewise_linear_regression(np.array([1.0, 2.0, 3.0]), significance=0.0)
        with pytest.raises(ValueError, match="significance"):
            piecewise_linear_regression(np.array([1.0, 2.0, 3.0]), significance=1.0)


# ── plr_to_hmm_params ────────────────────────────────────────────────────────


class TestPlrToHmmParams:
    def _make_plr_result(self, slopes, variances, lengths=None):
        """Helper: build a minimal PLR result dict."""
        n = len(slopes)
        if lengths is None:
            lengths = [100] * n
        segments = []
        start = 0
        for i in range(n):
            segments.append({
                "start": start,
                "end": start + lengths[i],
                "slope": slopes[i],
                "intercept": 0.0,
                "residual_variance": variances[i],
                "residuals": np.zeros(lengths[i]),
            })
            start += lengths[i]
        return {
            "change_points": list(range(1, n)),
            "n_segments": n,
            "segments": segments,
        }

    def test_sticky_transition_matrix(self):
        """A should be sticky with β on diagonal, (1-β)/(K-1) off-diagonal."""
        plr = self._make_plr_result([0.1, -0.1], [0.01, 0.02])
        beta = 0.5
        params = plr_to_hmm_params(plr, K=2, beta=beta)
        A = params["A"]
        assert A.shape == (2, 2)
        np.testing.assert_allclose(A[0, 0], beta)
        np.testing.assert_allclose(A[1, 1], beta)
        np.testing.assert_allclose(A[0, 1], (1 - beta))
        np.testing.assert_allclose(A[1, 0], (1 - beta))
        # Rows sum to 1
        np.testing.assert_allclose(A.sum(axis=1), 1.0)

    def test_sticky_matrix_k3(self):
        """K=3 sticky matrix: diagonal=β, off-diagonal=(1-β)/2."""
        plr = self._make_plr_result([-0.2, 0.0, 0.3], [0.01, 0.01, 0.01])
        beta = 0.6
        params = plr_to_hmm_params(plr, K=3, beta=beta)
        A = params["A"]
        assert A.shape == (3, 3)
        for k in range(3):
            assert A[k, k] == pytest.approx(beta)
            for j in range(3):
                if j != k:
                    assert A[k, j] == pytest.approx((1 - beta) / 2)
        np.testing.assert_allclose(A.sum(axis=1), 1.0)

    def test_pi_is_uniform(self):
        """Ergodic distribution of symmetric sticky matrix is uniform."""
        plr = self._make_plr_result([0.1, -0.1], [0.01, 0.01])
        params = plr_to_hmm_params(plr, K=2)
        np.testing.assert_allclose(params["pi"], [0.5, 0.5])

    def test_mu_sorted_ascending(self):
        """Emission means should be sorted ascending."""
        plr = self._make_plr_result([0.3, -0.1, 0.0], [0.01, 0.02, 0.01])
        params = plr_to_hmm_params(plr, K=3)
        assert list(params["mu"]) == sorted(params["mu"])

    def test_k1_degenerate(self):
        """K=1: single state, A=[[1]], pi=[1]."""
        plr = self._make_plr_result([0.05], [0.01])
        params = plr_to_hmm_params(plr, K=1)
        assert params["A"].shape == (1, 1)
        assert params["A"][0, 0] == pytest.approx(1.0)
        assert params["pi"][0] == pytest.approx(1.0)
        assert len(params["mu"]) == 1
        assert len(params["sigma2"]) == 1

    def test_more_segments_than_K(self):
        """If PLR gives more segments than K, should cluster down to K."""
        plr = self._make_plr_result(
            [-0.3, -0.1, 0.0, 0.1, 0.4],
            [0.01, 0.02, 0.01, 0.02, 0.01],
        )
        params = plr_to_hmm_params(plr, K=2)
        assert len(params["mu"]) == 2
        assert len(params["sigma2"]) == 2

    def test_fewer_segments_than_K(self):
        """If PLR gives 1 segment but K=3, should expand."""
        plr = self._make_plr_result([0.05], [0.01])
        params = plr_to_hmm_params(plr, K=3)
        assert len(params["mu"]) == 3
        assert len(params["sigma2"]) == 3
        # States should be distinct
        assert len(set(params["mu"])) == 3

    def test_compatible_with_inference(self):
        """Output params should have correct shapes for run_inference_numba."""
        plr = self._make_plr_result([-0.1, 0.0, 0.2], [0.01, 0.01, 0.01])
        params = plr_to_hmm_params(plr, K=3)
        assert params["A"].shape == (3, 3)
        assert params["pi"].shape == (3,)
        assert params["mu"].shape == (3,)
        assert params["sigma2"].shape == (3,)
        assert np.all(params["sigma2"] > 0)

    def test_invalid_K(self):
        plr = self._make_plr_result([0.1], [0.01])
        with pytest.raises(ValueError, match="K must be >= 1"):
            plr_to_hmm_params(plr, K=0)

    def test_invalid_beta(self):
        plr = self._make_plr_result([0.1], [0.01])
        with pytest.raises(ValueError, match="beta"):
            plr_to_hmm_params(plr, K=2, beta=0.0)
        with pytest.raises(ValueError, match="beta"):
            plr_to_hmm_params(plr, K=2, beta=1.0)


# ── durbin_watson ─────────────────────────────────────────────────────────────


class TestDurbinWatson:
    def test_no_autocorrelation(self):
        """White noise residuals should give DW ≈ 2."""
        rng = np.random.default_rng(42)
        residuals = rng.normal(0, 1, size=1000)
        dw = durbin_watson(residuals)
        # For n=1000 white noise, DW ≈ 2 with std ≈ 2/sqrt(n) ≈ 0.063
        assert dw == pytest.approx(2.0, abs=4 * 2.0 / np.sqrt(1000))

    def test_strong_positive_autocorrelation(self):
        """AR(1) with φ=0.9 should give DW << 2."""
        rng = np.random.default_rng(42)
        n = 500
        residuals = np.empty(n)
        residuals[0] = rng.normal()
        for t in range(1, n):
            residuals[t] = 0.9 * residuals[t - 1] + rng.normal() * 0.1
        dw = durbin_watson(residuals)
        # DW ≈ 2(1 - φ) = 2(1 - 0.9) = 0.2
        assert dw < 1.0

    def test_negative_autocorrelation(self):
        """Alternating residuals should give DW > 2."""
        residuals = np.array([1.0, -1.0] * 50, dtype=np.float64)
        dw = durbin_watson(residuals)
        # Perfectly alternating: DW = 4.0
        assert dw == pytest.approx(4.0, abs=0.1)

    def test_perfect_fit(self):
        """Zero residuals should return DW=2 (no autocorrelation to detect)."""
        assert durbin_watson(np.zeros(10)) == 2.0

    def test_too_few_residuals(self):
        with pytest.raises(ValueError, match="at least 2"):
            durbin_watson(np.array([1.0]))
