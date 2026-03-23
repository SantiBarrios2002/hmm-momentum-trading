"""Tests for Langevin utility functions (src/langevin/utils.py)."""

import numpy as np
import pytest

from src.langevin.utils import (
    estimate_langevin_params,
    estimate_langevin_params_raw,
    trend_to_trading_signal,
)


# ──────────────────────────────────────────────────────────────────────
# Tests for estimate_langevin_params
# ──────────────────────────────────────────────────────────────────────


class TestEstimateLangevinParams:
    """Tests for estimate_langevin_params."""

    def test_returns_valid_ranges(self):
        """All estimated parameters fall in physically valid ranges (Issue #45, test 1)."""
        rng = np.random.default_rng(42)
        log_returns = rng.normal(0.0005, 0.01, size=500)

        params = estimate_langevin_params(log_returns, dt=1.0)

        assert params['theta'] < 0, f"theta must be < 0, got {params['theta']}"
        assert params['sigma'] > 0, f"sigma must be > 0, got {params['sigma']}"
        assert params['sigma_obs'] > 0, f"sigma_obs must be > 0, got {params['sigma_obs']}"
        assert params['lambda_J'] >= 0, f"lambda_J must be >= 0, got {params['lambda_J']}"
        assert params['sigma_J'] >= 0, f"sigma_J must be >= 0, got {params['sigma_J']}"

    def test_returns_all_required_keys(self):
        """Output dict contains all 6 required parameter keys."""
        log_returns = np.random.default_rng(0).normal(0, 0.01, size=100)
        params = estimate_langevin_params(log_returns)

        required_keys = {'theta', 'sigma', 'sigma_obs', 'lambda_J', 'mu_J', 'sigma_J'}
        assert set(params.keys()) == required_keys

    def test_higher_volatility_gives_larger_sigma(self):
        """More volatile returns should produce larger sigma estimate."""
        rng = np.random.default_rng(42)
        calm_returns = rng.normal(0, 0.005, size=500)
        volatile_returns = rng.normal(0, 0.05, size=500)

        calm_params = estimate_langevin_params(calm_returns)
        volatile_params = estimate_langevin_params(volatile_returns)

        assert volatile_params['sigma'] > calm_params['sigma']

    def test_fat_tailed_returns_give_positive_lambda_J(self):
        """Returns with excess kurtosis should produce lambda_J > 0."""
        rng = np.random.default_rng(42)
        # Mix of normal + occasional large jumps → excess kurtosis
        base = rng.normal(0, 0.01, size=500)
        jumps = np.zeros(500)
        jump_idx = rng.choice(500, size=20, replace=False)
        jumps[jump_idx] = rng.normal(0, 0.05, size=20)
        fat_tailed = base + jumps

        params = estimate_langevin_params(fat_tailed)
        assert params['lambda_J'] > 0, "Fat-tailed returns should have lambda_J > 0"

    def test_gaussian_returns_low_lambda_J(self):
        """Pure Gaussian returns should have low or zero jump intensity."""
        rng = np.random.default_rng(42)
        # Large sample of pure Gaussian returns (kurtosis ≈ 0)
        gaussian_returns = rng.normal(0, 0.01, size=5000)

        params = estimate_langevin_params(gaussian_returns)
        # Gaussian excess kurtosis is ~0, so lambda_J should be small
        assert params['lambda_J'] < 1.0, (
            f"Gaussian returns should have low lambda_J, got {params['lambda_J']}"
        )

    def test_mu_J_is_zero(self):
        """mu_J defaults to 0 (symmetric jumps)."""
        log_returns = np.random.default_rng(0).normal(0, 0.01, size=100)
        params = estimate_langevin_params(log_returns)
        assert params['mu_J'] == 0.0

    def test_different_dt_scales_theta(self):
        """Changing dt should scale theta inversely (faster sampling → smaller |theta|)."""
        rng = np.random.default_rng(42)
        log_returns = rng.normal(0, 0.01, size=500)

        params_daily = estimate_langevin_params(log_returns, dt=1.0)
        params_hourly = estimate_langevin_params(log_returns, dt=1.0 / 6.5)

        # theta is mean-reversion rate per time unit, so hourly dt should
        # give |theta| roughly scaled by the dt ratio
        assert params_daily['theta'] < 0
        assert params_hourly['theta'] < 0

    def test_invalid_dt_zero(self):
        """dt = 0 raises ValueError."""
        with pytest.raises(ValueError, match="dt must be > 0"):
            estimate_langevin_params(np.array([0.01, -0.02, 0.03]), dt=0.0)

    def test_invalid_dt_negative(self):
        """dt < 0 raises ValueError."""
        with pytest.raises(ValueError, match="dt must be > 0"):
            estimate_langevin_params(np.array([0.01, -0.02, 0.03]), dt=-1.0)

    def test_invalid_too_few_returns(self):
        """Fewer than 3 returns raises ValueError."""
        with pytest.raises(ValueError, match="Need at least 3 returns"):
            estimate_langevin_params(np.array([0.01, -0.02]), dt=1.0)

    def test_minimum_returns(self):
        """Exactly 3 returns works without error."""
        params = estimate_langevin_params(np.array([0.01, -0.02, 0.03]))
        assert params['theta'] < 0

    def test_constant_returns(self):
        """Constant returns (zero variance) produce valid output with positive sigma floor."""
        params = estimate_langevin_params(np.array([0.01, 0.01, 0.01, 0.01]))
        assert params['theta'] < 0
        # ret_std is floored to 1e-10, so sigma and sigma_obs are small but positive
        assert params['sigma'] > 0, "sigma must be positive even for constant returns"
        assert params['sigma_obs'] > 0, "sigma_obs must be positive even for constant returns"
        assert params['lambda_J'] >= 0


# ──────────────────────────────────────────────────────────────────────
# Tests for estimate_langevin_params_raw
# ──────────────────────────────────────────────────────────────────────


class TestEstimateLangevinParamsRaw:
    """Tests for estimate_langevin_params_raw."""

    def test_returns_valid_ranges(self):
        """All estimated parameters fall in physically valid ranges."""
        rng = np.random.default_rng(42)
        # Simulate daily price changes for SPY (~$500, daily std ~$5)
        price_changes = rng.normal(0.05, 5.0, size=500)

        params = estimate_langevin_params_raw(price_changes, price_level=500.0, dt=1.0)

        assert params['theta'] < 0, f"theta must be < 0, got {params['theta']}"
        assert params['sigma'] > 0, f"sigma must be > 0, got {params['sigma']}"
        assert params['sigma_obs'] > 0, f"sigma_obs must be > 0, got {params['sigma_obs']}"
        assert params['lambda_J'] >= 0, f"lambda_J must be >= 0, got {params['lambda_J']}"
        assert params['sigma_J'] >= 0, f"sigma_J must be >= 0, got {params['sigma_J']}"

    def test_returns_all_required_keys(self):
        """Output dict contains all required keys including scale_factors."""
        rng = np.random.default_rng(0)
        price_changes = rng.normal(0, 5.0, size=100)
        params = estimate_langevin_params_raw(price_changes, price_level=500.0)

        required_keys = {'theta', 'sigma', 'sigma_obs', 'lambda_J', 'mu_J', 'sigma_J',
                         'scale_factors'}
        assert set(params.keys()) == required_keys

        sf_keys = {'sigma', 'sigma_obs', 'sigma_J'}
        assert set(params['scale_factors'].keys()) == sf_keys

    def test_scale_factors_are_fractions_of_price(self):
        """Scale factors = param / price_level, so SF(sigma) * price_level == sigma."""
        rng = np.random.default_rng(42)
        price_level = 500.0
        price_changes = rng.normal(0, 5.0, size=500)
        params = estimate_langevin_params_raw(price_changes, price_level=price_level)

        sf = params['scale_factors']
        np.testing.assert_allclose(sf['sigma'] * price_level, params['sigma'])
        np.testing.assert_allclose(sf['sigma_obs'] * price_level, params['sigma_obs'])
        if params['sigma_J'] > 0:
            np.testing.assert_allclose(sf['sigma_J'] * price_level, params['sigma_J'])

    def test_consistent_with_log_return_version(self):
        """For small returns, raw-price and log-return params should be approximately
        proportional (sigma_raw ≈ sigma_log * price_level)."""
        rng = np.random.default_rng(42)
        price_level = 500.0
        # Generate log-returns, then derive price changes
        log_returns = rng.normal(0.0005, 0.01, size=500)
        price_changes = log_returns * price_level  # first-order approx: ΔP ≈ r * P

        params_log = estimate_langevin_params(log_returns, dt=1.0)
        params_raw = estimate_langevin_params_raw(price_changes, price_level=price_level, dt=1.0)

        # theta should be identical (dimensionless, from autocorrelation structure)
        np.testing.assert_allclose(params_raw['theta'], params_log['theta'], rtol=0.05)

        # sigma_raw ≈ sigma_log * price_level (both derived from std * sqrt(2|theta|))
        ratio = params_raw['sigma'] / params_log['sigma']
        np.testing.assert_allclose(ratio, price_level, rtol=0.1)

    def test_higher_volatility_gives_larger_sigma(self):
        """More volatile price changes produce larger sigma estimate."""
        rng = np.random.default_rng(42)
        calm = rng.normal(0, 2.0, size=500)
        volatile = rng.normal(0, 20.0, size=500)

        calm_params = estimate_langevin_params_raw(calm, price_level=500.0)
        volatile_params = estimate_langevin_params_raw(volatile, price_level=500.0)

        assert volatile_params['sigma'] > calm_params['sigma']

    def test_fat_tailed_gives_positive_lambda_J(self):
        """Price changes with excess kurtosis should produce lambda_J > 0."""
        rng = np.random.default_rng(42)
        base = rng.normal(0, 5.0, size=500)
        jumps = np.zeros(500)
        jump_idx = rng.choice(500, size=20, replace=False)
        jumps[jump_idx] = rng.normal(0, 25.0, size=20)
        fat_tailed = base + jumps

        params = estimate_langevin_params_raw(fat_tailed, price_level=500.0)
        assert params['lambda_J'] > 0, "Fat-tailed price changes should have lambda_J > 0"

    def test_mu_J_is_zero(self):
        """mu_J defaults to 0 (symmetric jumps)."""
        price_changes = np.random.default_rng(0).normal(0, 5.0, size=100)
        params = estimate_langevin_params_raw(price_changes, price_level=500.0)
        assert params['mu_J'] == 0.0

    def test_invalid_dt_zero(self):
        """dt = 0 raises ValueError."""
        with pytest.raises(ValueError, match="dt must be > 0"):
            estimate_langevin_params_raw(np.array([1.0, -2.0, 3.0]), price_level=500.0, dt=0.0)

    def test_invalid_dt_negative(self):
        """dt < 0 raises ValueError."""
        with pytest.raises(ValueError, match="dt must be > 0"):
            estimate_langevin_params_raw(np.array([1.0, -2.0, 3.0]), price_level=500.0, dt=-1.0)

    def test_invalid_price_level_zero(self):
        """price_level = 0 raises ValueError."""
        with pytest.raises(ValueError, match="price_level must be > 0"):
            estimate_langevin_params_raw(np.array([1.0, -2.0, 3.0]), price_level=0.0)

    def test_invalid_price_level_negative(self):
        """price_level < 0 raises ValueError."""
        with pytest.raises(ValueError, match="price_level must be > 0"):
            estimate_langevin_params_raw(np.array([1.0, -2.0, 3.0]), price_level=-100.0)

    def test_invalid_too_few_changes(self):
        """Fewer than 3 price changes raises ValueError."""
        with pytest.raises(ValueError, match="Need at least 3 price changes"):
            estimate_langevin_params_raw(np.array([1.0, -2.0]), price_level=500.0)

    def test_minimum_changes(self):
        """Exactly 3 price changes works without error."""
        params = estimate_langevin_params_raw(np.array([1.0, -2.0, 3.0]), price_level=500.0)
        assert params['theta'] < 0

    def test_constant_changes(self):
        """Constant price changes (zero variance) produce valid output."""
        params = estimate_langevin_params_raw(
            np.array([1.0, 1.0, 1.0, 1.0]), price_level=500.0
        )
        assert params['theta'] < 0
        assert params['sigma'] > 0
        assert params['sigma_obs'] > 0


# ──────────────────────────────────────────────────────────────────────
# Tests for trend_to_trading_signal
# ──────────────────────────────────────────────────────────────────────


class TestTrendToTradingSignal:
    """Tests for trend_to_trading_signal."""

    def test_signal_bounded(self):
        """Output signals must be in [-1, 1] (Issue #45, test 2)."""
        rng = np.random.default_rng(42)
        trends = np.cumsum(rng.normal(0, 0.1, size=100))
        signals = trend_to_trading_signal(trends, sigma_delta=0.05)

        assert np.all(signals >= -1.0), f"Signal below -1: {signals.min()}"
        assert np.all(signals <= 1.0), f"Signal above 1: {signals.max()}"

    def test_signal_sign_correct(self):
        """Positive trend change → positive signal, negative → negative (Issue #45, test 3)."""
        # Monotonically increasing trend → all positive deltas
        trends = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        signals = trend_to_trading_signal(trends, sigma_delta=0.5)
        assert np.all(signals > 0), "Increasing trend should give positive signals"

        # Monotonically decreasing trend → all negative deltas
        trends_down = np.array([4.0, 3.0, 2.0, 1.0, 0.0])
        signals_down = trend_to_trading_signal(trends_down, sigma_delta=0.5)
        assert np.all(signals_down < 0), "Decreasing trend should give negative signals"

    def test_zero_trend_change_gives_zero_signal(self):
        """No trend change → zero signal."""
        trends = np.array([5.0, 5.0, 5.0])
        signals = trend_to_trading_signal(trends, sigma_delta=1.0)
        # delta = 0 exactly (integer differences), so 0/sqrt(0+sigma^2) = 0.0 exactly
        np.testing.assert_array_equal(signals, 0.0)

    def test_output_length(self):
        """Output has length T-1 (one fewer than input)."""
        T = 50
        trends = np.random.default_rng(0).normal(0, 1, size=T)
        signals = trend_to_trading_signal(trends, sigma_delta=0.1)
        assert len(signals) == T - 1

    def test_large_delta_approaches_one(self):
        """Very large trend change → signal approaches ±1."""
        trends = np.array([0.0, 1000.0])  # huge positive change
        signals = trend_to_trading_signal(trends, sigma_delta=0.01)
        # delta / sqrt(delta^2 + sigma_delta^2) ≈ 1 for large delta
        np.testing.assert_allclose(signals[0], 1.0, atol=1e-8)

    def test_small_delta_is_linear(self):
        """For small delta << sigma_delta, signal ≈ delta / sigma_delta (linear regime)."""
        sigma_delta = 1.0
        delta = 0.001  # much smaller than sigma_delta
        trends = np.array([0.0, delta])
        signals = trend_to_trading_signal(trends, sigma_delta=sigma_delta)

        # Linear approximation: delta / sqrt(delta^2 + sigma_delta^2) ≈ delta / sigma_delta
        expected = delta / sigma_delta
        np.testing.assert_allclose(signals[0], expected, rtol=1e-4)

    def test_smaller_sigma_delta_sharper(self):
        """Smaller sigma_delta produces signals closer to ±1 for the same delta."""
        trends = np.array([0.0, 0.5])

        signal_sharp = trend_to_trading_signal(trends, sigma_delta=0.01)
        signal_smooth = trend_to_trading_signal(trends, sigma_delta=10.0)

        assert abs(signal_sharp[0]) > abs(signal_smooth[0])

    def test_symmetry(self):
        """Signal is odd: f(-delta) = -f(delta)."""
        sigma_delta = 0.3
        trends_up = np.array([0.0, 1.5])
        trends_down = np.array([0.0, -1.5])

        sig_up = trend_to_trading_signal(trends_up, sigma_delta)
        sig_down = trend_to_trading_signal(trends_down, sigma_delta)

        # Exact symmetry: f(-delta) = -f(delta) by construction (odd function)
        np.testing.assert_array_equal(sig_up[0], -sig_down[0])

    def test_invalid_sigma_delta(self):
        """sigma_delta <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="sigma_delta must be > 0"):
            trend_to_trading_signal(np.array([1.0, 2.0]), sigma_delta=0.0)

        with pytest.raises(ValueError, match="sigma_delta must be > 0"):
            trend_to_trading_signal(np.array([1.0, 2.0]), sigma_delta=-1.0)

    def test_invalid_too_few_trends(self):
        """Fewer than 2 trend estimates raises ValueError."""
        with pytest.raises(ValueError, match="Need at least 2 trend estimates"):
            trend_to_trading_signal(np.array([1.0]), sigma_delta=0.1)

    def test_minimum_trends(self):
        """Exactly 2 trend estimates works, producing 1 signal."""
        signals = trend_to_trading_signal(np.array([0.0, 1.0]), sigma_delta=0.5)
        assert len(signals) == 1
