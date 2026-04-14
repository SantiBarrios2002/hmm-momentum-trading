"""Tests for rolling-window daily retraining (Paper §2.3, §7)."""

import numpy as np
import pytest

from src.hmm.rolling import rolling_hmm, split_by_day


# ── Dummy train/inference functions for unit testing ─────────────────────────


def _dummy_train(obs, K, **kwargs):
    """Dummy Baum-Welch that returns fixed params based on obs statistics."""
    mu = np.linspace(obs.min(), obs.max(), K)
    sigma2 = np.full(K, max(np.var(obs), 1e-8))
    A = np.full((K, K), 1.0 / K)
    pi = np.full(K, 1.0 / K)
    ll = float(np.sum(obs))  # fake LL
    return {"A": A, "pi": pi, "mu": mu, "sigma2": sigma2}, [ll]


def _dummy_inference(obs, A, pi, mu, sigma2):
    """Dummy inference that returns constant predictions and uniform posteriors."""
    T = len(obs)
    K = len(mu)
    predictions = np.full(T, np.mean(mu))
    state_probs = np.full((T, K), 1.0 / K)
    return predictions, state_probs


# ── split_by_day ─────────────────────────────────────────────────────────────


class TestSplitByDay:
    def test_basic_split(self):
        returns = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        day_idx = np.array([0, 0, 1, 1, 1])
        result = split_by_day(returns, day_idx)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], [1.0, 2.0])
        np.testing.assert_array_equal(result[1], [3.0, 4.0, 5.0])

    def test_single_day(self):
        returns = np.array([1.0, 2.0, 3.0])
        day_idx = np.array([0, 0, 0])
        result = split_by_day(returns, day_idx)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0])

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            split_by_day(np.array([1.0, 2.0]), np.array([0]))

    def test_preserves_order(self):
        """Days should be sorted by index."""
        returns = np.array([10.0, 20.0, 30.0, 40.0])
        day_idx = np.array([2, 2, 0, 0])
        result = split_by_day(returns, day_idx)
        assert len(result) == 2
        # Day 0 comes first (sorted), then day 2
        np.testing.assert_array_equal(result[0], [30.0, 40.0])
        np.testing.assert_array_equal(result[1], [10.0, 20.0])


# ── rolling_hmm ──────────────────────────────────────────────────────────────


class TestRollingHmm:
    def _make_daily_returns(self, n_days=20, bars_per_day=10, seed=42):
        """Generate synthetic daily return arrays."""
        rng = np.random.default_rng(seed)
        return [rng.normal(0, 0.01, size=bars_per_day) for _ in range(n_days)]

    def test_basic_output_shape(self):
        """Check output shapes are correct."""
        daily = self._make_daily_returns(n_days=15, bars_per_day=10)
        K, H = 2, 5
        result = rolling_hmm(daily, K, H, _dummy_train, _dummy_inference)

        n_test_days = 15 - H  # = 10
        expected_T = n_test_days * 10  # 10 bars/day * 10 test days

        assert result["predictions"].shape == (expected_T,)
        assert result["state_probs"].shape == (expected_T, K)
        assert len(result["daily_params"]) == n_test_days
        assert len(result["daily_ll"]) == n_test_days
        assert result["n_train_days"] == H
        assert result["n_test_days"] == n_test_days

    def test_predictions_are_finite(self):
        daily = self._make_daily_returns(n_days=12, bars_per_day=20)
        result = rolling_hmm(daily, 2, 5, _dummy_train, _dummy_inference)
        assert np.all(np.isfinite(result["predictions"]))
        assert np.all(np.isfinite(result["state_probs"]))

    def test_state_probs_sum_to_one(self):
        daily = self._make_daily_returns(n_days=12, bars_per_day=20)
        result = rolling_hmm(daily, 3, 5, _dummy_train, _dummy_inference)
        row_sums = result["state_probs"].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_daily_params_are_dicts(self):
        daily = self._make_daily_returns(n_days=10, bars_per_day=5)
        result = rolling_hmm(daily, 2, 3, _dummy_train, _dummy_inference)
        for p in result["daily_params"]:
            assert "A" in p
            assert "pi" in p
            assert "mu" in p
            assert "sigma2" in p

    def test_h_equals_total_minus_one(self):
        """H = D-1 means only 1 test day."""
        daily = self._make_daily_returns(n_days=6, bars_per_day=10)
        result = rolling_hmm(daily, 2, 5, _dummy_train, _dummy_inference)
        assert result["n_test_days"] == 1
        assert result["predictions"].shape == (10,)

    def test_invalid_H_zero(self):
        daily = self._make_daily_returns(n_days=5)
        with pytest.raises(ValueError, match="H must be >= 1"):
            rolling_hmm(daily, 2, 0, _dummy_train, _dummy_inference)

    def test_invalid_H_too_large(self):
        daily = self._make_daily_returns(n_days=5)
        with pytest.raises(ValueError, match="H.*must be less than total days"):
            rolling_hmm(daily, 2, 5, _dummy_train, _dummy_inference)

    def test_invalid_K(self):
        daily = self._make_daily_returns(n_days=5)
        with pytest.raises(ValueError, match="K must be >= 1"):
            rolling_hmm(daily, 0, 2, _dummy_train, _dummy_inference)

    def test_different_H_changes_results(self):
        """Different window sizes should produce different parameter histories."""
        daily = self._make_daily_returns(n_days=20, bars_per_day=10)
        r1 = rolling_hmm(daily, 2, 5, _dummy_train, _dummy_inference)
        r2 = rolling_hmm(daily, 2, 10, _dummy_train, _dummy_inference)
        # Different number of test days
        assert r1["n_test_days"] != r2["n_test_days"]

    def test_seed_varies_per_day(self):
        """Each day should use a different random seed for training."""
        seeds_used = []

        def tracking_train(obs, K, **kwargs):
            seeds_used.append(kwargs.get("random_state"))
            return _dummy_train(obs, K, **kwargs)

        daily = self._make_daily_returns(n_days=8, bars_per_day=5)
        rolling_hmm(daily, 2, 3, tracking_train, _dummy_inference)
        # Seeds should be unique (one per test day)
        assert len(set(seeds_used)) == len(seeds_used)
