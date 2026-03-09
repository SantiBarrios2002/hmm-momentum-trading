"""End-to-end integration tests for the full HMM trading pipeline.

These tests verify that the complete pipeline works correctly on synthetic
data where ground truth is known: generate observations from a known HMM,
train via Baum-Welch, decode via Viterbi, run online inference, generate
signals, and backtest.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.hmm.baum_welch import baum_welch
from src.hmm.forward import forward
from src.hmm.inference import run_inference
from src.hmm.utils import sort_states
from src.hmm.viterbi import viterbi
from src.strategy.backtest import backtest
from src.strategy.signals import apply_no_trade_zone, smooth_signal, states_to_signal


def _generate_hmm_data(
    T: int,
    A: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    sigma2: np.ndarray,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic observations and states from a known HMM."""
    rng = np.random.default_rng(seed)
    K = mu.size
    states = np.empty(T, dtype=int)
    observations = np.empty(T, dtype=float)

    states[0] = rng.choice(K, p=pi)
    observations[0] = rng.normal(mu[states[0]], np.sqrt(sigma2[states[0]]))

    for t in range(1, T):
        states[t] = rng.choice(K, p=A[states[t - 1]])
        observations[t] = rng.normal(mu[states[t]], np.sqrt(sigma2[states[t]]))

    return observations, states


# Well-separated 3-state HMM for testing
TRUE_A = np.array([
    [0.90, 0.07, 0.03],
    [0.05, 0.85, 0.10],
    [0.03, 0.07, 0.90],
])
TRUE_PI = np.array([0.2, 0.5, 0.3])
TRUE_MU = np.array([-0.02, 0.0, 0.02])
TRUE_SIGMA2 = np.array([0.0004, 0.0001, 0.0002])


class TestE2ETrainDecodeBacktest:
    """Test 1: Full pipeline from synthetic data to backtest metrics."""

    def test_full_pipeline_produces_valid_backtest(self):
        """Generate data → train EM → Viterbi → online inference → backtest."""
        observations, true_states = _generate_hmm_data(
            T=500, A=TRUE_A, pi=TRUE_PI, mu=TRUE_MU, sigma2=TRUE_SIGMA2, seed=42
        )

        # Train
        params, history, gamma = baum_welch(
            observations, K=3, max_iter=100, tol=1e-6, n_restarts=3, random_state=99
        )
        params = sort_states(params)

        # Log-likelihood must increase monotonically
        for i in range(1, len(history)):
            assert history[i] >= history[i - 1] - 1e-10

        # Viterbi decode
        states, log_prob = viterbi(
            observations, params["A"], params["pi"], params["mu"], params["sigma2"]
        )
        assert states.shape == (500,)
        assert set(np.unique(states)).issubset({0, 1, 2})

        # Online inference
        predictions, state_probs = run_inference(
            observations, params["A"], params["pi"], params["mu"], params["sigma2"]
        )
        assert predictions.shape == (500,)
        assert state_probs.shape == (500, 3)
        np.testing.assert_allclose(state_probs.sum(axis=1), 1.0, atol=1e-10)

        # Signal generation
        signals = states_to_signal(state_probs, params["mu"])
        assert signals.shape == (500,)
        assert np.all(signals >= -1.0) and np.all(signals <= 1.0)

        # Backtest
        result = backtest(observations, signals, transaction_cost_bps=5)
        assert "net_returns" in result
        assert "cumulative" in result
        assert "metrics" in result
        assert result["net_returns"].shape == (500,)
        assert result["cumulative"].shape == (500,)
        assert np.isfinite(result["metrics"]["sharpe"])
        assert result["metrics"]["max_drawdown"] >= 0.0

    def test_recovered_means_close_to_truth(self):
        """EM should recover emission means within reasonable tolerance."""
        observations, _ = _generate_hmm_data(
            T=2000, A=TRUE_A, pi=TRUE_PI, mu=TRUE_MU, sigma2=TRUE_SIGMA2, seed=7
        )

        params, _, _ = baum_welch(
            observations, K=3, max_iter=200, tol=1e-8, n_restarts=5, random_state=123
        )
        params = sort_states(params)

        # Sorted means should be close to true sorted means
        recovered_mu = np.sort(params["mu"])
        true_mu_sorted = np.sort(TRUE_MU)
        np.testing.assert_allclose(recovered_mu, true_mu_sorted, atol=0.005)


class TestE2ESignalRefinement:
    """Test 2: Pipeline with signal refinement (no-trade zone + EMA)."""

    def test_refinement_reduces_turnover(self):
        """No-trade zone and EMA smoothing should reduce signal turnover."""
        observations, _ = _generate_hmm_data(
            T=500, A=TRUE_A, pi=TRUE_PI, mu=TRUE_MU, sigma2=TRUE_SIGMA2, seed=55
        )

        params, _, _ = baum_welch(
            observations, K=3, max_iter=100, tol=1e-6, n_restarts=3, random_state=77
        )
        params = sort_states(params)

        _, state_probs = run_inference(
            observations, params["A"], params["pi"], params["mu"], params["sigma2"]
        )
        raw_signals = states_to_signal(state_probs, params["mu"])

        # Apply no-trade zone
        filtered = apply_no_trade_zone(
            raw_signals, state_probs, neutral_idx=1, threshold=0.6
        )
        # Apply EMA smoothing
        smoothed = smooth_signal(filtered, alpha=0.3)

        # Backtest both
        raw_result = backtest(observations, raw_signals, transaction_cost_bps=5)
        refined_result = backtest(observations, smoothed, transaction_cost_bps=5)

        # Refined signal should have lower turnover
        assert refined_result["metrics"]["turnover"] < raw_result["metrics"]["turnover"]

        # Both backtests should produce valid outputs
        assert np.all(np.isfinite(raw_result["net_returns"]))
        assert np.all(np.isfinite(refined_result["net_returns"]))


class TestE2EForwardConsistency:
    """Test 3: Cross-check online inference against batch forward algorithm."""

    def test_online_matches_batch_forward_likelihood(self):
        """Online filtered probabilities should be consistent with batch forward."""
        observations, _ = _generate_hmm_data(
            T=200, A=TRUE_A, pi=TRUE_PI, mu=TRUE_MU, sigma2=TRUE_SIGMA2, seed=13
        )

        params, _, _ = baum_welch(
            observations, K=3, max_iter=100, tol=1e-6, n_restarts=3, random_state=42
        )
        params = sort_states(params)
        A, pi, mu, sigma2 = params["A"], params["pi"], params["mu"], params["sigma2"]

        # Batch forward
        log_alpha, batch_ll = forward(observations, A, pi, mu, sigma2)

        # Online inference
        _, state_probs_online = run_inference(observations, A, pi, mu, sigma2)

        # Batch forward gives alpha_t(k) = p(y_1:t, m_t=k)
        # Normalizing each row gives p(m_t | y_1:t) = filtered posteriors
        from scipy.special import logsumexp

        log_filtered = log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
        batch_filtered = np.exp(log_filtered)

        # Online filtered posteriors should match batch-derived posteriors
        np.testing.assert_allclose(
            state_probs_online, batch_filtered, atol=1e-6,
            err_msg="Online inference should match batch forward filtered posteriors"
        )
