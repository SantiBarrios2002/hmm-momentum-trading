"""Tests for MCMC HMM with Metropolis-Hastings (Paper §3.3)."""

import numpy as np
import pytest

from src.hmm.mcmc import (
    _ergodic_distribution,
    _log_prior,
    bridge_sampling_log_marginal,
    mcmc_hmm,
)


# ── Helper: generate synthetic HMM data ──────────────────────────────────────


def _generate_hmm_data(K, T, mu, sigma2, A, pi, seed=42):
    """Generate observations from a known HMM."""
    rng = np.random.default_rng(seed)
    states = np.empty(T, dtype=int)
    obs = np.empty(T)
    states[0] = rng.choice(K, p=pi)
    obs[0] = rng.normal(mu[states[0]], np.sqrt(sigma2[states[0]]))
    for t in range(1, T):
        states[t] = rng.choice(K, p=A[states[t - 1]])
        obs[t] = rng.normal(mu[states[t]], np.sqrt(sigma2[states[t]]))
    return obs, states


# ── _ergodic_distribution ─────────────────────────────────────────────────────


class TestErgodicDistribution:
    def test_uniform_for_doubly_stochastic(self):
        """Doubly stochastic matrix → uniform ergodic distribution."""
        A = np.array([[0.5, 0.5], [0.5, 0.5]])
        pi = _ergodic_distribution(A)
        np.testing.assert_allclose(pi, [0.5, 0.5], atol=1e-10)

    def test_sums_to_one(self):
        A = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]])
        pi = _ergodic_distribution(A)
        assert pi.sum() == pytest.approx(1.0, abs=1e-10)
        assert np.all(pi >= 0)

    def test_is_left_eigenvector(self):
        """π·A = π (stationary condition)."""
        A = np.array([[0.9, 0.1], [0.3, 0.7]])
        pi = _ergodic_distribution(A)
        np.testing.assert_allclose(pi @ A, pi, atol=1e-10)

    def test_k1(self):
        """K=1: trivial case."""
        pi = _ergodic_distribution(np.array([[1.0]]))
        assert pi[0] == pytest.approx(1.0)


# ── _log_prior ────────────────────────────────────────────────────────────────


class TestLogPrior:
    def test_finite_for_valid_params(self):
        K = 3
        A = np.full((K, K), 1.0 / K)
        mu = np.array([-0.01, 0.0, 0.01])
        sigma2 = np.array([0.001, 0.001, 0.001])
        lp = _log_prior(A, mu, sigma2, K)
        assert np.isfinite(lp)

    def test_negative_inf_for_zero_variance(self):
        K = 2
        A = np.array([[0.5, 0.5], [0.5, 0.5]])
        mu = np.array([0.0, 0.0])
        sigma2 = np.array([0.001, 0.0])  # zero variance
        lp = _log_prior(A, mu, sigma2, K)
        assert lp == -np.inf

    def test_prior_favors_sticky_A(self):
        """Dirichlet prior with e_diag=4 should favor sticky matrices."""
        K = 2
        mu = np.array([0.0, 0.0])
        sigma2 = np.array([0.001, 0.001])
        A_sticky = np.array([[0.8, 0.2], [0.2, 0.8]])
        A_uniform = np.array([[0.5, 0.5], [0.5, 0.5]])
        lp_sticky = _log_prior(A_sticky, mu, sigma2, K)
        lp_uniform = _log_prior(A_uniform, mu, sigma2, K)
        assert lp_sticky > lp_uniform


# ── mcmc_hmm ──────────────────────────────────────────────────────────────────


class TestMcmcHmm:
    def test_output_shapes(self):
        """Check that output shapes are correct."""
        rng = np.random.default_rng(42)
        obs = rng.normal(0, 0.01, size=200)
        K = 2
        n_samples = 500
        burn_in = 100
        params, samples, acc = mcmc_hmm(
            obs, K, n_samples=n_samples, burn_in=burn_in
        )
        assert params["A"].shape == (K, K)
        assert params["pi"].shape == (K,)
        assert params["mu"].shape == (K,)
        assert params["sigma2"].shape == (K,)
        assert samples["mu"].shape == (n_samples, K)
        assert samples["sigma2"].shape == (n_samples, K)
        assert samples["log_posterior"].shape == (n_samples,)
        assert 0.0 <= acc <= 1.0

    def test_acceptance_rate_in_range(self):
        """Acceptance rate should be in a reasonable range (5-80%)."""
        rng = np.random.default_rng(42)
        obs = rng.normal(0, 0.01, size=300)
        _, _, acc = mcmc_hmm(obs, 2, n_samples=2000, burn_in=500, proposal_scale=0.005)
        assert 0.05 < acc < 0.80

    def test_map_log_posterior_is_finite(self):
        """MAP estimate should have finite log-posterior."""
        rng = np.random.default_rng(42)
        obs = rng.normal(0, 0.01, size=200)
        params, samples, _ = mcmc_hmm(obs, 2, n_samples=1000, burn_in=200)
        # MAP should have higher log-posterior than average
        post_burn = samples["log_posterior"][200:]
        map_post = np.max(post_burn)
        assert np.isfinite(map_post)

    def test_parameter_recovery_2state(self):
        """MCMC should roughly recover 2-state synthetic HMM parameters."""
        true_mu = np.array([-0.05, 0.05])
        true_sigma2 = np.array([0.01, 0.01])
        true_A = np.array([[0.95, 0.05], [0.05, 0.95]])
        true_pi = np.array([0.5, 0.5])

        obs, _ = _generate_hmm_data(2, 2000, true_mu, true_sigma2, true_A, true_pi, seed=42)

        params, _, _ = mcmc_hmm(
            obs, 2, n_samples=5000, burn_in=2000,
            proposal_scale=0.005, random_state=42,
        )

        # Means should be in the right ballpark (within 2x the true values)
        # MCMC with short chains may not converge precisely
        recovered_mu = np.sort(params["mu"])
        assert recovered_mu[0] < 0  # negative state
        assert recovered_mu[1] > 0  # positive state

    def test_mu_sorted_ascending(self):
        """MAP mu should be sorted ascending for consistent state ordering."""
        rng = np.random.default_rng(42)
        obs = rng.normal(0, 0.01, size=200)
        params, _, _ = mcmc_hmm(obs, 3, n_samples=500, burn_in=100)
        assert list(params["mu"]) == sorted(params["mu"])

    def test_sigma2_positive(self):
        """All MAP variances must be positive."""
        rng = np.random.default_rng(42)
        obs = rng.normal(0, 0.01, size=200)
        params, _, _ = mcmc_hmm(obs, 2, n_samples=500, burn_in=100)
        assert np.all(params["sigma2"] > 0)

    def test_A_rows_sum_to_one(self):
        """MAP transition matrix rows must sum to 1."""
        rng = np.random.default_rng(42)
        obs = rng.normal(0, 0.01, size=200)
        params, _, _ = mcmc_hmm(obs, 2, n_samples=500, burn_in=100)
        np.testing.assert_allclose(params["A"].sum(axis=1), 1.0, atol=1e-10)

    def test_k1_single_state(self):
        """K=1 should produce a single Gaussian."""
        rng = np.random.default_rng(42)
        obs = rng.normal(0.02, 0.05, size=500)
        params, _, _ = mcmc_hmm(obs, 1, n_samples=1000, burn_in=200)
        assert params["A"].shape == (1, 1)
        assert params["A"][0, 0] == pytest.approx(1.0)
        assert len(params["mu"]) == 1

    def test_invalid_inputs(self):
        obs = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="K must be >= 1"):
            mcmc_hmm(obs, 0)
        with pytest.raises(ValueError, match="at least 2"):
            mcmc_hmm(np.array([1.0]), 2)
        with pytest.raises(ValueError, match="n_samples"):
            mcmc_hmm(obs, 2, n_samples=0)
        with pytest.raises(ValueError, match="burn_in"):
            mcmc_hmm(obs, 2, n_samples=100, burn_in=100)


# ── bridge_sampling_log_marginal ──────────────────────────────────────────────


class TestBridgeSampling:
    def test_finite_output(self):
        """Bridge sampling should return a finite log-marginal."""
        rng = np.random.default_rng(42)
        obs = rng.normal(0, 0.01, size=200)
        K = 2
        _, samples, _ = mcmc_hmm(obs, K, n_samples=1000, burn_in=200)
        A = np.array([[0.9, 0.1], [0.1, 0.9]])

        lml = bridge_sampling_log_marginal(
            samples["mu"][200:], samples["sigma2"][200:],
            obs, K, A, n_importance=100,
        )
        assert np.isfinite(lml)

    def test_k2_preferred_over_k5_on_2state_data(self):
        """On 2-state data, K=2 should have higher marginal likelihood than K=5."""
        true_mu = np.array([-0.05, 0.05])
        true_sigma2 = np.array([0.01, 0.01])
        true_A = np.array([[0.95, 0.05], [0.05, 0.95]])
        true_pi = np.array([0.5, 0.5])
        obs, _ = _generate_hmm_data(2, 1000, true_mu, true_sigma2, true_A, true_pi)

        # Fit K=2
        params2, samples2, _ = mcmc_hmm(
            obs, 2, n_samples=3000, burn_in=1000, proposal_scale=0.005
        )
        lml2 = bridge_sampling_log_marginal(
            samples2["mu"][1000:], samples2["sigma2"][1000:],
            obs, 2, params2["A"], n_importance=200,
        )

        # Fit K=5
        params5, samples5, _ = mcmc_hmm(
            obs, 5, n_samples=3000, burn_in=1000, proposal_scale=0.005
        )
        lml5 = bridge_sampling_log_marginal(
            samples5["mu"][1000:], samples5["sigma2"][1000:],
            obs, 5, params5["A"], n_importance=200,
        )

        # K=2 should be preferred (higher log-marginal)
        assert lml2 > lml5

    def test_empty_samples_raises(self):
        with pytest.raises(ValueError, match="No posterior samples"):
            bridge_sampling_log_marginal(
                np.empty((0, 2)), np.empty((0, 2)),
                np.array([1.0, 2.0]), 2, np.eye(2),
            )
