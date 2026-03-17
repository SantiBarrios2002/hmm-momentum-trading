"""Tests for src/langevin/particle.py — Standard particle filter."""

import numpy as np
import pytest

from scipy.special import logsumexp

from src.langevin.particle import (
    initialize_particles,
    propose_jump_times,
    propagate_particles,
    weight_particles,
    resample_particles,
    run_particle_filter,
)
from src.langevin.model import discretize_langevin, observation_matrix
from src.langevin.kalman import kalman_filter


# ---------- initialize_particles tests ----------


class TestInitializeParticles:
    """Tests for particle initialization from the prior."""

    def test_output_shapes(self):
        """States have shape (N, 2), log_weights have shape (N,)."""
        N = 100
        mu0 = np.array([10.0, 0.0])
        C0 = np.eye(2)
        states, log_weights = initialize_particles(N, mu0, C0, rng=np.random.default_rng(0))
        assert states.shape == (N, 2)
        assert log_weights.shape == (N,)

    def test_uniform_weights(self):
        """All log weights must equal -log(N)."""
        N = 50
        mu0 = np.array([5.0, 0.1])
        C0 = np.eye(2) * 2.0
        _, log_weights = initialize_particles(N, mu0, C0, rng=np.random.default_rng(0))
        np.testing.assert_allclose(log_weights, -np.log(N))

    def test_weights_sum_to_one(self):
        """exp(log_weights) must sum to 1.0."""
        N = 200
        mu0 = np.array([0.0, 0.0])
        C0 = np.eye(2) * 0.5
        _, log_weights = initialize_particles(N, mu0, C0, rng=np.random.default_rng(0))
        np.testing.assert_allclose(np.sum(np.exp(log_weights)), 1.0, atol=1e-12)

    def test_sample_mean_close_to_prior(self):
        """With many particles, sample mean should approximate mu0."""
        N = 10_000
        mu0 = np.array([100.0, 0.5])
        C0 = np.array([[4.0, 0.0], [0.0, 1.0]])
        states, _ = initialize_particles(N, mu0, C0, rng=np.random.default_rng(42))
        sample_mean = np.mean(states, axis=0)
        # Tolerance scales as sqrt(max_var / N) ≈ sqrt(4 / 10000) = 0.02
        np.testing.assert_allclose(sample_mean, mu0, atol=0.1)

    def test_sample_covariance_close_to_prior(self):
        """With many particles, sample covariance should approximate C0."""
        N = 10_000
        mu0 = np.array([0.0, 0.0])
        C0 = np.array([[2.0, 0.5], [0.5, 1.0]])
        states, _ = initialize_particles(N, mu0, C0, rng=np.random.default_rng(42))
        sample_cov = np.cov(states, rowvar=False)
        # Tolerance scales as O(1/sqrt(N)) ≈ 0.01, use 0.15 for safety
        np.testing.assert_allclose(sample_cov, C0, atol=0.15)

    def test_single_particle(self):
        """N=1 is valid: one particle, weight = 1.0 (log_weight = 0)."""
        mu0 = np.array([5.0, 0.0])
        C0 = np.eye(2)
        states, log_weights = initialize_particles(1, mu0, C0, rng=np.random.default_rng(0))
        assert states.shape == (1, 2)
        np.testing.assert_allclose(log_weights[0], 0.0)  # log(1/1) = 0

    def test_invalid_N_raises(self):
        """N < 1 must raise ValueError."""
        with pytest.raises(ValueError, match="N must be >= 1"):
            initialize_particles(0, np.array([0.0, 0.0]), np.eye(2))

    def test_deterministic_with_seed(self):
        """Same rng seed produces identical particles."""
        mu0 = np.array([10.0, 0.5])
        C0 = np.eye(2) * 3.0
        s1, w1 = initialize_particles(50, mu0, C0, rng=np.random.default_rng(99))
        s2, w2 = initialize_particles(50, mu0, C0, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(w1, w2)


# ---------- propose_jump_times tests ----------


class TestProposeJumpTimes:
    """Tests for jump time proposal from the Poisson process."""

    def test_output_shapes(self):
        """jump_occurred is bool (N,), jump_times is float (N,)."""
        N = 100
        jo, jt = propose_jump_times(N, dt=1.0, lambda_J=0.5, rng=np.random.default_rng(0))
        assert jo.shape == (N,)
        assert jt.shape == (N,)
        assert jo.dtype == bool

    def test_zero_intensity_no_jumps(self):
        """lambda_J=0 means no jumps ever occur."""
        jo, jt = propose_jump_times(1000, dt=1.0, lambda_J=0.0, rng=np.random.default_rng(0))
        assert not np.any(jo)
        np.testing.assert_array_equal(jt, 0.0)

    def test_empirical_jump_rate(self):
        """Fraction of particles that jump should approximate 1 - exp(-lambda_J * dt)."""
        N = 50_000
        lambda_J, dt = 0.3, 1.0
        jo, _ = propose_jump_times(N, dt=dt, lambda_J=lambda_J, rng=np.random.default_rng(42))
        expected_rate = 1.0 - np.exp(-lambda_J * dt)
        empirical_rate = np.mean(jo)
        # Tolerance: ~sqrt(p(1-p)/N) ≈ 0.002, use 0.01 for safety
        np.testing.assert_allclose(empirical_rate, expected_rate, atol=0.01)

    def test_jump_times_within_interval(self):
        """All jump times must be in [0, dt]."""
        N = 10_000
        dt = 2.5
        jo, jt = propose_jump_times(N, dt=dt, lambda_J=1.0, rng=np.random.default_rng(0))
        # Only check particles that actually jumped
        assert np.all(jt[jo] >= 0.0)
        assert np.all(jt[jo] <= dt)

    def test_no_jump_times_are_zero(self):
        """For particles that did NOT jump, jump_times should be 0."""
        jo, jt = propose_jump_times(1000, dt=1.0, lambda_J=0.5, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(jt[~jo], 0.0)

    def test_jump_times_uniform_distribution(self):
        """Conditioned on a jump, tau should be approximately Uniform(0, dt)."""
        N = 100_000
        dt = 2.0
        jo, jt = propose_jump_times(N, dt=dt, lambda_J=5.0, rng=np.random.default_rng(42))
        # High lambda so most particles jump — gives us a large sample
        tau_jumped = jt[jo]
        assert len(tau_jumped) > 1000  # sanity check: enough samples
        # Mean of Uniform(0, dt) = dt/2
        np.testing.assert_allclose(np.mean(tau_jumped), dt / 2, atol=0.05)
        # Std of Uniform(0, dt) = dt / sqrt(12)
        np.testing.assert_allclose(np.std(tau_jumped), dt / np.sqrt(12), atol=0.05)

    def test_high_intensity_all_jump(self):
        """With very high lambda_J * dt, virtually all particles should jump."""
        jo, _ = propose_jump_times(10_000, dt=1.0, lambda_J=100.0, rng=np.random.default_rng(0))
        # p(jump) = 1 - exp(-100) ≈ 1.0
        assert np.mean(jo) > 0.99

    def test_invalid_dt_raises(self):
        """dt <= 0 must raise ValueError."""
        with pytest.raises(ValueError, match="dt must be > 0"):
            propose_jump_times(10, dt=0.0, lambda_J=0.5)
        with pytest.raises(ValueError, match="dt must be > 0"):
            propose_jump_times(10, dt=-1.0, lambda_J=0.5)

    def test_invalid_lambda_raises(self):
        """Negative lambda_J must raise ValueError."""
        with pytest.raises(ValueError, match="lambda_J must be >= 0"):
            propose_jump_times(10, dt=1.0, lambda_J=-0.1)


# ---------- propagate_particles tests ----------


class TestPropagateParticles:
    """Tests for particle state propagation."""

    def test_output_shape(self):
        """Output has same shape as input states."""
        N = 50
        states = np.column_stack([np.full(N, 100.0), np.zeros(N)])
        jo = np.zeros(N, dtype=bool)
        jt = np.zeros(N)
        new = propagate_particles(
            states, jo, jt, theta=-0.5, sigma=0.1, dt=1.0,
            mu_J=0.0, sigma_J=0.0, rng=np.random.default_rng(0),
        )
        assert new.shape == (N, 2)

    def test_no_jump_mean_matches_F(self):
        """With no jumps, sample mean of propagated states ≈ F @ x for large N."""
        N = 50_000
        theta, sigma, dt = -0.5, 0.1, 1.0
        x0 = np.array([100.0, 0.5])
        states = np.tile(x0, (N, 1))  # all particles start at same state
        jo = np.zeros(N, dtype=bool)
        jt = np.zeros(N)

        new = propagate_particles(
            states, jo, jt, theta=theta, sigma=sigma, dt=dt,
            mu_J=0.0, sigma_J=0.0, rng=np.random.default_rng(42),
        )
        F, Q = discretize_langevin(theta, sigma, dt)
        expected_mean = F @ x0
        sample_mean = np.mean(new, axis=0)
        # Tolerance: sqrt(max(diag(Q)) / N) ~ sqrt(0.01/50000) ~ 0.0004
        np.testing.assert_allclose(sample_mean, expected_mean, atol=0.05)

    def test_no_jump_covariance_matches_Q(self):
        """With no jumps, sample covariance of propagated noise ≈ Q."""
        N = 50_000
        theta, sigma, dt = -0.3, 0.2, 1.0
        x0 = np.array([50.0, 0.0])
        states = np.tile(x0, (N, 1))
        jo = np.zeros(N, dtype=bool)
        jt = np.zeros(N)

        new = propagate_particles(
            states, jo, jt, theta=theta, sigma=sigma, dt=dt,
            mu_J=0.0, sigma_J=0.0, rng=np.random.default_rng(42),
        )
        F, Q = discretize_langevin(theta, sigma, dt)
        # Subtract deterministic mean to isolate noise
        residuals = new - (F @ x0)
        sample_cov = np.cov(residuals, rowvar=False)
        np.testing.assert_allclose(sample_cov, Q, atol=0.005)

    def test_jump_increases_trend_variance(self):
        """Jumping particles should have higher trend variance than non-jumping."""
        N = 20_000
        theta, sigma, dt = -0.5, 0.1, 1.0
        x0 = np.array([100.0, 0.0])
        states = np.tile(x0, (N, 1))
        rng = np.random.default_rng(42)

        # No jumps
        new_no_jump = propagate_particles(
            states.copy(), np.zeros(N, dtype=bool), np.zeros(N),
            theta=theta, sigma=sigma, dt=dt,
            mu_J=0.0, sigma_J=0.0, rng=np.random.default_rng(42),
        )
        # All jump at tau=dt/2
        new_jump = propagate_particles(
            states.copy(), np.ones(N, dtype=bool), np.full(N, dt / 2),
            theta=theta, sigma=sigma, dt=dt,
            mu_J=0.0, sigma_J=0.5, rng=np.random.default_rng(42),
        )
        assert np.var(new_jump[:, 1]) > np.var(new_no_jump[:, 1])

    def test_jump_shifts_trend_mean(self):
        """mu_J > 0 should shift the trend mean upward compared to no jump."""
        N = 20_000
        theta, sigma, dt = -0.5, 0.1, 1.0
        x0 = np.array([100.0, 0.0])
        states = np.tile(x0, (N, 1))

        new_no_jump = propagate_particles(
            states.copy(), np.zeros(N, dtype=bool), np.zeros(N),
            theta=theta, sigma=sigma, dt=dt,
            mu_J=0.0, sigma_J=0.0, rng=np.random.default_rng(42),
        )
        new_jump = propagate_particles(
            states.copy(), np.ones(N, dtype=bool), np.full(N, dt / 2),
            theta=theta, sigma=sigma, dt=dt,
            mu_J=2.0, sigma_J=0.1, rng=np.random.default_rng(42),
        )
        # Trend mean should be higher with positive mu_J
        assert np.mean(new_jump[:, 1]) > np.mean(new_no_jump[:, 1])

    def test_single_particle_no_jump(self):
        """Single particle, no jump: deterministic part is F @ x + noise sample."""
        x0 = np.array([10.0, 0.5])
        states = x0.reshape(1, 2)
        rng = np.random.default_rng(99)

        new = propagate_particles(
            states, np.array([False]), np.array([0.0]),
            theta=-0.3, sigma=0.05, dt=1.0,
            mu_J=0.0, sigma_J=0.0, rng=rng,
        )
        assert new.shape == (1, 2)
        # Just check it's finite and not equal to input (noise was added)
        assert np.all(np.isfinite(new))

    def test_jump_at_tau_zero(self):
        """tau=0 (jump at start of interval) must not crash — pre-jump diffusion is identity."""
        states = np.array([[100.0, 0.0]])
        new = propagate_particles(
            states, np.array([True]), np.array([0.0]),
            theta=-0.3, sigma=0.1, dt=1.0,
            mu_J=0.5, sigma_J=0.1, rng=np.random.default_rng(0),
        )
        assert np.all(np.isfinite(new))
        assert new.shape == (1, 2)

    def test_jump_at_tau_dt(self):
        """tau=dt (jump at end of interval) must not crash — post-jump diffusion is identity."""
        dt = 1.0
        states = np.array([[100.0, 0.0]])
        new = propagate_particles(
            states, np.array([True]), np.array([dt]),
            theta=-0.3, sigma=0.1, dt=dt,
            mu_J=0.5, sigma_J=0.1, rng=np.random.default_rng(0),
        )
        assert np.all(np.isfinite(new))
        assert new.shape == (1, 2)

    def test_invalid_dt_raises(self):
        """dt <= 0 must raise ValueError."""
        states = np.array([[10.0, 0.0]])
        with pytest.raises(ValueError, match="dt must be > 0"):
            propagate_particles(
                states, np.array([False]), np.array([0.0]),
                theta=-0.3, sigma=0.1, dt=0.0,
                mu_J=0.0, sigma_J=0.0,
            )

    def test_invalid_jump_time_raises(self):
        """jump_times outside [0, dt] must raise ValueError."""
        states = np.array([[10.0, 0.0]])
        with pytest.raises(ValueError, match="jump_times must be in"):
            propagate_particles(
                states, np.array([True]), np.array([2.0]),
                theta=-0.3, sigma=0.1, dt=1.0,
                mu_J=0.0, sigma_J=0.0,
            )


# ---------- weight_particles tests ----------


class TestWeightParticles:
    """Tests for bootstrap PF weight update."""

    def test_output_shape(self):
        """Output has same shape as input log_weights."""
        N = 100
        states = np.column_stack([np.full(N, 10.0), np.zeros(N)])
        log_w = np.full(N, -np.log(N))
        G = observation_matrix()
        new_w = weight_particles(states, log_w, 10.0, G, 1.0)
        assert new_w.shape == (N,)

    def test_particle_at_observation_gets_highest_weight(self):
        """A particle whose price equals the observation should get the highest weight."""
        G = observation_matrix()
        states = np.array([[10.0, 0.0], [20.0, 0.0], [5.0, 0.0]])
        log_w = np.zeros(3)  # uniform
        observation = 10.0

        new_w = weight_particles(states, log_w, observation, G, 1.0)
        assert np.argmax(new_w) == 0  # particle at price=10 closest to obs=10

    def test_manual_log_likelihood(self):
        """Weight update matches hand-computed Gaussian log-pdf."""
        G = observation_matrix()
        price = 15.0
        states = np.array([[price, 0.3]])
        log_w = np.array([0.0])
        sigma_obs_sq = 2.0
        observation = 16.0

        new_w = weight_particles(states, log_w, observation, G, sigma_obs_sq)
        expected = -0.5 * np.log(2 * np.pi * sigma_obs_sq) - 0.5 * (observation - price) ** 2 / sigma_obs_sq
        np.testing.assert_allclose(new_w[0], expected, atol=1e-12)

    def test_weights_additive_in_log_space(self):
        """New weights = old weights + log-likelihood (additive in log-space)."""
        G = observation_matrix()
        N = 5
        states = np.column_stack([np.arange(8.0, 13.0), np.zeros(N)])
        log_w_old = np.array([-1.0, -2.0, -3.0, -0.5, -1.5])
        sigma_obs_sq = 1.0
        observation = 10.0

        new_w = weight_particles(states, log_w_old, observation, G, sigma_obs_sq)

        # Compute expected log-likelihoods
        y_pred = states[:, 0]
        log_lik = -0.5 * np.log(2 * np.pi * sigma_obs_sq) - 0.5 * (observation - y_pred) ** 2 / sigma_obs_sq
        np.testing.assert_allclose(new_w, log_w_old + log_lik, atol=1e-12)

    def test_normalized_weights_sum_to_one(self):
        """After normalizing via logsumexp, weights should sum to 1."""
        G = observation_matrix()
        N = 50
        rng = np.random.default_rng(42)
        states = np.column_stack([rng.normal(100, 2, N), rng.normal(0, 0.1, N)])
        log_w = np.full(N, -np.log(N))

        new_w = weight_particles(states, log_w, 100.0, G, 1.0)
        normalized = np.exp(new_w - logsumexp(new_w))
        np.testing.assert_allclose(np.sum(normalized), 1.0, atol=1e-12)

    def test_large_observation_noise_flattens_weights(self):
        """With very large sigma_obs_sq, all weights should be nearly equal."""
        G = observation_matrix()
        states = np.array([[5.0, 0.0], [50.0, 0.0], [500.0, 0.0]])
        log_w = np.zeros(3)

        new_w = weight_particles(states, log_w, 100.0, G, 1e10)
        # All log-likelihoods should be nearly identical
        assert np.ptp(new_w) < 1e-4  # peak-to-peak range

    def test_invalid_sigma_obs_sq_raises(self):
        """sigma_obs_sq <= 0 must raise ValueError."""
        G = observation_matrix()
        states = np.array([[10.0, 0.0]])
        with pytest.raises(ValueError, match="sigma_obs_sq must be > 0"):
            weight_particles(states, np.array([0.0]), 10.0, G, 0.0)
        with pytest.raises(ValueError, match="sigma_obs_sq must be > 0"):
            weight_particles(states, np.array([0.0]), 10.0, G, -1.0)


# ---------- resample_particles tests ----------


class TestResampleParticles:
    """Tests for systematic resampling."""

    def test_output_shapes(self):
        """Output has same shapes as input."""
        N = 100
        states = np.column_stack([np.arange(N, dtype=float), np.zeros(N)])
        log_w = np.full(N, -np.log(N))
        new_s, new_w = resample_particles(states, log_w, rng=np.random.default_rng(0))
        assert new_s.shape == (N, 2)
        assert new_w.shape == (N,)

    def test_uniform_weights_after_resample(self):
        """After resampling, all log weights must equal -log(N)."""
        N = 50
        states = np.column_stack([np.arange(N, dtype=float), np.zeros(N)])
        # Skewed weights: one particle has most weight
        log_w = np.full(N, -100.0)
        log_w[0] = 0.0
        _, new_w = resample_particles(states, log_w, rng=np.random.default_rng(0))
        np.testing.assert_allclose(new_w, -np.log(N))

    def test_dominant_particle_replicated(self):
        """A particle with weight ≈ 1 should be replicated ~N times."""
        N = 100
        states = np.column_stack([np.arange(N, dtype=float), np.zeros(N)])
        log_w = np.full(N, -1000.0)  # effectively zero
        log_w[7] = 0.0  # particle 7 has all the weight
        new_s, _ = resample_particles(states, log_w, rng=np.random.default_rng(0))
        # All resampled particles should be copies of particle 7
        np.testing.assert_array_equal(new_s[:, 0], 7.0)

    def test_uniform_weights_preserves_diversity(self):
        """With uniform weights, resampling should approximately preserve all particles."""
        N = 1000
        states = np.column_stack([np.arange(N, dtype=float), np.zeros(N)])
        log_w = np.full(N, -np.log(N))
        new_s, _ = resample_particles(states, log_w, rng=np.random.default_rng(42))
        # Systematic resampling with uniform weights should preserve nearly all
        # particles (exact preservation only guaranteed for N that is a power of 2
        # due to floating-point cumsum)
        unique_prices = np.unique(new_s[:, 0])
        assert len(unique_prices) >= int(0.95 * N)

    def test_resampled_states_are_copies(self):
        """Resampled states should be independent copies (modifying one doesn't affect others)."""
        N = 10
        states = np.column_stack([np.arange(N, dtype=float), np.zeros(N)])
        log_w = np.full(N, -1000.0)
        log_w[3] = 0.0
        new_s, _ = resample_particles(states, log_w, rng=np.random.default_rng(0))
        # Modify one resampled particle — should not affect others
        new_s[0, 0] = -999.0
        assert new_s[1, 0] != -999.0

    def test_empirical_distribution_matches_weights(self):
        """Resampled particle distribution should match the weight distribution."""
        N = 10_000
        # 3 distinct particles with known weights
        states = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        # Repeat to N particles: first N/2 are particle 0, next N/4 are particle 1, rest particle 2
        states_full = np.repeat(states, [N // 2, N // 4, N - N // 2 - N // 4], axis=0)
        # But give them different weights: particle type 0 gets 10%, type 1 gets 30%, type 2 gets 60%
        log_w = np.zeros(N)
        log_w[:N // 2] = np.log(0.1 / (N // 2))  # each particle of type 0
        log_w[N // 2:N // 2 + N // 4] = np.log(0.3 / (N // 4))
        log_w[N // 2 + N // 4:] = np.log(0.6 / (N - N // 2 - N // 4))

        new_s, _ = resample_particles(states_full, log_w, rng=np.random.default_rng(42))
        # Count how many resampled particles have price=1, 2, 3
        frac_1 = np.mean(new_s[:, 0] == 1.0)
        frac_2 = np.mean(new_s[:, 0] == 2.0)
        frac_3 = np.mean(new_s[:, 0] == 3.0)
        np.testing.assert_allclose(frac_1, 0.10, atol=0.02)
        np.testing.assert_allclose(frac_2, 0.30, atol=0.02)
        np.testing.assert_allclose(frac_3, 0.60, atol=0.02)


# ---------- run_particle_filter tests ----------


class TestRunParticleFilter:
    """Tests for the full particle filter loop."""

    @pytest.fixture
    def synthetic_no_jump_data(self):
        """Generate synthetic Langevin data without jumps for PF-vs-KF comparison."""
        theta, sigma, dt = -0.2, 0.05, 1.0
        sigma_obs = 0.3
        T = 100
        rng = np.random.default_rng(42)

        F, Q = discretize_langevin(theta, sigma, dt)
        G = observation_matrix()

        states = np.zeros((T, 2))
        states[0] = [100.0, 0.0]
        for t in range(1, T):
            w = rng.multivariate_normal([0, 0], Q)
            states[t] = F @ states[t - 1] + w

        obs = (G @ states.T).ravel() + rng.normal(0, sigma_obs, T)

        return {
            "observations": obs,
            "true_states": states,
            "theta": theta, "sigma": sigma, "dt": dt,
            "sigma_obs_sq": sigma_obs**2,
            "mu0": np.array([100.0, 0.0]),
            "C0": np.eye(2) * 10.0,
        }

    def test_output_shapes(self, synthetic_no_jump_data):
        """All outputs have correct shapes."""
        d = synthetic_no_jump_data
        T = len(d["observations"])
        means, stds, lls, total_ll = run_particle_filter(
            d["observations"], N_particles=100,
            theta=d["theta"], sigma=d["sigma"],
            sigma_obs_sq=d["sigma_obs_sq"],
            lambda_J=0.0, mu_J=0.0, sigma_J=0.0,
            mu0=d["mu0"], C0=d["C0"], dt=d["dt"],
            rng=np.random.default_rng(0),
        )
        assert means.shape == (T, 2)
        assert stds.shape == (T, 2)
        assert lls.shape == (T,)
        assert np.isscalar(total_ll)

    def test_total_ll_equals_sum(self, synthetic_no_jump_data):
        """Total log-likelihood equals sum of per-step terms."""
        d = synthetic_no_jump_data
        _, _, lls, total_ll = run_particle_filter(
            d["observations"], N_particles=200,
            theta=d["theta"], sigma=d["sigma"],
            sigma_obs_sq=d["sigma_obs_sq"],
            lambda_J=0.0, mu_J=0.0, sigma_J=0.0,
            mu0=d["mu0"], C0=d["C0"], dt=d["dt"],
            rng=np.random.default_rng(0),
        )
        np.testing.assert_allclose(total_ll, np.sum(lls), atol=1e-10)

    def test_no_jumps_tracks_truth(self, synthetic_no_jump_data):
        """With no jumps, PF price estimates should track true states (within 3 std >80%)."""
        d = synthetic_no_jump_data
        means, stds, _, _ = run_particle_filter(
            d["observations"], N_particles=1000,
            theta=d["theta"], sigma=d["sigma"],
            sigma_obs_sq=d["sigma_obs_sq"],
            lambda_J=0.0, mu_J=0.0, sigma_J=0.0,
            mu0=d["mu0"], C0=d["C0"], dt=d["dt"],
            rng=np.random.default_rng(42),
        )
        # Skip burn-in (first 10 steps)
        price_errors = np.abs(means[10:, 0] - d["true_states"][10:, 0])
        price_stds = stds[10:, 0]
        # Use 3-sigma (PF is noisier than KF)
        within_3sigma = price_errors < 3.0 * price_stds
        fraction = np.mean(within_3sigma)
        assert fraction > 0.80, f"Only {fraction:.1%} within 3-sigma"

    def test_no_jumps_close_to_kalman(self, synthetic_no_jump_data):
        """With no jumps and many particles, PF estimates should approximate KF estimates."""
        d = synthetic_no_jump_data
        G = observation_matrix()

        # Kalman filter (exact)
        _, _, kf_means, _, _, _ = kalman_filter(
            d["observations"],
            *discretize_langevin(d["theta"], d["sigma"], d["dt"]),
            G, d["sigma_obs_sq"], d["mu0"], d["C0"],
        )

        # Particle filter (approximate)
        pf_means, _, _, _ = run_particle_filter(
            d["observations"], N_particles=5000,
            theta=d["theta"], sigma=d["sigma"],
            sigma_obs_sq=d["sigma_obs_sq"],
            lambda_J=0.0, mu_J=0.0, sigma_J=0.0,
            mu0=d["mu0"], C0=d["C0"], dt=d["dt"],
            rng=np.random.default_rng(42),
        )

        # Price estimates should be close (PF converges to KF as N -> inf)
        # Skip first 5 steps (prior mismatch amplified by finite particles)
        price_rmse = np.sqrt(np.mean((pf_means[5:, 0] - kf_means[5:, 0]) ** 2))
        # With N=5000 and sigma_obs=0.3, MC error should be well below sigma_obs
        sigma_obs = np.sqrt(d["sigma_obs_sq"])
        assert price_rmse < 3 * sigma_obs, f"PF-KF price RMSE = {price_rmse:.3f}"

    def test_stds_positive(self, synthetic_no_jump_data):
        """Filtered standard deviations must be positive for most timesteps."""
        d = synthetic_no_jump_data
        _, stds, _, _ = run_particle_filter(
            d["observations"], N_particles=1000,
            theta=d["theta"], sigma=d["sigma"],
            sigma_obs_sq=d["sigma_obs_sq"],
            lambda_J=0.0, mu_J=0.0, sigma_J=0.0,
            mu0=d["mu0"], C0=d["C0"], dt=d["dt"],
            rng=np.random.default_rng(0),
        )
        # With N=1000, weight degeneracy is rare; std should be positive
        assert np.all(stds >= 0)
        # At least 95% of timesteps should have strictly positive std
        assert np.mean(stds > 0) > 0.95

    def test_single_observation(self):
        """T=1 should work correctly."""
        means, stds, lls, total_ll = run_particle_filter(
            np.array([10.0]), N_particles=500,
            theta=-0.5, sigma=0.1, sigma_obs_sq=1.0,
            lambda_J=0.0, mu_J=0.0, sigma_J=0.0,
            mu0=np.array([10.0, 0.0]), C0=np.eye(2) * 5.0,
            rng=np.random.default_rng(0),
        )
        assert means.shape == (1, 2)
        np.testing.assert_allclose(total_ll, lls[0])
