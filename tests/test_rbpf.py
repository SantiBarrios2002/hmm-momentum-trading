"""Tests for the Rao-Blackwellized Particle Filter (src/langevin/rbpf.py)."""

import numpy as np
import pytest
from scipy.special import logsumexp

from src.langevin.rbpf import (
    initialize_rbpf_particles,
    rbpf_predict_update,
    extract_rbpf_signal,
    run_rbpf,
)
from src.langevin.model import observation_matrix, discretize_langevin
from src.langevin.kalman import kalman_filter
from src.langevin.particle import run_particle_filter


# ──────────────────────────────────────────────────────────────────────
# Tests for initialize_rbpf_particles
# ──────────────────────────────────────────────────────────────────────


class TestInitializeRBPFParticles:
    """Tests for initialize_rbpf_particles."""

    def test_output_shapes(self):
        """Particles dict has correct array shapes."""
        N = 50
        mu0 = np.array([100.0, 0.5])
        C0 = np.eye(2) * 0.1
        particles = initialize_rbpf_particles(N, mu0, C0)

        assert particles['mu'].shape == (N, 2)
        assert particles['C'].shape == (N, 2, 2)
        assert particles['log_weights'].shape == (N,)

    def test_all_particles_share_prior_mean(self):
        """Every particle's mu equals mu0 (no sampling, analytical initialization)."""
        mu0 = np.array([50.0, -0.3])
        C0 = np.diag([1.0, 0.5])
        particles = initialize_rbpf_particles(10, mu0, C0)

        for i in range(10):
            np.testing.assert_array_equal(particles['mu'][i], mu0)

    def test_all_particles_share_prior_covariance(self):
        """Every particle's C equals C0."""
        mu0 = np.array([50.0, -0.3])
        C0 = np.array([[1.0, 0.2], [0.2, 0.5]])
        particles = initialize_rbpf_particles(10, mu0, C0)

        for i in range(10):
            np.testing.assert_array_equal(particles['C'][i], C0)

    def test_uniform_weights(self):
        """All log-weights equal -log(N), summing to 1 in probability space."""
        N = 25
        particles = initialize_rbpf_particles(N, np.zeros(2), np.eye(2))

        expected_log_w = -np.log(N)
        np.testing.assert_allclose(particles['log_weights'], expected_log_w)
        # Weights sum to 1 in probability space
        total = np.exp(logsumexp(particles['log_weights']))
        np.testing.assert_allclose(total, 1.0, atol=1e-14)

    def test_single_particle(self):
        """N=1: single particle with log-weight 0 (weight = 1)."""
        mu0 = np.array([10.0, 0.0])
        C0 = np.eye(2) * 2.0
        particles = initialize_rbpf_particles(1, mu0, C0)

        assert particles['mu'].shape == (1, 2)
        np.testing.assert_array_equal(particles['mu'][0], mu0)
        np.testing.assert_allclose(particles['log_weights'][0], 0.0, atol=1e-15)

    def test_invalid_N_zero(self):
        """N=0 raises ValueError."""
        with pytest.raises(ValueError, match="N must be >= 1"):
            initialize_rbpf_particles(0, np.zeros(2), np.eye(2))

    def test_invalid_N_negative(self):
        """N<0 raises ValueError."""
        with pytest.raises(ValueError, match="N must be >= 1"):
            initialize_rbpf_particles(-5, np.zeros(2), np.eye(2))

    def test_invalid_mu0_shape(self):
        """Wrong mu0 shape raises ValueError."""
        with pytest.raises(ValueError, match="mu0 must have shape"):
            initialize_rbpf_particles(10, np.zeros(3), np.eye(2))

    def test_invalid_C0_shape(self):
        """Wrong C0 shape raises ValueError."""
        with pytest.raises(ValueError, match="C0 must have shape"):
            initialize_rbpf_particles(10, np.zeros(2), np.eye(3))

    def test_particles_are_independent_copies(self):
        """Modifying one particle's mu/C does not affect others."""
        particles = initialize_rbpf_particles(5, np.array([1.0, 2.0]), np.eye(2))
        particles['mu'][0, 0] = 999.0
        particles['C'][0, 0, 0] = 999.0

        # Other particles unchanged
        np.testing.assert_array_equal(particles['mu'][1], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(particles['C'][1], np.eye(2))


# ──────────────────────────────────────────────────────────────────────
# Tests for rbpf_predict_update
# ──────────────────────────────────────────────────────────────────────


class TestRBPFPredictUpdate:
    """Tests for rbpf_predict_update."""

    # Shared test parameters
    theta = -0.5
    sigma = 0.3
    dt = 1.0
    sigma_obs_sq = 0.1
    mu0 = np.array([100.0, 0.0])
    C0 = np.diag([1.0, 0.5])
    G = observation_matrix()

    def test_output_shapes(self):
        """Output dict has same shapes as input."""
        N = 20
        particles = initialize_rbpf_particles(N, self.mu0, self.C0)
        result = rbpf_predict_update(
            particles, observation=100.5,
            theta=self.theta, sigma=self.sigma, dt=self.dt,
            lambda_J=0.0, mu_J=0.0, sigma_J=0.0,
            G=self.G, sigma_obs_sq=self.sigma_obs_sq,
            rng=np.random.default_rng(42),
        )
        assert result['mu'].shape == (N, 2)
        assert result['C'].shape == (N, 2, 2)
        assert result['log_weights'].shape == (N,)

    def test_no_jumps_all_particles_identical(self):
        """With lambda_J=0, all particles get the same Kalman update (no jump diversity)."""
        N = 10
        particles = initialize_rbpf_particles(N, self.mu0, self.C0)
        result = rbpf_predict_update(
            particles, observation=100.5,
            theta=self.theta, sigma=self.sigma, dt=self.dt,
            lambda_J=0.0, mu_J=0.0, sigma_J=0.0,
            G=self.G, sigma_obs_sq=self.sigma_obs_sq,
            rng=np.random.default_rng(42),
        )
        # All means should be identical (same prior, same observation, no jumps)
        for i in range(1, N):
            np.testing.assert_allclose(result['mu'][i], result['mu'][0], atol=1e-14)
            np.testing.assert_allclose(result['C'][i], result['C'][0], atol=1e-14)
            np.testing.assert_allclose(
                result['log_weights'][i], result['log_weights'][0], atol=1e-14,
            )

    def test_no_jumps_matches_kalman_filter(self):
        """With lambda_J=0, RBPF predict-update matches a Kalman predict+update step.

        rbpf_predict_update always does predict+update (designed for t>0).
        At t=0, the caller (run_rbpf) handles the update-only case separately.
        So we test at t=1: set particles to KF state after t=0, then compare.
        """
        from src.langevin.model import discretize_langevin
        from src.langevin.kalman import kalman_update

        N = 5
        F, Q = discretize_langevin(self.theta, self.sigma, self.dt)
        observations = np.array([100.3, 100.8])

        # Run full KF on 2 observations
        _, _, kf_means, kf_covs, kf_lls, _ = kalman_filter(
            observations, F, Q, self.G, self.sigma_obs_sq, self.mu0, self.C0,
        )

        # Set up RBPF particles at KF state after t=0
        particles = initialize_rbpf_particles(N, kf_means[0], kf_covs[0])

        # Run rbpf_predict_update for t=1
        result = rbpf_predict_update(
            particles, observation=observations[1],
            theta=self.theta, sigma=self.sigma, dt=self.dt,
            lambda_J=0.0, mu_J=0.0, sigma_J=0.0,
            G=self.G, sigma_obs_sq=self.sigma_obs_sq,
            rng=np.random.default_rng(0),
        )

        # RBPF particle 0 should match KF at t=1
        np.testing.assert_allclose(result['mu'][0], kf_means[1], atol=1e-12)
        np.testing.assert_allclose(result['C'][0], kf_covs[1], atol=1e-12)

    def test_weights_differ_with_jumps(self):
        """With lambda_J > 0, particles that jump get different weights."""
        N = 100
        particles = initialize_rbpf_particles(N, self.mu0, self.C0)
        result = rbpf_predict_update(
            particles, observation=100.5,
            theta=self.theta, sigma=self.sigma, dt=self.dt,
            lambda_J=5.0, mu_J=0.0, sigma_J=1.0,
            G=self.G, sigma_obs_sq=self.sigma_obs_sq,
            rng=np.random.default_rng(42),
        )
        # With high jump rate, some particles jump and others don't
        # Their weights should differ
        weights = result['log_weights']
        assert np.std(weights) > 0, "All weights identical — jumps had no effect"

    def test_covariances_remain_psd(self):
        """Filtered covariances must be positive semi-definite after update."""
        N = 30
        particles = initialize_rbpf_particles(N, self.mu0, self.C0)
        result = rbpf_predict_update(
            particles, observation=100.5,
            theta=self.theta, sigma=self.sigma, dt=self.dt,
            lambda_J=2.0, mu_J=0.1, sigma_J=0.5,
            G=self.G, sigma_obs_sq=self.sigma_obs_sq,
            rng=np.random.default_rng(99),
        )
        for i in range(N):
            eigvals = np.linalg.eigvalsh(result['C'][i])
            assert np.all(eigvals >= -1e-12), (
                f"Particle {i} has non-PSD covariance, eigenvalues: {eigvals}"
            )

    def test_covariances_symmetric(self):
        """Filtered covariances must be symmetric."""
        N = 20
        particles = initialize_rbpf_particles(N, self.mu0, self.C0)
        result = rbpf_predict_update(
            particles, observation=100.5,
            theta=self.theta, sigma=self.sigma, dt=self.dt,
            lambda_J=2.0, mu_J=0.0, sigma_J=0.5,
            G=self.G, sigma_obs_sq=self.sigma_obs_sq,
            rng=np.random.default_rng(7),
        )
        for i in range(N):
            np.testing.assert_allclose(
                result['C'][i], result['C'][i].T, atol=1e-14,
            )

    def test_invalid_dt(self):
        """dt <= 0 raises ValueError."""
        particles = initialize_rbpf_particles(5, self.mu0, self.C0)
        with pytest.raises(ValueError, match="dt must be > 0"):
            rbpf_predict_update(
                particles, observation=100.0,
                theta=self.theta, sigma=self.sigma, dt=0.0,
                lambda_J=0.0, mu_J=0.0, sigma_J=0.0,
                G=self.G, sigma_obs_sq=self.sigma_obs_sq,
            )

    def test_invalid_sigma_obs_sq(self):
        """sigma_obs_sq <= 0 raises ValueError."""
        particles = initialize_rbpf_particles(5, self.mu0, self.C0)
        with pytest.raises(ValueError, match="sigma_obs_sq must be > 0"):
            rbpf_predict_update(
                particles, observation=100.0,
                theta=self.theta, sigma=self.sigma, dt=self.dt,
                lambda_J=0.0, mu_J=0.0, sigma_J=0.0,
                G=self.G, sigma_obs_sq=-0.1,
            )

    def test_multi_step_no_jumps_matches_kalman(self):
        """Running RBPF predict-update sequentially with lambda_J=0 matches full Kalman filter."""
        from src.langevin.model import discretize_langevin
        from src.langevin.kalman import kalman_update

        rng = np.random.default_rng(123)
        T = 20
        F, Q = discretize_langevin(self.theta, self.sigma, self.dt)

        # Generate synthetic observations
        true_state = self.mu0.copy()
        observations = np.zeros(T)
        for t in range(T):
            if t > 0:
                true_state = F @ true_state + rng.multivariate_normal(np.zeros(2), Q)
            observations[t] = (self.G @ true_state).item() + rng.normal(0, np.sqrt(self.sigma_obs_sq))

        # Run Kalman filter
        _, _, kf_means, kf_covs, kf_lls, kf_total_ll = kalman_filter(
            observations, F, Q, self.G, self.sigma_obs_sq, self.mu0, self.C0,
        )

        # Run RBPF step-by-step (no jumps → should match KF exactly)
        N = 5
        particles = initialize_rbpf_particles(N, self.mu0, self.C0)
        rbpf_rng = np.random.default_rng(0)

        for t in range(T):
            if t == 0:
                # At t=0, KF uses prior directly (no predict step).
                # rbpf_predict_update always does predict+update (for t>0).
                # So handle t=0 the same way run_rbpf will: kalman_update only.
                mu_upd, C_upd, ll = kalman_update(
                    self.mu0, self.C0, self.G, self.sigma_obs_sq, observations[0],
                )
                for i in range(N):
                    particles['mu'][i] = mu_upd
                    particles['C'][i] = C_upd
                    particles['log_weights'][i] = -np.log(N) + ll
            else:
                particles = rbpf_predict_update(
                    particles, observation=observations[t],
                    theta=self.theta, sigma=self.sigma, dt=self.dt,
                    lambda_J=0.0, mu_J=0.0, sigma_J=0.0,
                    G=self.G, sigma_obs_sq=self.sigma_obs_sq,
                    rng=rbpf_rng,
                )

            np.testing.assert_allclose(
                particles['mu'][0], kf_means[t], atol=1e-10,
                err_msg=f"Mean mismatch at t={t}",
            )
            np.testing.assert_allclose(
                particles['C'][0], kf_covs[t], atol=1e-10,
                err_msg=f"Covariance mismatch at t={t}",
            )


# ──────────────────────────────────────────────────────────────────────
# Tests for extract_rbpf_signal
# ──────────────────────────────────────────────────────────────────────


class TestExtractRBPFSignal:
    """Tests for extract_rbpf_signal."""

    def test_uniform_weights_gives_simple_mean(self):
        """With uniform weights, the signal is just the arithmetic mean of particle means."""
        N = 4
        particles = {
            'mu': np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            'C': np.tile(np.eye(2) * 0.1, (N, 1, 1)),
            'log_weights': np.full(N, -np.log(N)),
        }
        mean, std = extract_rbpf_signal(particles)
        np.testing.assert_allclose(mean, [4.0, 5.0], atol=1e-14)

    def test_single_particle_returns_its_mean(self):
        """With N=1, the signal is that particle's mean."""
        mu_val = np.array([42.0, -1.5])
        C_val = np.array([[0.5, 0.1], [0.1, 0.3]])
        particles = {
            'mu': mu_val.reshape(1, 2),
            'C': C_val.reshape(1, 2, 2),
            'log_weights': np.array([0.0]),
        }
        mean, std = extract_rbpf_signal(particles)
        np.testing.assert_allclose(mean, mu_val, atol=1e-14)

    def test_std_includes_within_and_between_variance(self):
        """Std must reflect both Kalman uncertainty (within) and particle spread (between)."""
        N = 2
        # Two particles with identical small covariance but different means
        particles = {
            'mu': np.array([[10.0, 0.0], [20.0, 0.0]]),
            'C': np.tile(np.eye(2) * 0.01, (N, 1, 1)),
            'log_weights': np.full(N, -np.log(N)),
        }
        mean, std = extract_rbpf_signal(particles)
        # Between-variance in price: ((10-15)^2 + (20-15)^2)/2 = 25
        # Within-variance in price: 0.01
        # Total variance: 25.01, std ≈ 5.0
        np.testing.assert_allclose(std[0], np.sqrt(25.01), atol=1e-10)

    def test_dominant_weight_concentrates_on_one_particle(self):
        """When one particle has overwhelming weight, signal ≈ that particle's mean."""
        N = 3
        particles = {
            'mu': np.array([[100.0, 1.0], [0.0, 0.0], [0.0, 0.0]]),
            'C': np.tile(np.eye(2) * 0.1, (N, 1, 1)),
            'log_weights': np.array([0.0, -100.0, -100.0]),  # particle 0 dominates
        }
        mean, std = extract_rbpf_signal(particles)
        np.testing.assert_allclose(mean, [100.0, 1.0], atol=1e-10)

    def test_output_shapes(self):
        """Mean and std both have shape (2,)."""
        particles = initialize_rbpf_particles(10, np.zeros(2), np.eye(2))
        mean, std = extract_rbpf_signal(particles)
        assert mean.shape == (2,)
        assert std.shape == (2,)


# ──────────────────────────────────────────────────────────────────────
# Tests for run_rbpf
# ──────────────────────────────────────────────────────────────────────


class TestRunRBPF:
    """Tests for run_rbpf."""

    # Shared parameters
    theta = -0.5
    sigma = 0.3
    dt = 1.0
    sigma_obs_sq = 0.1
    mu0 = np.array([100.0, 0.0])
    C0 = np.diag([1.0, 0.5])

    def _generate_synthetic_data(self, T, lambda_J=0.0, mu_J=0.0, sigma_J=0.0, seed=42):
        """Generate synthetic observations from the Langevin model."""
        rng = np.random.default_rng(seed)
        F, Q = discretize_langevin(self.theta, self.sigma, self.dt)
        G = observation_matrix()

        true_states = np.zeros((T, 2))
        observations = np.zeros(T)
        true_states[0] = self.mu0.copy()

        for t in range(T):
            if t > 0:
                true_states[t] = F @ true_states[t - 1] + rng.multivariate_normal(np.zeros(2), Q)
                # Optionally add jumps
                if lambda_J > 0 and rng.random() < (1.0 - np.exp(-lambda_J * self.dt)):
                    true_states[t, 1] += rng.normal(mu_J, sigma_J)
            observations[t] = (G @ true_states[t]).item() + rng.normal(0, np.sqrt(self.sigma_obs_sq))

        return observations, true_states

    def test_output_shapes(self):
        """All outputs have correct shapes."""
        T = 30
        N = 20
        obs, _ = self._generate_synthetic_data(T)
        means, stds, lls, total_ll, n_eff = run_rbpf(
            obs, N, self.theta, self.sigma, self.sigma_obs_sq,
            lambda_J=0.0, mu_J=0.0, sigma_J=0.0,
            mu0=self.mu0, C0=self.C0, dt=self.dt,
            rng=np.random.default_rng(0),
        )
        assert means.shape == (T, 2)
        assert stds.shape == (T, 2)
        assert lls.shape == (T,)
        assert isinstance(total_ll, float)
        assert n_eff.shape == (T,)

    def test_n_eff_bounded(self):
        """Effective sample size must be in [1, N]."""
        T, N = 30, 50
        obs, _ = self._generate_synthetic_data(T)
        _, _, _, _, n_eff = run_rbpf(
            obs, N, self.theta, self.sigma, self.sigma_obs_sq,
            lambda_J=1.0, mu_J=0.0, sigma_J=0.5,
            mu0=self.mu0, C0=self.C0, dt=self.dt,
            rng=np.random.default_rng(0),
        )
        assert np.all(n_eff >= 1.0 - 1e-10), f"N_eff below 1: {n_eff.min()}"
        assert np.all(n_eff <= N + 1e-10), f"N_eff above N: {n_eff.max()}"

    def test_no_jumps_matches_kalman(self):
        """With lambda_J=0, RBPF reduces to a pure Kalman filter (Issue #44, test 4).

        This is the fundamental correctness check: when there are no jumps,
        all particles should track identical Kalman states, and the RBPF
        output should match the Kalman filter exactly.
        """
        T = 50
        N = 20
        obs, _ = self._generate_synthetic_data(T, lambda_J=0.0)
        F, Q = discretize_langevin(self.theta, self.sigma, self.dt)
        G = observation_matrix()

        # Run Kalman filter
        _, _, kf_means, _, _, kf_total_ll = kalman_filter(
            obs, F, Q, G, self.sigma_obs_sq, self.mu0, self.C0,
        )

        # Run RBPF with no jumps
        rbpf_means, _, _, _, _ = run_rbpf(
            obs, N, self.theta, self.sigma, self.sigma_obs_sq,
            lambda_J=0.0, mu_J=0.0, sigma_J=0.0,
            mu0=self.mu0, C0=self.C0, dt=self.dt,
            rng=np.random.default_rng(0),
        )

        # Filtered means should match KF
        np.testing.assert_allclose(
            rbpf_means, kf_means, atol=1e-10,
            err_msg="RBPF (no jumps) should match Kalman filter exactly",
        )

    def test_rbpf_outperforms_pf_variance(self):
        """RBPF variance <= PF variance (Issue #44, test 3 — THE MOST IMPORTANT TEST).

        The Rao-Blackwell theorem guarantees that analytically marginalizing
        the continuous state (as RBPF does via Kalman filters) yields
        lower-or-equal variance estimates compared to sampling the state
        (as the standard PF does).

        We test this by running both filters multiple times on the same data
        and comparing the variance of their price estimates across runs.
        """
        T = 30
        N = 100  # enough particles for meaningful comparison

        # Generate data WITH jumps (where the RBPF advantage matters)
        obs, true_states = self._generate_synthetic_data(
            T, lambda_J=1.0, mu_J=0.0, sigma_J=0.5, seed=42,
        )

        n_runs = 30
        rbpf_prices = np.zeros((n_runs, T))
        pf_prices = np.zeros((n_runs, T))

        for run in range(n_runs):
            rng_rbpf = np.random.default_rng(run)
            rng_pf = np.random.default_rng(run)

            rbpf_means, _, _, _, _ = run_rbpf(
                obs, N, self.theta, self.sigma, self.sigma_obs_sq,
                lambda_J=1.0, mu_J=0.0, sigma_J=0.5,
                mu0=self.mu0, C0=self.C0, dt=self.dt,
                rng=rng_rbpf,
            )
            rbpf_prices[run] = rbpf_means[:, 0]

            pf_means, _, _, _ = run_particle_filter(
                obs, N, self.theta, self.sigma, self.sigma_obs_sq,
                lambda_J=1.0, mu_J=0.0, sigma_J=0.5,
                mu0=self.mu0, C0=self.C0, dt=self.dt,
                rng=rng_pf,
            )
            pf_prices[run] = pf_means[:, 0]

        # Compare variance of price estimates across runs
        rbpf_var = np.var(rbpf_prices, axis=0)  # (T,)
        pf_var = np.var(pf_prices, axis=0)       # (T,)

        # RBPF variance should be <= PF variance on average
        # (not necessarily at every single timestep due to finite samples)
        mean_rbpf_var = np.mean(rbpf_var)
        mean_pf_var = np.mean(pf_var)

        assert mean_rbpf_var < mean_pf_var, (
            f"Rao-Blackwell violation: RBPF variance ({mean_rbpf_var:.6f}) "
            f">= PF variance ({mean_pf_var:.6f})"
        )

    def test_stds_positive(self):
        """Filtered standard deviations must be strictly positive."""
        T, N = 20, 30
        obs, _ = self._generate_synthetic_data(T)
        _, stds, _, _, _ = run_rbpf(
            obs, N, self.theta, self.sigma, self.sigma_obs_sq,
            lambda_J=1.0, mu_J=0.0, sigma_J=0.5,
            mu0=self.mu0, C0=self.C0, dt=self.dt,
            rng=np.random.default_rng(0),
        )
        assert np.all(stds > 0), "Filtered stds must be positive"

    def test_total_ll_equals_sum(self):
        """Total log-likelihood equals sum of per-step values."""
        T, N = 25, 15
        obs, _ = self._generate_synthetic_data(T)
        _, _, lls, total_ll, _ = run_rbpf(
            obs, N, self.theta, self.sigma, self.sigma_obs_sq,
            lambda_J=0.5, mu_J=0.0, sigma_J=0.3,
            mu0=self.mu0, C0=self.C0, dt=self.dt,
            rng=np.random.default_rng(0),
        )
        np.testing.assert_allclose(total_ll, np.sum(lls), atol=1e-10)

    def test_single_observation(self):
        """T=1: RBPF handles a single observation without error."""
        obs = np.array([100.5])
        means, stds, lls, total_ll, n_eff = run_rbpf(
            obs, 10, self.theta, self.sigma, self.sigma_obs_sq,
            lambda_J=0.0, mu_J=0.0, sigma_J=0.0,
            mu0=self.mu0, C0=self.C0, dt=self.dt,
            rng=np.random.default_rng(0),
        )
        assert means.shape == (1, 2)
        assert np.isfinite(total_ll)

    def test_single_particle(self):
        """N=1: RBPF runs with a single particle (degenerates to KF with jumps)."""
        T = 15
        obs, _ = self._generate_synthetic_data(T)
        means, stds, lls, total_ll, n_eff = run_rbpf(
            obs, 1, self.theta, self.sigma, self.sigma_obs_sq,
            lambda_J=0.5, mu_J=0.0, sigma_J=0.3,
            mu0=self.mu0, C0=self.C0, dt=self.dt,
            rng=np.random.default_rng(0),
        )
        assert means.shape == (T, 2)
        assert np.all(np.isfinite(means))
        # N_eff should always be 1 with a single particle
        np.testing.assert_allclose(n_eff, 1.0, atol=1e-10)
