"""Tests for src/langevin/model.py — Langevin state-space model."""

import numpy as np
import pytest

from src.langevin.model import (
    discretize_langevin,
    observation_matrix,
    transition_density_with_jump,
)


# ---------- discretize_langevin tests ----------


class TestDiscretizeLangevin:
    """Tests for the Langevin SDE discretization."""

    def test_identity_at_zero_dt(self):
        """F approaches I and Q approaches 0 as dt -> 0."""
        F, Q = discretize_langevin(theta=-0.5, sigma=0.1, dt=1e-10)
        np.testing.assert_allclose(F, np.eye(2), atol=1e-8)
        # Q elements scale as sigma^2 * dt ≈ 1e-12, so use matching tolerance
        np.testing.assert_allclose(Q, np.zeros((2, 2)), atol=1e-11)

    def test_Q_positive_semidefinite(self):
        """Q must be positive semi-definite for any valid parameters."""
        for theta in [-1.0, -0.1, -0.01]:
            for sigma in [0.01, 0.1, 1.0]:
                for dt in [0.01, 0.1, 1.0]:
                    _, Q = discretize_langevin(theta, sigma, dt)
                    eigenvalues = np.linalg.eigvalsh(Q)
                    assert np.all(eigenvalues >= -1e-12), (
                        f"Q not PSD for theta={theta}, sigma={sigma}, dt={dt}: "
                        f"eigenvalues={eigenvalues}"
                    )

    def test_Q_symmetric(self):
        """Q must be symmetric."""
        _, Q = discretize_langevin(theta=-0.5, sigma=0.1, dt=1.0)
        np.testing.assert_allclose(Q, Q.T, atol=1e-15)

    def test_F_shape_and_structure(self):
        """F has correct shape and known structure for the Langevin model."""
        F, _ = discretize_langevin(theta=-0.5, sigma=0.1, dt=1.0)
        assert F.shape == (2, 2)
        # F[0,0] = 1 always (price integrates trend, no decay)
        np.testing.assert_allclose(F[0, 0], 1.0, atol=1e-12)
        # F[1,0] = 0 always (trend does not depend on price)
        np.testing.assert_allclose(F[1, 0], 0.0, atol=1e-12)
        # F[1,1] = exp(theta * dt) (trend mean-reverts)
        np.testing.assert_allclose(F[1, 1], np.exp(-0.5 * 1.0), atol=1e-12)

    def test_F_analytical_formula(self):
        """F matches the known analytical formula for the 2x2 Langevin case.

        F = [[1, (exp(theta*dt) - 1) / theta],
             [0, exp(theta*dt)]]
        """
        theta, dt = -0.3, 2.0
        F, _ = discretize_langevin(theta=theta, sigma=0.1, dt=dt)
        e_td = np.exp(theta * dt)
        F_expected = np.array([
            [1.0, (e_td - 1.0) / theta],
            [0.0, e_td],
        ])
        np.testing.assert_allclose(F, F_expected, atol=1e-12)

    def test_positive_theta_raises(self):
        """theta > 0 (explosive dynamics) must raise ValueError."""
        with pytest.raises(ValueError, match="theta must be <= 0"):
            discretize_langevin(theta=0.5, sigma=0.1, dt=1.0)

    def test_theta_zero_works(self):
        """theta=0 (pure random walk) is valid and gives F = [[1, dt], [0, 1]]."""
        F, Q = discretize_langevin(theta=0.0, sigma=0.1, dt=1.0)
        np.testing.assert_allclose(F, np.array([[1.0, 1.0], [0.0, 1.0]]), atol=1e-12)
        assert Q[1, 1] > 0  # non-degenerate noise

    def test_Q_scales_with_sigma_squared(self):
        """Doubling sigma should quadruple Q (Q is proportional to sigma^2)."""
        _, Q1 = discretize_langevin(theta=-0.5, sigma=0.1, dt=1.0)
        _, Q2 = discretize_langevin(theta=-0.5, sigma=0.2, dt=1.0)
        np.testing.assert_allclose(Q2, Q1 * 4.0, atol=1e-14)

    def test_small_dt_approximation(self):
        """For small dt, F ≈ I + A*dt and Q ≈ b*b'*dt."""
        theta, sigma, dt = -0.5, 0.1, 1e-6
        F, Q = discretize_langevin(theta, sigma, dt)

        A = np.array([[0.0, 1.0], [0.0, theta]])
        F_approx = np.eye(2) + A * dt
        np.testing.assert_allclose(F, F_approx, atol=1e-10)

        bbT_dt = np.array([[0.0, 0.0], [0.0, sigma**2]]) * dt
        # Off-diagonal terms are O(dt^2), use relative tolerance for [1,1]
        np.testing.assert_allclose(Q[1, 1], bbT_dt[1, 1], rtol=1e-4)
        # Q[0,0] ≈ sigma^2 * dt^3 / 3 (higher-order), Q[0,1] ≈ sigma^2 * dt^2 / 2
        expected_Q00 = sigma**2 * dt**3 / 3.0
        assert abs(Q[0, 0]) < 10 * expected_Q00
        expected_Q01 = sigma**2 * dt**2 / 2.0
        assert abs(Q[0, 1]) < 10 * expected_Q01


# ---------- observation_matrix tests ----------


class TestObservationMatrix:
    """Tests for the observation matrix G."""

    def test_shape(self):
        """G has shape (1, 2)."""
        G = observation_matrix()
        assert G.shape == (1, 2)

    def test_extracts_price(self):
        """G @ x extracts the price component x1."""
        G = observation_matrix()
        x = np.array([3.14, -0.5])  # [price, trend]
        y = G @ x
        np.testing.assert_allclose(y, np.array([3.14]))


# ---------- transition_density_with_jump tests ----------


class TestTransitionDensityWithJump:
    """Tests for the conditional state transition density."""

    def test_no_jump_matches_discretize(self):
        """With jump_occurred=False, output must match discretize_langevin exactly."""
        theta, sigma, dt = -0.5, 0.1, 1.0
        x_prev = np.array([100.0, 0.05])

        F, Q = discretize_langevin(theta, sigma, dt)
        mean, cov = transition_density_with_jump(
            x_prev, theta, sigma, dt, jump_occurred=False
        )

        np.testing.assert_allclose(mean, F @ x_prev, atol=1e-12)
        np.testing.assert_allclose(cov, Q, atol=1e-14)

    def test_jump_increases_variance(self):
        """A jump must increase the covariance compared to no jump."""
        theta, sigma, dt = -0.5, 0.1, 1.0
        x_prev = np.array([100.0, 0.05])

        _, cov_no_jump = transition_density_with_jump(
            x_prev, theta, sigma, dt, jump_occurred=False
        )
        _, cov_jump = transition_density_with_jump(
            x_prev, theta, sigma, dt,
            jump_occurred=True, tau=0.5, mu_J=0.0, sigma_J=0.5,
        )

        # The jump adds sigma_J^2 to the trend variance, so trace must increase
        assert np.trace(cov_jump) > np.trace(cov_no_jump)

    def test_jump_shifts_mean(self):
        """A jump with nonzero mu_J must shift the conditional mean."""
        theta, sigma, dt = -0.5, 0.1, 1.0
        x_prev = np.array([100.0, 0.0])

        mean_no, _ = transition_density_with_jump(
            x_prev, theta, sigma, dt, jump_occurred=False
        )
        mean_up, _ = transition_density_with_jump(
            x_prev, theta, sigma, dt,
            jump_occurred=True, tau=0.5, mu_J=1.0, sigma_J=0.1,
        )

        # Positive jump in trend should increase the price mean
        assert mean_up[0] > mean_no[0]
        # Trend mean should also be higher
        assert mean_up[1] > mean_no[1]

    def test_jump_cov_symmetric_and_psd(self):
        """Covariance with jump must be symmetric and PSD."""
        theta, sigma, dt = -0.3, 0.2, 1.0
        x_prev = np.array([50.0, -0.1])

        _, cov = transition_density_with_jump(
            x_prev, theta, sigma, dt,
            jump_occurred=True, tau=0.3, mu_J=0.5, sigma_J=0.8,
        )

        np.testing.assert_allclose(cov, cov.T, atol=1e-14)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-12)

    def test_zero_jump_size_matches_no_jump(self):
        """sigma_J=0 and mu_J=0 must give same result as jump_occurred=False (semigroup identity)."""
        theta, sigma, dt = -0.5, 0.1, 1.0
        x_prev = np.array([100.0, 0.05])

        mean_no, cov_no = transition_density_with_jump(
            x_prev, theta, sigma, dt, jump_occurred=False
        )
        for tau in [0.0, 0.3, 0.5, 1.0]:
            mean_j, cov_j = transition_density_with_jump(
                x_prev, theta, sigma, dt,
                jump_occurred=True, tau=tau, mu_J=0.0, sigma_J=0.0,
            )
            np.testing.assert_allclose(mean_j, mean_no, atol=1e-10)
            np.testing.assert_allclose(cov_j, cov_no, atol=1e-10)

    def test_jump_at_tau_zero(self):
        """tau=0: pre-jump diffusion is identity; result = full diffusion after jump."""
        theta, sigma, dt, mu_J, sigma_J = -0.5, 0.1, 1.0, 0.3, 0.2
        x_prev = np.array([10.0, 0.1])

        mean, cov = transition_density_with_jump(
            x_prev, theta, sigma, dt,
            jump_occurred=True, tau=0.0, mu_J=mu_J, sigma_J=sigma_J,
        )
        F_full, Q_full = discretize_langevin(theta, sigma, dt)
        expected_mean = F_full @ x_prev + F_full @ np.array([0.0, mu_J])
        expected_cov = (
            F_full @ np.array([[0.0, 0.0], [0.0, sigma_J**2]]) @ F_full.T + Q_full
        )
        np.testing.assert_allclose(mean, expected_mean, atol=1e-10)
        np.testing.assert_allclose(cov, expected_cov, atol=1e-10)

    def test_jump_at_tau_dt(self):
        """tau=dt: post-jump diffusion is identity; jump cov added directly."""
        theta, sigma, dt, sigma_J = -0.5, 0.1, 1.0, 0.5
        x_prev = np.array([10.0, 0.0])

        mean, cov = transition_density_with_jump(
            x_prev, theta, sigma, dt,
            jump_occurred=True, tau=dt, mu_J=0.0, sigma_J=sigma_J,
        )
        F_full, Q_full = discretize_langevin(theta, sigma, dt)
        expected_cov = Q_full + np.array([[0.0, 0.0], [0.0, sigma_J**2]])
        np.testing.assert_allclose(cov, expected_cov, atol=1e-10)

    def test_tau_out_of_bounds_raises(self):
        """tau outside [0, dt] must raise ValueError."""
        x_prev = np.array([10.0, 0.0])
        with pytest.raises(ValueError, match="tau must be in"):
            transition_density_with_jump(
                x_prev, -0.5, 0.1, 1.0,
                jump_occurred=True, tau=-0.1,
            )
        with pytest.raises(ValueError, match="tau must be in"):
            transition_density_with_jump(
                x_prev, -0.5, 0.1, 1.0,
                jump_occurred=True, tau=1.5,
            )
