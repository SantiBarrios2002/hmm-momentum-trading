"""Tests for src/langevin/kalman.py — Kalman filter."""

import numpy as np
import pytest

from src.langevin.kalman import kalman_predict, kalman_update, kalman_filter
from src.langevin.model import discretize_langevin, observation_matrix


# ---------- kalman_predict tests ----------


class TestKalmanPredict:
    """Tests for the Kalman prediction step."""

    def test_identity_no_noise(self):
        """F=I, Q=0 → predicted state equals previous state."""
        mu = np.array([1.0, 2.0])
        C = np.array([[0.5, 0.1], [0.1, 0.3]])
        F = np.eye(2)
        Q = np.zeros((2, 2))

        mu_pred, C_pred = kalman_predict(mu, C, F, Q)
        np.testing.assert_allclose(mu_pred, mu)
        np.testing.assert_allclose(C_pred, C)

    def test_mean_propagation(self):
        """Predicted mean equals F @ mu."""
        mu = np.array([10.0, 0.5])
        C = np.eye(2)
        F = np.array([[1.0, 1.0], [0.0, 0.8]])
        Q = np.zeros((2, 2))

        mu_pred, _ = kalman_predict(mu, C, F, Q)
        np.testing.assert_allclose(mu_pred, F @ mu)

    def test_covariance_grows_with_Q(self):
        """Adding process noise Q increases the predicted covariance."""
        mu = np.array([0.0, 0.0])
        C = np.eye(2) * 0.1
        F = np.eye(2)
        Q = np.eye(2) * 0.5

        _, C_pred = kalman_predict(mu, C, F, Q)
        # C_pred = F C F' + Q = C + Q, so trace should increase by trace(Q)
        np.testing.assert_allclose(np.trace(C_pred), np.trace(C) + np.trace(Q))

    def test_output_symmetric_and_psd(self):
        """Predicted covariance must be symmetric and PSD."""
        mu = np.array([5.0, -1.0])
        C = np.array([[1.0, 0.3], [0.3, 0.5]])
        F = np.array([[1.0, 0.7], [0.0, 0.6]])
        Q = np.array([[0.01, 0.005], [0.005, 0.02]])

        _, C_pred = kalman_predict(mu, C, F, Q)
        np.testing.assert_allclose(C_pred, C_pred.T, atol=1e-14)
        eigenvalues = np.linalg.eigvalsh(C_pred)
        assert np.all(eigenvalues >= -1e-12)


# ---------- kalman_update tests ----------


class TestKalmanUpdate:
    """Tests for the Kalman update (measurement) step."""

    def test_zero_observation_noise(self):
        """With sigma_obs=0, posterior collapses to the observation (for price)."""
        mu_pred = np.array([10.0, 0.5])
        C_pred = np.eye(2)
        G = observation_matrix()
        observation = 12.0

        mu_new, C_new, _ = kalman_update(mu_pred, C_pred, G, 1e-15, observation)
        # Price component should snap to observation
        np.testing.assert_allclose(mu_new[0], observation, atol=1e-6)
        # Price variance should collapse to ~0
        assert C_new[0, 0] < 1e-10

    def test_infinite_observation_noise(self):
        """With huge sigma_obs, posterior equals prior (observation ignored)."""
        mu_pred = np.array([10.0, 0.5])
        C_pred = np.array([[1.0, 0.2], [0.2, 0.5]])
        G = observation_matrix()

        mu_new, C_new, _ = kalman_update(mu_pred, C_pred, G, 1e12, 999.0)
        np.testing.assert_allclose(mu_new, mu_pred, atol=1e-6)
        np.testing.assert_allclose(C_new, C_pred, atol=1e-4)

    def test_kalman_gain_between_zero_and_one(self):
        """For scalar observation, effective gain K[0] must be in [0, 1]."""
        mu_pred = np.array([5.0, 0.0])
        C_pred = np.array([[2.0, 0.0], [0.0, 1.0]])
        G = observation_matrix()
        sigma_obs_sq = 1.0

        # Manually compute gain for price component
        S = C_pred[0, 0] + sigma_obs_sq  # = 3.0
        K0 = C_pred[0, 0] / S  # = 2/3
        assert 0.0 < K0 < 1.0

        mu_new, _, _ = kalman_update(mu_pred, C_pred, G, sigma_obs_sq, 8.0)
        # mu_new[0] = mu_pred[0] + K0 * (8 - 5) = 5 + 2 = 7
        np.testing.assert_allclose(mu_new[0], 5.0 + K0 * 3.0, atol=1e-12)

    def test_log_likelihood_gaussian(self):
        """PED log-likelihood matches direct Gaussian log-pdf evaluation."""
        mu_pred = np.array([10.0, 0.5])
        C_pred = np.array([[2.0, 0.3], [0.3, 1.0]])
        G = observation_matrix()
        sigma_obs_sq = 0.5
        y = 11.5

        _, _, log_lik = kalman_update(mu_pred, C_pred, G, sigma_obs_sq, y)

        # Direct computation
        y_pred = G @ mu_pred  # = 10.0
        S = (G @ C_pred @ G.T).item() + sigma_obs_sq  # = 2.0 + 0.5 = 2.5
        expected = -0.5 * np.log(2 * np.pi * S) - 0.5 * (y - y_pred.item())**2 / S
        np.testing.assert_allclose(log_lik, expected, atol=1e-12)

    def test_output_covariance_symmetric_psd(self):
        """Filtered covariance must be symmetric and PSD."""
        mu_pred = np.array([3.0, -0.2])
        C_pred = np.array([[1.5, 0.4], [0.4, 0.8]])
        G = observation_matrix()

        _, C_new, _ = kalman_update(mu_pred, C_pred, G, 0.3, 3.5)
        np.testing.assert_allclose(C_new, C_new.T, atol=1e-14)
        eigenvalues = np.linalg.eigvalsh(C_new)
        assert np.all(eigenvalues >= -1e-12)

    def test_covariance_shrinks(self):
        """Filtering must reduce uncertainty: trace(C_new) < trace(C_pred)."""
        mu_pred = np.array([5.0, 0.1])
        C_pred = np.array([[2.0, 0.5], [0.5, 1.0]])
        G = observation_matrix()

        _, C_new, _ = kalman_update(mu_pred, C_pred, G, 0.5, 5.5)
        assert np.trace(C_new) < np.trace(C_pred)


# ---------- kalman_filter tests ----------


class TestKalmanFilter:
    """Tests for the full Kalman filter forward pass."""

    @pytest.fixture
    def synthetic_langevin_data(self):
        """Generate synthetic data from a known Langevin model (no jumps)."""
        theta, sigma, dt = -0.2, 0.05, 1.0
        sigma_obs = 0.3
        T = 200
        rng = np.random.default_rng(42)

        F, Q = discretize_langevin(theta, sigma, dt)
        G = observation_matrix()

        # True states
        states = np.zeros((T, 2))
        states[0] = [100.0, 0.0]
        for t in range(1, T):
            w = rng.multivariate_normal([0, 0], Q)
            states[t] = F @ states[t - 1] + w

        # Observations
        obs = (G @ states.T).ravel() + rng.normal(0, sigma_obs, T)

        return {
            "observations": obs,
            "true_states": states,
            "F": F, "Q": Q, "G": G,
            "sigma_obs_sq": sigma_obs**2,
            "mu0": np.array([100.0, 0.0]),
            "C0": np.eye(2) * 10.0,
        }

    def test_output_shapes(self, synthetic_langevin_data):
        """All outputs have correct shapes."""
        d = synthetic_langevin_data
        T = len(d["observations"])

        pred_m, pred_c, filt_m, filt_c, lls, total_ll = kalman_filter(
            d["observations"], d["F"], d["Q"], d["G"],
            d["sigma_obs_sq"], d["mu0"], d["C0"],
        )

        assert pred_m.shape == (T, 2)
        assert pred_c.shape == (T, 2, 2)
        assert filt_m.shape == (T, 2)
        assert filt_c.shape == (T, 2, 2)
        assert lls.shape == (T,)
        assert isinstance(total_ll, float)

    def test_total_ll_equals_sum(self, synthetic_langevin_data):
        """Total log-likelihood equals sum of per-step PED terms."""
        d = synthetic_langevin_data
        _, _, _, _, lls, total_ll = kalman_filter(
            d["observations"], d["F"], d["Q"], d["G"],
            d["sigma_obs_sq"], d["mu0"], d["C0"],
        )
        np.testing.assert_allclose(total_ll, np.sum(lls), atol=1e-10)

    def test_filtered_tracks_truth(self, synthetic_langevin_data):
        """Filtered price mean tracks true price within 2 std for >90% of timesteps."""
        d = synthetic_langevin_data
        _, _, filt_m, filt_c, _, _ = kalman_filter(
            d["observations"], d["F"], d["Q"], d["G"],
            d["sigma_obs_sq"], d["mu0"], d["C0"],
        )

        # Skip first 20 steps (burn-in from vague prior)
        price_errors = np.abs(filt_m[20:, 0] - d["true_states"][20:, 0])
        price_stds = np.sqrt(filt_c[20:, 0, 0])
        within_2sigma = price_errors < 2.0 * price_stds
        fraction = np.mean(within_2sigma)
        assert fraction > 0.90, f"Only {fraction:.1%} within 2-sigma"

    def test_covariance_converges(self, synthetic_langevin_data):
        """Filtered covariance converges to steady state (last 50 steps are stable)."""
        d = synthetic_langevin_data
        _, _, _, filt_c, _, _ = kalman_filter(
            d["observations"], d["F"], d["Q"], d["G"],
            d["sigma_obs_sq"], d["mu0"], d["C0"],
        )

        # Trace of covariance should stabilize
        traces = np.array([np.trace(filt_c[t]) for t in range(len(d["observations"]))])
        # Last 50 traces should have very low variance (converged)
        assert np.std(traces[-50:]) < 0.01 * np.mean(traces[-50:])

    def test_covariances_stay_psd(self, synthetic_langevin_data):
        """All filtered covariances must be PSD throughout the sequence."""
        d = synthetic_langevin_data
        _, _, _, filt_c, _, _ = kalman_filter(
            d["observations"], d["F"], d["Q"], d["G"],
            d["sigma_obs_sq"], d["mu0"], d["C0"],
        )

        for t in range(len(d["observations"])):
            eigenvalues = np.linalg.eigvalsh(filt_c[t])
            assert np.all(eigenvalues >= -1e-12), f"Non-PSD at t={t}"

    def test_single_observation(self):
        """Filter handles T=1 correctly."""
        F, Q = discretize_langevin(-0.5, 0.1, 1.0)
        G = observation_matrix()
        mu0 = np.array([10.0, 0.0])
        C0 = np.eye(2) * 5.0

        pred_m, _, filt_m, _, lls, total_ll = kalman_filter(
            np.array([10.5]), F, Q, G, 0.5, mu0, C0,
        )
        assert pred_m.shape == (1, 2)
        assert filt_m.shape == (1, 2)
        np.testing.assert_allclose(total_ll, lls[0])
