import numpy as np
import pytest

from src.hmm.forward import forward
from src.hmm.forward_backward import compute_posteriors


def test_compute_posteriors_shapes_and_normalization():
    observations = np.array([0.01, -0.02, 0.015, 0.005], dtype=float)
    A = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float)
    pi = np.array([0.6, 0.4], dtype=float)
    mu = np.array([0.0, 0.01], dtype=float)
    sigma2 = np.array([0.0004, 0.0009], dtype=float)

    gamma, xi, log_likelihood = compute_posteriors(observations, A, pi, mu, sigma2)

    assert gamma.shape == (4, 2)
    assert xi.shape == (3, 2, 2)
    np.testing.assert_allclose(gamma.sum(axis=1), np.ones(4), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        xi.sum(axis=(1, 2)), np.ones(3), rtol=1e-12, atol=1e-12
    )
    assert np.isfinite(log_likelihood)


def test_compute_posteriors_matches_forward_likelihood():
    observations = np.array([0.02, -0.01, 0.01], dtype=float)
    A = np.array([[0.85, 0.15], [0.10, 0.90]], dtype=float)
    pi = np.array([0.5, 0.5], dtype=float)
    mu = np.array([-0.005, 0.01], dtype=float)
    sigma2 = np.array([0.001, 0.002], dtype=float)

    _, forward_ll = forward(observations, A, pi, mu, sigma2)
    _, _, fb_ll = compute_posteriors(observations, A, pi, mu, sigma2)

    np.testing.assert_allclose(fb_ll, forward_ll, rtol=1e-12, atol=1e-12)


def test_compute_posteriors_rejects_empty_observations():
    with pytest.raises(ValueError, match="non-empty 1D"):
        compute_posteriors(
            np.array([]),
            np.array([[1.0]]),
            np.array([1.0]),
            np.array([0.0]),
            np.array([1.0]),
        )
