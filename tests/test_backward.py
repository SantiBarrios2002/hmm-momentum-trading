import numpy as np
import pytest
from scipy.special import logsumexp

from src.hmm.backward import backward
from src.hmm.forward import forward


def test_backward_output_shape_and_terminal_condition():
    observations = np.array([0.01, -0.02, 0.03], dtype=float)
    A = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=float)
    mu = np.array([0.0, 0.02], dtype=float)
    sigma2 = np.array([0.01, 0.02], dtype=float)

    log_beta = backward(observations, A, mu, sigma2)

    assert log_beta.shape == (3, 2)
    np.testing.assert_allclose(log_beta[-1], np.zeros(2), rtol=1e-12, atol=1e-12)


def test_backward_forward_likelihood_consistency():
    observations = np.array([0.01, -0.02, 0.015, 0.005], dtype=float)
    A = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float)
    pi = np.array([0.6, 0.4], dtype=float)
    mu = np.array([0.0, 0.01], dtype=float)
    sigma2 = np.array([0.0004, 0.0009], dtype=float)

    log_alpha, log_likelihood_forward = forward(observations, A, pi, mu, sigma2)
    log_beta = backward(observations, A, mu, sigma2)
    log_likelihood_alpha_beta = float(logsumexp(log_alpha[0] + log_beta[0]))

    np.testing.assert_allclose(
        log_likelihood_alpha_beta, log_likelihood_forward, rtol=1e-10, atol=1e-10
    )


def test_backward_rejects_empty_observations():
    A = np.array([[1.0]], dtype=float)
    mu = np.array([0.0], dtype=float)
    sigma2 = np.array([1.0], dtype=float)
    with pytest.raises(ValueError, match="non-empty 1D"):
        backward(np.array([]), A, mu, sigma2)


def test_backward_rejects_invalid_transition_rows():
    observations = np.array([0.1, 0.2], dtype=float)
    A = np.array([[0.7, 0.4], [0.2, 0.8]], dtype=float)
    mu = np.array([0.0, 0.1], dtype=float)
    sigma2 = np.array([1.0, 1.0], dtype=float)
    with pytest.raises(ValueError, match="rows of A must sum to 1"):
        backward(observations, A, mu, sigma2)


def test_backward_k_equals_one_case():
    observations = np.array([0.1, 0.0, -0.1], dtype=float)
    A = np.array([[1.0]], dtype=float)
    mu = np.array([0.0], dtype=float)
    sigma2 = np.array([0.5], dtype=float)

    log_beta = backward(observations, A, mu, sigma2)

    assert log_beta.shape == (3, 1)
    np.testing.assert_allclose(log_beta[-1], np.array([0.0]), rtol=1e-12, atol=1e-12)
