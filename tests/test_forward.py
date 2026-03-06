import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM

from src.hmm.forward import forward


def test_forward_output_shapes_and_single_observation():
    observations = np.array([0.01], dtype=float)
    A = np.array([[0.7, 0.3], [0.2, 0.8]], dtype=float)
    pi = np.array([0.6, 0.4], dtype=float)
    mu = np.array([0.0, 0.02], dtype=float)
    sigma2 = np.array([0.01, 0.02], dtype=float)

    log_alpha, log_likelihood = forward(observations, A, pi, mu, sigma2)

    assert log_alpha.shape == (1, 2)
    assert np.isfinite(log_alpha).all()
    assert np.isfinite(log_likelihood)


def test_forward_matches_hmmlearn_log_likelihood():
    observations = np.array([0.01, -0.02, 0.015, 0.005, -0.01], dtype=float)
    A = np.array([[0.92, 0.08], [0.15, 0.85]], dtype=float)
    pi = np.array([0.55, 0.45], dtype=float)
    mu = np.array([0.0, 0.01], dtype=float)
    sigma2 = np.array([0.0004, 0.0009], dtype=float)

    _, ours_log_likelihood = forward(observations, A, pi, mu, sigma2)

    model = GaussianHMM(
        n_components=2,
        covariance_type="diag",
        init_params="",
        params="",
    )
    model.startprob_ = pi
    model.transmat_ = A
    model.means_ = mu.reshape(-1, 1)
    model.covars_ = sigma2.reshape(-1, 1)
    hmmlearn_log_likelihood = model.score(observations.reshape(-1, 1))

    np.testing.assert_allclose(
        ours_log_likelihood, hmmlearn_log_likelihood, rtol=1e-6, atol=1e-6
    )


def test_forward_rejects_empty_observations():
    A = np.array([[1.0]], dtype=float)
    pi = np.array([1.0], dtype=float)
    mu = np.array([0.0], dtype=float)
    sigma2 = np.array([1.0], dtype=float)
    with pytest.raises(ValueError, match="non-empty 1D"):
        forward(np.array([]), A, pi, mu, sigma2)


def test_forward_rejects_invalid_transition_rows():
    observations = np.array([0.1, 0.2], dtype=float)
    A = np.array([[0.8, 0.3], [0.2, 0.8]], dtype=float)
    pi = np.array([0.5, 0.5], dtype=float)
    mu = np.array([0.0, 0.1], dtype=float)
    sigma2 = np.array([1.0, 1.0], dtype=float)
    with pytest.raises(ValueError, match="rows of A must sum to 1"):
        forward(observations, A, pi, mu, sigma2)


def test_forward_k_equals_one_case():
    observations = np.array([0.1, 0.0, -0.1], dtype=float)
    A = np.array([[1.0]], dtype=float)
    pi = np.array([1.0], dtype=float)
    mu = np.array([0.0], dtype=float)
    sigma2 = np.array([0.5], dtype=float)

    log_alpha, log_likelihood = forward(observations, A, pi, mu, sigma2)

    assert log_alpha.shape == (3, 1)
    assert np.isfinite(log_likelihood)
