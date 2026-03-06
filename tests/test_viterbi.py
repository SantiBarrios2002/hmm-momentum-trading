import numpy as np
import pytest

from src.hmm.viterbi import viterbi


def test_viterbi_known_dominant_state_sequence():
    observations = np.array([-0.11, -0.09, 0.10, 0.12], dtype=float)
    A = np.array([[0.995, 0.005], [0.005, 0.995]], dtype=float)
    pi = np.array([0.5, 0.5], dtype=float)
    mu = np.array([-0.10, 0.10], dtype=float)
    sigma2 = np.array([1e-4, 1e-4], dtype=float)

    states, log_prob = viterbi(observations, A, pi, mu, sigma2)

    np.testing.assert_array_equal(states, np.array([0, 0, 1, 1]))
    assert np.isfinite(log_prob)


def test_viterbi_k_equals_one_case():
    observations = np.array([0.01, -0.02, 0.03], dtype=float)
    A = np.array([[1.0]], dtype=float)
    pi = np.array([1.0], dtype=float)
    mu = np.array([0.0], dtype=float)
    sigma2 = np.array([1.0], dtype=float)

    states, log_prob = viterbi(observations, A, pi, mu, sigma2)

    np.testing.assert_array_equal(states, np.array([0, 0, 0]))
    assert np.isfinite(log_prob)


def test_viterbi_rejects_empty_observations():
    with pytest.raises(ValueError, match="non-empty 1D"):
        viterbi(
            np.array([]),
            np.array([[1.0]]),
            np.array([1.0]),
            np.array([0.0]),
            np.array([1.0]),
        )
