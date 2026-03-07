import numpy as np
import pytest


@pytest.fixture(autouse=True)
def deterministic_numpy_seed():
    """Reset NumPy global RNG before each test for deterministic behavior."""
    np.random.seed(0)


@pytest.fixture
def sample_hmm():
    """Sample synthetic states/observations from a Gaussian-emission HMM."""

    def _sample_hmm(T, A, pi, mu, sigma2, seed=0):
        rng = np.random.default_rng(seed)
        K = A.shape[0]
        states = np.empty(T, dtype=int)
        obs = np.empty(T, dtype=float)

        states[0] = rng.choice(K, p=pi)
        obs[0] = rng.normal(mu[states[0]], np.sqrt(sigma2[states[0]]))
        for t in range(1, T):
            states[t] = rng.choice(K, p=A[states[t - 1]])
            obs[t] = rng.normal(mu[states[t]], np.sqrt(sigma2[states[t]]))
        return states, obs

    return _sample_hmm


def pytest_configure(config):
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*joblib will operate in serial mode.*:UserWarning",
    )
