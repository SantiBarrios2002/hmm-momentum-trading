import numpy as np

from src.data.features import log_returns, normalize_returns
from src.hmm.baum_welch import baum_welch
from src.hmm.forward import forward
from src.hmm.forward_backward import compute_posteriors
from src.hmm.inference import run_inference


def _synthetic_prices(n=300):
    t = np.arange(n, dtype=float)
    # Smooth trend + oscillation to produce non-degenerate returns.
    return 100.0 + 0.03 * t + 0.8 * np.sin(t / 18.0)


def test_phase1_features_to_forward_pipeline():
    prices = _synthetic_prices(300)
    returns = log_returns(prices)
    z = normalize_returns(returns, window=20)
    observations = z[~np.isnan(z)]

    A = np.array([[0.92, 0.08], [0.10, 0.90]], dtype=float)
    pi = np.array([0.5, 0.5], dtype=float)
    mu = np.array([-0.2, 0.2], dtype=float)
    sigma2 = np.array([0.4, 0.4], dtype=float)

    log_alpha, log_likelihood = forward(observations, A, pi, mu, sigma2)

    assert observations.ndim == 1 and observations.size > 0
    assert log_alpha.shape == (observations.size, 2)
    assert np.isfinite(log_alpha).all()
    assert np.isfinite(log_likelihood)


def test_phase1_features_to_forward_backward_posteriors():
    prices = _synthetic_prices(320)
    returns = log_returns(prices)
    observations = normalize_returns(returns, window=15)
    observations = observations[~np.isnan(observations)]

    A = np.array([[0.9, 0.1], [0.15, 0.85]], dtype=float)
    pi = np.array([0.6, 0.4], dtype=float)
    mu = np.array([-0.1, 0.1], dtype=float)
    sigma2 = np.array([0.5, 0.5], dtype=float)

    gamma, xi, _ = compute_posteriors(observations, A, pi, mu, sigma2)

    np.testing.assert_allclose(gamma.sum(axis=1), np.ones(gamma.shape[0]), atol=1e-10)
    np.testing.assert_allclose(
        xi.sum(axis=(1, 2)), np.ones(xi.shape[0]), atol=1e-10
    )
    assert gamma.shape[0] == observations.size
    assert xi.shape[0] == observations.size - 1


def test_phase1_features_to_baum_welch_and_inference():
    prices = _synthetic_prices(420)
    returns = log_returns(prices)
    observations = normalize_returns(returns, window=20)
    observations = observations[~np.isnan(observations)]

    params, history, gamma = baum_welch(
        observations,
        K=2,
        max_iter=35,
        tol=1e-6,
        n_restarts=2,
        random_state=17,
    )
    predictions, state_probs = run_inference(
        observations,
        params["A"],
        params["pi"],
        params["mu"],
        params["sigma2"],
    )

    assert len(history) >= 1
    assert np.isfinite(history).all()
    assert gamma.shape == (observations.size, 2)
    assert predictions.shape == (observations.size,)
    assert state_probs.shape == (observations.size, 2)
    np.testing.assert_allclose(state_probs.sum(axis=1), np.ones(observations.size), atol=1e-10)
