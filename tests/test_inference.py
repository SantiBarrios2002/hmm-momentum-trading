import numpy as np
from scipy.special import logsumexp

from src.hmm.forward import forward
from src.hmm.inference import predict_update_step, run_inference


def test_predict_update_step_outputs_valid_distribution():
    omega_prev = np.array([0.6, 0.4], dtype=float)
    A = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float)
    mu = np.array([0.0, 0.01], dtype=float)
    sigma2 = np.array([0.0004, 0.0009], dtype=float)

    omega_new, prediction = predict_update_step(
        omega_prev, A, mu, sigma2, observation=0.005
    )

    assert omega_new.shape == (2,)
    np.testing.assert_allclose(omega_new.sum(), 1.0, rtol=1e-12, atol=1e-12)
    assert np.all(omega_new >= 0.0)
    assert np.isfinite(prediction)


def test_run_inference_matches_forward_filtering_distribution():
    observations = np.array([0.01, -0.02, 0.015, 0.005], dtype=float)
    A = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float)
    pi = np.array([0.6, 0.4], dtype=float)
    mu = np.array([0.0, 0.01], dtype=float)
    sigma2 = np.array([0.0004, 0.0009], dtype=float)

    _, state_probs_online = run_inference(observations, A, pi, mu, sigma2)
    log_alpha, _ = forward(observations, A, pi, mu, sigma2)
    state_probs_batch = np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))

    np.testing.assert_allclose(
        state_probs_online, state_probs_batch, rtol=1e-10, atol=1e-10
    )


def test_run_inference_shapes():
    observations = np.array([0.01, 0.0, -0.01], dtype=float)
    A = np.array([[1.0]], dtype=float)
    pi = np.array([1.0], dtype=float)
    mu = np.array([0.0], dtype=float)
    sigma2 = np.array([1.0], dtype=float)

    predictions, state_probs = run_inference(observations, A, pi, mu, sigma2)

    assert predictions.shape == (3,)
    assert state_probs.shape == (3, 1)
    np.testing.assert_allclose(state_probs, np.ones((3, 1)), atol=1e-12)
