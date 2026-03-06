import numpy as np

from src.hmm.baum_welch import baum_welch, m_step


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


def test_m_step_outputs_valid_parameters():
    observations = np.array([0.0, 1.0, -0.5], dtype=float)
    gamma = np.array(
        [[0.7, 0.3], [0.4, 0.6], [0.5, 0.5]],
        dtype=float,
    )
    xi = np.array(
        [
            [[0.3, 0.4], [0.1, 0.2]],
            [[0.2, 0.2], [0.3, 0.3]],
        ],
        dtype=float,
    )

    A_new, pi_new, mu_new, sigma2_new = m_step(observations, gamma, xi)

    assert A_new.shape == (2, 2)
    assert pi_new.shape == (2,)
    assert mu_new.shape == (2,)
    assert sigma2_new.shape == (2,)
    np.testing.assert_allclose(A_new.sum(axis=1), np.ones(2), atol=1e-12)
    np.testing.assert_allclose(pi_new.sum(), 1.0, atol=1e-12)
    assert np.all(sigma2_new > 0.0)


def test_baum_welch_log_likelihood_is_monotone():
    A_true = np.array([[0.95, 0.05], [0.08, 0.92]], dtype=float)
    pi_true = np.array([0.6, 0.4], dtype=float)
    mu_true = np.array([-0.01, 0.02], dtype=float)
    sigma2_true = np.array([0.0003, 0.0008], dtype=float)
    _, observations = _sample_hmm(
        T=300, A=A_true, pi=pi_true, mu=mu_true, sigma2=sigma2_true, seed=7
    )

    _, history, _ = baum_welch(
        observations,
        K=2,
        max_iter=40,
        tol=1e-8,
        n_restarts=1,
        random_state=11,
    )
    diffs = np.diff(history)
    assert np.all(diffs >= -1e-8)


def test_baum_welch_recovers_emission_means_synthetically():
    A_true = np.array([[0.97, 0.03], [0.05, 0.95]], dtype=float)
    pi_true = np.array([0.5, 0.5], dtype=float)
    mu_true = np.array([-0.02, 0.025], dtype=float)
    sigma2_true = np.array([0.0002, 0.0004], dtype=float)
    _, observations = _sample_hmm(
        T=1200, A=A_true, pi=pi_true, mu=mu_true, sigma2=sigma2_true, seed=42
    )

    params, _, _ = baum_welch(
        observations,
        K=2,
        max_iter=80,
        tol=1e-6,
        n_restarts=3,
        random_state=123,
    )

    estimated_mu_sorted = np.sort(params["mu"])
    true_mu_sorted = np.sort(mu_true)
    np.testing.assert_allclose(estimated_mu_sorted, true_mu_sorted, atol=0.01)
