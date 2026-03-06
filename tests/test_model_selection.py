import numpy as np

from src.hmm.model_selection import compute_aic, compute_bic, select_K


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


def test_compute_aic_formula():
    ll = -100.0
    K = 3
    p = K * (K - 1) + 2 * K + (K - 1)
    expected = -2 * ll + 2 * p
    assert compute_aic(ll, K) == expected


def test_compute_bic_formula():
    ll = -100.0
    K = 3
    n = 500
    p = K * (K - 1) + 2 * K + (K - 1)
    expected = -2 * ll + p * np.log(n)
    np.testing.assert_allclose(compute_bic(ll, K, n), expected, rtol=1e-12, atol=1e-12)


def test_select_K_recovers_true_state_count_on_synthetic_data():
    A_true = np.array([[0.97, 0.03], [0.04, 0.96]], dtype=float)
    pi_true = np.array([0.5, 0.5], dtype=float)
    mu_true = np.array([-0.02, 0.02], dtype=float)
    sigma2_true = np.array([0.0003, 0.0003], dtype=float)
    _, observations = _sample_hmm(
        T=900, A=A_true, pi=pi_true, mu=mu_true, sigma2=sigma2_true, seed=123
    )

    result = select_K(
        observations,
        K_range=[1, 2, 3],
        criterion="bic",
        max_iter=60,
        n_restarts=2,
        random_state=42,
    )

    assert result["best_K"] == 2
    assert set(result["scores"].keys()) == {1, 2, 3}
