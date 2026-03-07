import numpy as np

from src.hmm.gibbs import (
    gibbs_sampler,
    sample_emission_means,
    sample_emission_variances,
    sample_initial_distribution,
    sample_states_ffbs,
    sample_transition_matrix,
)


def test_sample_states_ffbs_returns_valid_state_sequence():
    observations = np.array([0.01, -0.02, 0.015, 0.005, -0.01], dtype=float)
    A = np.array([[0.92, 0.08], [0.15, 0.85]], dtype=float)
    pi = np.array([0.7, 0.3], dtype=float)
    mu = np.array([-0.01, 0.02], dtype=float)
    sigma2 = np.array([0.0003, 0.0005], dtype=float)

    states = sample_states_ffbs(
        observations,
        A,
        pi,
        mu,
        sigma2,
        rng=np.random.default_rng(7),
    )

    assert states.shape == (observations.size,)
    assert states.dtype == np.int64
    assert np.all((states >= 0) & (states < 2))


def test_conjugate_conditionals_return_valid_parameters():
    observations = np.array([0.01, -0.015, 0.02, 0.018, -0.01, 0.004], dtype=float)
    states = np.array([1, 0, 1, 1, 0, 1], dtype=int)
    sigma2 = np.array([0.0004, 0.0006], dtype=float)
    rng = np.random.default_rng(11)

    pi = sample_initial_distribution(states, alpha_prior=1.0, rng=rng)
    A = sample_transition_matrix(states, K=2, alpha_prior=1.0, rng=rng)
    mu = sample_emission_means(
        observations, states, sigma2, mu0=0.0, tau2=0.01, rng=rng
    )
    sigma2_new = sample_emission_variances(
        observations,
        states,
        mu,
        alpha0=2.0,
        beta0=1e-4,
        min_variance=1e-8,
        rng=rng,
    )

    assert pi.shape == (2,)
    np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-12)
    assert np.all(pi > 0.0)

    assert A.shape == (2, 2)
    np.testing.assert_allclose(A.sum(axis=1), np.ones(2), atol=1e-12)
    assert np.all(A > 0.0)

    assert mu.shape == (2,)
    assert np.all(np.isfinite(mu))

    assert sigma2_new.shape == (2,)
    assert np.all(sigma2_new > 0.0)


def test_gibbs_sampler_recovers_emission_means_on_synthetic_data(sample_hmm):
    A_true = np.array([[0.96, 0.04], [0.05, 0.95]], dtype=float)
    pi_true = np.array([0.55, 0.45], dtype=float)
    mu_true = np.array([-0.02, 0.025], dtype=float)
    sigma2_true = np.array([0.00025, 0.0005], dtype=float)

    _, observations = sample_hmm(
        T=1000,
        A=A_true,
        pi=pi_true,
        mu=mu_true,
        sigma2=sigma2_true,
        seed=123,
    )

    samples = gibbs_sampler(
        observations,
        K=2,
        n_samples=120,
        burn_in=120,
        thin=2,
        random_state=9,
        alpha_pi=1.0,
        alpha_A=1.0,
        mu0=0.0,
        tau2=0.01,
        sigma2_alpha0=2.0,
        sigma2_beta0=1e-4,
    )

    posterior_mu = np.sort(samples["posterior_mean"]["mu"])
    np.testing.assert_allclose(posterior_mu, np.sort(mu_true), atol=0.01)
