"""Gibbs sampling for Gaussian-emission Hidden Markov Models."""

import numpy as np
from scipy.special import logsumexp

from src.hmm.forward import forward


def sample_states_ffbs(observations, A, pi, mu, sigma2, rng=None):
    """
    Sample a full hidden-state trajectory via FFBS (Paper §6 style filtering).

    Forward pass:
        Compute log alpha_t(k) = log p(y_1:t, m_t = k) using forward().

    Backward sampling:
        1) Draw m_T ~ p(m_T | y_1:T) proportional alpha_T(m_T)
        2) For t = T-1..1, draw:
           p(m_t = i | m_{t+1} = j, y_1:T) proportional alpha_t(i) * A[i, j]

    Parameters:
        observations: np.ndarray, shape (T,)
            Observed sequence y_1, ..., y_T.
        A: np.ndarray, shape (K, K)
            Transition matrix.
        pi: np.ndarray, shape (K,)
            Initial state probabilities.
        mu: np.ndarray, shape (K,)
            Emission means.
        sigma2: np.ndarray, shape (K,)
            Emission variances.
        rng: np.random.Generator or None
            Random generator. If None, a fresh default generator is used.

    Returns:
        states: np.ndarray, shape (T,)
            Sampled hidden-state sequence in {0, ..., K-1}.
    """
    if rng is None:
        rng = np.random.default_rng()

    observations = np.asarray(observations, dtype=float)
    A = np.asarray(A, dtype=float)
    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)

    log_alpha, _ = forward(observations, A, pi, mu, sigma2)
    log_A = np.log(A)

    T, K = log_alpha.shape
    states = np.empty(T, dtype=int)

    # Sample m_T from p(m_T | y_1:T)
    log_w = log_alpha[T - 1] - logsumexp(log_alpha[T - 1])
    states[T - 1] = int(rng.choice(K, p=np.exp(log_w)))

    # Sample m_t conditioned on sampled m_{t+1}
    for t in range(T - 2, -1, -1):
        next_state = states[t + 1]
        log_w = log_alpha[t] + log_A[:, next_state]
        log_w -= logsumexp(log_w)
        states[t] = int(rng.choice(K, p=np.exp(log_w)))

    return states


def sample_initial_distribution(states, alpha_prior=1.0, rng=None):
    """
    Sample initial distribution pi from its Dirichlet posterior.

    Posterior:
        pi | S ~ Dirichlet(alpha_prior + e_{S_1})

    Parameters:
        states: np.ndarray, shape (T,)
            Hidden-state sequence.
        alpha_prior: float or np.ndarray, shape (K,)
            Dirichlet prior concentration(s), strictly positive.
        rng: np.random.Generator or None
            Random generator.

    Returns:
        pi: np.ndarray, shape (K,)
            Sampled initial distribution.
    """
    if rng is None:
        rng = np.random.default_rng()

    states = np.asarray(states, dtype=int)
    if states.ndim != 1 or states.size == 0:
        raise ValueError("states must be a non-empty 1D array")

    K = int(np.max(states)) + 1
    alpha = _normalize_dirichlet_prior(alpha_prior, shape=(K,))
    counts = np.zeros(K, dtype=float)
    counts[states[0]] = 1.0
    return rng.dirichlet(alpha + counts)


def sample_transition_matrix(states, K, alpha_prior=1.0, rng=None):
    """
    Sample transition matrix A from row-wise Dirichlet posteriors.

    Posterior for each row i:
        A[i, :] | S ~ Dirichlet(alpha_prior[i, :] + N_{i,:})
    where N_{i,j} counts transitions i -> j in S_1:T.

    Parameters:
        states: np.ndarray, shape (T,)
            Hidden-state sequence.
        K: int
            Number of hidden states.
        alpha_prior: float or np.ndarray, shape (K, K)
            Dirichlet prior concentration(s), strictly positive.
        rng: np.random.Generator or None
            Random generator.

    Returns:
        A: np.ndarray, shape (K, K)
            Sampled transition matrix with row sums equal to 1.
    """
    if rng is None:
        rng = np.random.default_rng()

    states = np.asarray(states, dtype=int)
    if states.ndim != 1 or states.size == 0:
        raise ValueError("states must be a non-empty 1D array")
    if K < 1:
        raise ValueError("K must be >= 1")

    alpha = _normalize_dirichlet_prior(alpha_prior, shape=(K, K))

    counts = np.zeros((K, K), dtype=float)
    for t in range(states.size - 1):
        counts[states[t], states[t + 1]] += 1.0

    A = np.empty((K, K), dtype=float)
    for i in range(K):
        A[i] = rng.dirichlet(alpha[i] + counts[i])
    return A


def sample_emission_means(
    observations,
    states,
    sigma2,
    mu0=0.0,
    tau2=1.0,
    rng=None,
):
    """
    Sample emission means mu_k from Normal conditional posteriors.

    Model:
        y_t | (m_t=k, mu_k, sigma2_k) ~ N(mu_k, sigma2_k)
        mu_k ~ N(mu0, tau2)

    Posterior:
        tau2_n = 1 / (1/tau2 + n_k/sigma2_k)
        mu_n = tau2_n * (mu0/tau2 + sum_{t:m_t=k} y_t / sigma2_k)
        mu_k | rest ~ N(mu_n, tau2_n)

    Parameters:
        observations: np.ndarray, shape (T,)
            Observed sequence.
        states: np.ndarray, shape (T,)
            Hidden-state sequence.
        sigma2: np.ndarray, shape (K,)
            Emission variances.
        mu0: float
            Prior mean for each mu_k.
        tau2: float
            Prior variance for each mu_k, strictly positive.
        rng: np.random.Generator or None
            Random generator.

    Returns:
        mu: np.ndarray, shape (K,)
            Sampled emission means.
    """
    if rng is None:
        rng = np.random.default_rng()

    observations = np.asarray(observations, dtype=float)
    states = np.asarray(states, dtype=int)
    sigma2 = np.asarray(sigma2, dtype=float)

    if observations.ndim != 1 or observations.size == 0:
        raise ValueError("observations must be a non-empty 1D array")
    if states.shape != observations.shape:
        raise ValueError("states must have the same shape as observations")
    if np.any(sigma2 <= 0.0):
        raise ValueError("sigma2 entries must be strictly positive")
    if tau2 <= 0.0:
        raise ValueError("tau2 must be strictly positive")

    K = sigma2.size
    mu = np.empty(K, dtype=float)
    inv_tau2 = 1.0 / float(tau2)

    for k in range(K):
        mask = states == k
        n_k = int(np.sum(mask))
        sum_y = float(np.sum(observations[mask]))

        precision_n = inv_tau2 + n_k / sigma2[k]
        var_n = 1.0 / precision_n
        mean_n = var_n * (mu0 * inv_tau2 + sum_y / sigma2[k])
        mu[k] = rng.normal(loc=mean_n, scale=np.sqrt(var_n))

    return mu


def sample_emission_variances(
    observations,
    states,
    mu,
    alpha0=2.0,
    beta0=1e-4,
    min_variance=1e-8,
    rng=None,
):
    """
    Sample emission variances sigma2_k from Inverse-Gamma conditionals.

    Model:
        y_t | (m_t=k, mu_k, sigma2_k) ~ N(mu_k, sigma2_k)
        sigma2_k ~ InvGamma(alpha0, beta0)

    Posterior:
        alpha_n = alpha0 + n_k / 2
        beta_n = beta0 + 0.5 * sum_{t:m_t=k} (y_t - mu_k)^2
        sigma2_k | rest ~ InvGamma(alpha_n, beta_n)

    Implementation detail:
        If X ~ InvGamma(a, b), then 1/X ~ Gamma(shape=a, rate=b).

    Parameters:
        observations: np.ndarray, shape (T,)
            Observed sequence.
        states: np.ndarray, shape (T,)
            Hidden-state sequence.
        mu: np.ndarray, shape (K,)
            Emission means.
        alpha0: float
            Inverse-Gamma shape prior, strictly positive.
        beta0: float
            Inverse-Gamma scale prior, strictly positive.
        min_variance: float
            Lower bound for sampled variances.
        rng: np.random.Generator or None
            Random generator.

    Returns:
        sigma2: np.ndarray, shape (K,)
            Sampled emission variances.
    """
    if rng is None:
        rng = np.random.default_rng()

    observations = np.asarray(observations, dtype=float)
    states = np.asarray(states, dtype=int)
    mu = np.asarray(mu, dtype=float)

    if observations.ndim != 1 or observations.size == 0:
        raise ValueError("observations must be a non-empty 1D array")
    if states.shape != observations.shape:
        raise ValueError("states must have the same shape as observations")
    if np.any(~np.isfinite(mu)):
        raise ValueError("mu must be finite")
    if alpha0 <= 0.0 or beta0 <= 0.0:
        raise ValueError("alpha0 and beta0 must be strictly positive")
    if min_variance <= 0.0:
        raise ValueError("min_variance must be strictly positive")

    K = mu.size
    sigma2 = np.empty(K, dtype=float)

    for k in range(K):
        mask = states == k
        n_k = int(np.sum(mask))
        sse = float(np.sum((observations[mask] - mu[k]) ** 2))

        alpha_n = alpha0 + 0.5 * n_k
        beta_n = beta0 + 0.5 * sse

        precision = rng.gamma(shape=alpha_n, scale=1.0 / beta_n)
        sigma2[k] = max(1.0 / precision, min_variance)

    return sigma2


def gibbs_sampler(
    observations,
    K,
    n_samples=200,
    burn_in=200,
    thin=2,
    random_state=None,
    alpha_pi=1.0,
    alpha_A=1.0,
    mu0=0.0,
    tau2=1.0,
    sigma2_alpha0=2.0,
    sigma2_beta0=1e-4,
    min_variance=1e-8,
):
    """
    Run blocked Gibbs sampling for a Gaussian-emission HMM.

    One Gibbs iteration:
        1) Sample hidden states S | A, pi, mu, sigma2, y   (FFBS)
        2) Sample pi | S                                   (Dirichlet)
        3) Sample A | S                                    (row-wise Dirichlet)
        4) Sample mu | S, sigma2, y                        (Normal)
        5) Sample sigma2 | S, mu, y                        (Inverse-Gamma)

    Samples are retained after burn-in and optional thinning.

    Parameters:
        observations: np.ndarray, shape (T,)
            Observed sequence.
        K: int
            Number of hidden states.
        n_samples: int
            Number of retained posterior samples.
        burn_in: int
            Number of initial iterations discarded.
        thin: int
            Keep one sample every `thin` iterations after burn-in.
        random_state: int or None
            Seed for deterministic sampling.
        alpha_pi: float or np.ndarray, shape (K,)
            Dirichlet prior for pi.
        alpha_A: float or np.ndarray, shape (K, K)
            Dirichlet prior for transition rows.
        mu0: float
            Prior mean for each mu_k.
        tau2: float
            Prior variance for mu_k.
        sigma2_alpha0: float
            Inverse-Gamma shape prior for sigma2_k.
        sigma2_beta0: float
            Inverse-Gamma scale prior for sigma2_k.
        min_variance: float
            Lower bound on sampled variances.

    Returns:
        samples: dict with keys
            "A_samples": np.ndarray, shape (n_samples, K, K)
            "pi_samples": np.ndarray, shape (n_samples, K)
            "mu_samples": np.ndarray, shape (n_samples, K)
            "sigma2_samples": np.ndarray, shape (n_samples, K)
            "state_samples": np.ndarray, shape (n_samples, T)
            "posterior_mean": dict with keys "A", "pi", "mu", "sigma2"
    """
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 1 or observations.size == 0:
        raise ValueError("observations must be a non-empty 1D array")
    if K < 1:
        raise ValueError("K must be >= 1")
    if n_samples < 1 or burn_in < 0 or thin < 1:
        raise ValueError("n_samples>=1, burn_in>=0 and thin>=1 are required")

    rng = np.random.default_rng(random_state)
    T = observations.size
    obs_mean = float(np.mean(observations))
    obs_var = float(np.var(observations) + 1e-8)

    # Random but stable initialization.
    A = rng.dirichlet(np.ones(K), size=K)
    pi = rng.dirichlet(np.ones(K))
    mu = obs_mean + rng.normal(scale=np.sqrt(obs_var), size=K)
    sigma2 = np.full(K, max(obs_var, min_variance), dtype=float)

    total_iters = burn_in + n_samples * thin
    A_samples = np.empty((n_samples, K, K), dtype=float)
    pi_samples = np.empty((n_samples, K), dtype=float)
    mu_samples = np.empty((n_samples, K), dtype=float)
    sigma2_samples = np.empty((n_samples, K), dtype=float)
    state_samples = np.empty((n_samples, T), dtype=int)

    save_idx = 0
    for it in range(total_iters):
        states = sample_states_ffbs(observations, A, pi, mu, sigma2, rng=rng)
        pi = sample_initial_distribution(states, alpha_prior=alpha_pi, rng=rng)
        A = sample_transition_matrix(states, K, alpha_prior=alpha_A, rng=rng)
        mu = sample_emission_means(
            observations, states, sigma2, mu0=mu0, tau2=tau2, rng=rng
        )
        sigma2 = sample_emission_variances(
            observations,
            states,
            mu,
            alpha0=sigma2_alpha0,
            beta0=sigma2_beta0,
            min_variance=min_variance,
            rng=rng,
        )

        # State-label ordering by ascending mean to reduce label switching.
        order = np.argsort(mu)
        inv_order = np.empty(K, dtype=int)
        inv_order[order] = np.arange(K)
        states = inv_order[states]
        A = A[np.ix_(order, order)]
        pi = pi[order]
        mu = mu[order]
        sigma2 = sigma2[order]

        if it >= burn_in and ((it - burn_in) % thin == 0):
            A_samples[save_idx] = A
            pi_samples[save_idx] = pi
            mu_samples[save_idx] = mu
            sigma2_samples[save_idx] = sigma2
            state_samples[save_idx] = states
            save_idx += 1

    posterior_mean = {
        "A": np.mean(A_samples, axis=0),
        "pi": np.mean(pi_samples, axis=0),
        "mu": np.mean(mu_samples, axis=0),
        "sigma2": np.mean(sigma2_samples, axis=0),
    }

    return {
        "A_samples": A_samples,
        "pi_samples": pi_samples,
        "mu_samples": mu_samples,
        "sigma2_samples": sigma2_samples,
        "state_samples": state_samples,
        "posterior_mean": posterior_mean,
    }


def _normalize_dirichlet_prior(alpha_prior, shape):
    """Return positive Dirichlet prior array with given shape."""
    alpha = np.asarray(alpha_prior, dtype=float)
    if alpha.ndim == 0:
        alpha = np.full(shape, float(alpha), dtype=float)
    if alpha.shape != shape:
        raise ValueError(f"alpha_prior must have shape {shape} or be scalar")
    if np.any(alpha <= 0.0):
        raise ValueError("Dirichlet prior concentrations must be strictly positive")
    return alpha
