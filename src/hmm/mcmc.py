"""MCMC HMM with Metropolis-Hastings — Paper §3.3.

The paper uses MH to sample from the posterior p(Θ|ΔY) ∝ p(ΔY|Θ) · p(Θ),
then selects the MAP estimate as a point estimate:

    Θ* = argmax_Θ [log p(ΔY|Θ) + log p(Θ)]

From the paper (§3.3):
    "Our implementation of MCMC approximates the posterior mode by keeping
     the sample with the highest posterior probability."

Prior specification (§3.3):
    - Each row of A ~ Dirichlet(e_{i1}, ..., e_{iK}) with e_{ii}=4,
      e_{ij}=1/(K-1) for i≠j. This makes the HMM "bounded away from
      a finite mixture model."
    - π drawn from the ergodic distribution of A.
    - Hierarchical prior on σ²_k (state-specific variances).
    - μ_k ~ N(0, σ²_prior) broad prior centered at zero.

The paper finds K=3 via bridge sampling of marginal likelihoods (Figure 3).
MCMC underperformed Baum-Welch in practice (Figure 8), likely due to
prior sensitivity and poor mixing.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp

from src.hmm.forward import forward


def mcmc_hmm(
    observations: np.ndarray,
    K: int,
    *,
    n_samples: int = 10000,
    burn_in: int = 2000,
    proposal_scale: float = 0.01,
    random_state: int = 42,
    min_variance: float = 1e-8,
) -> tuple[dict, dict, float]:
    """MCMC HMM parameter estimation via Metropolis-Hastings (Paper §3.3).

    Samples from the posterior:
        p(Θ | ΔY) ∝ p(ΔY | Θ) · p(Θ)

    where p(ΔY | Θ) is the HMM likelihood (computed via the forward algorithm)
    and p(Θ) is the prior defined below.

    The MH accept/reject step at iteration n:
        1. Propose Θ' ~ q(Θ' | Θ^(n))     (symmetric random walk)
        2. Compute α = min(1, p(Θ'|ΔY) / p(Θ^(n)|ΔY))
                      = min(1, exp(log p(ΔY|Θ') + log p(Θ') - log p(ΔY|Θ^(n)) - log p(Θ^(n))))
        3. Accept Θ' with probability α, else keep Θ^(n)

    MAP estimate: Θ* = argmax_{n > burn_in} [log p(ΔY|Θ^(n)) + log p(Θ^(n))]

    Parameters
    ----------
    observations : np.ndarray, shape (T,)
        Log-returns Δy_1, ..., Δy_T.
    K : int
        Number of hidden states (Paper §3.3: K=3 via bridge sampling).
    n_samples : int
        Total MCMC draws (including burn-in).
    burn_in : int
        Discard first burn_in samples.
    proposal_scale : float
        Standard deviation of the symmetric random-walk proposal.
    random_state : int
        Seed for reproducibility.
    min_variance : float
        Floor for σ²_k to prevent degenerate proposals.

    Returns
    -------
    params : dict with keys "A", "pi", "mu", "sigma2"
        MAP estimate (sample with highest log-posterior after burn-in).
    samples : dict with keys:
        "mu" : (n_samples, K) — posterior draws for emission means.
        "sigma2" : (n_samples, K) — posterior draws for emission variances.
        "log_posterior" : (n_samples,) — log-posterior at each draw.
    acceptance_rate : float
        Fraction of proposals accepted.
    """
    if len(observations) < 2:
        raise ValueError(f"Need at least 2 observations, got {len(observations)}")
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if burn_in < 0 or burn_in >= n_samples:
        raise ValueError(
            f"burn_in must be in [0, n_samples), got {burn_in}"
        )

    rng = np.random.default_rng(random_state)
    obs = np.asarray(observations, dtype=np.float64)

    # ── Initialize from data statistics ───────────────────────────────────
    mu_init = np.linspace(obs.min(), obs.max(), K)
    sigma2_init = np.full(K, max(np.var(obs), min_variance))
    A_init = _sample_dirichlet_rows(K, rng)
    pi_init = np.full(K, 1.0 / K)

    # Current state
    mu_cur = mu_init.copy()
    sigma2_cur = sigma2_init.copy()
    A_cur = A_init.copy()
    pi_cur = pi_init.copy()

    log_lik_cur = _safe_log_likelihood(obs, A_cur, pi_cur, mu_cur, sigma2_cur)
    log_prior_cur = _log_prior(A_cur, mu_cur, sigma2_cur, K)
    log_post_cur = log_lik_cur + log_prior_cur

    # Storage
    mu_samples = np.empty((n_samples, K))
    sigma2_samples = np.empty((n_samples, K))
    log_posterior_samples = np.empty(n_samples)
    n_accepted = 0

    # MAP tracking
    best_log_post = -np.inf
    best_params = None

    for n in range(n_samples):
        # ── Propose new parameters ────────────────────────────────────
        mu_prop = mu_cur + rng.normal(0, proposal_scale, K)
        sigma2_prop = sigma2_cur * np.exp(rng.normal(0, proposal_scale, K))
        sigma2_prop = np.maximum(sigma2_prop, min_variance)
        A_prop = _propose_transition_matrix(A_cur, proposal_scale, rng)
        pi_prop = _ergodic_distribution(A_prop)

        # ── Compute acceptance ratio ──────────────────────────────────
        log_lik_prop = _safe_log_likelihood(obs, A_prop, pi_prop, mu_prop, sigma2_prop)
        log_prior_prop = _log_prior(A_prop, mu_prop, sigma2_prop, K)
        log_post_prop = log_lik_prop + log_prior_prop

        # Log-acceptance with Jacobian for log-normal σ² proposal.
        # Proposal: σ²' = σ² · exp(Z), Z ~ N(0, scale).
        # q(σ²'|σ²) ≠ q(σ²|σ²') in σ² space → Jacobian = Π(σ²_cur / σ²_prop).
        # log J = Σ_k [log σ²_cur_k - log σ²_prop_k]
        log_jacobian = np.sum(np.log(sigma2_cur) - np.log(sigma2_prop))
        log_alpha = log_post_prop - log_post_cur + log_jacobian

        # Accept/reject (avoid log(0) warning by flooring uniform draw)
        log_u = np.log(max(rng.uniform(), np.finfo(float).tiny))
        if np.isfinite(log_alpha) and log_u < log_alpha:
            mu_cur = mu_prop
            sigma2_cur = sigma2_prop
            A_cur = A_prop
            pi_cur = pi_prop
            log_post_cur = log_post_prop
            n_accepted += 1

        # Store
        mu_samples[n] = mu_cur
        sigma2_samples[n] = sigma2_cur
        log_posterior_samples[n] = log_post_cur

        # Track MAP (only after burn-in)
        if n >= burn_in and log_post_cur > best_log_post:
            best_log_post = log_post_cur
            best_params = {
                "A": A_cur.copy(),
                "pi": pi_cur.copy(),
                "mu": mu_cur.copy(),
                "sigma2": sigma2_cur.copy(),
            }

    if best_params is None:
        # Fallback: use last sample
        best_params = {
            "A": A_cur.copy(),
            "pi": pi_cur.copy(),
            "mu": mu_cur.copy(),
            "sigma2": sigma2_cur.copy(),
        }

    # Sort states by ascending mu for consistent ordering
    order = np.argsort(best_params["mu"])
    best_params["mu"] = best_params["mu"][order]
    best_params["sigma2"] = best_params["sigma2"][order]
    best_params["A"] = best_params["A"][np.ix_(order, order)]
    best_params["pi"] = best_params["pi"][order]

    samples = {
        "mu": mu_samples,
        "sigma2": sigma2_samples,
        "log_posterior": log_posterior_samples,
    }

    acceptance_rate = n_accepted / n_samples
    return best_params, samples, acceptance_rate


def _safe_log_likelihood(
    obs: np.ndarray,
    A: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    sigma2: np.ndarray,
) -> float:
    """Compute log p(ΔY|Θ) via forward algorithm, returning -inf on failure."""
    try:
        _, ll = forward(obs, A, pi, mu, sigma2)
        if not np.isfinite(ll):
            return -np.inf
        return ll
    except (ValueError, FloatingPointError):
        return -np.inf


def _log_prior(
    A: np.ndarray,
    mu: np.ndarray,
    sigma2: np.ndarray,
    K: int,
) -> float:
    """Log-prior p(Θ) for the MCMC HMM (Paper §3.3).

    Prior components:
        A rows ~ Dirichlet(e_{i1}, ..., e_{iK}), e_{ii}=4, e_{ij}=1/(K-1)
        μ_k ~ N(0, σ²_prior=1.0)  (broad prior)
        σ²_k ~ InvGamma(α=2, β=0.001)  (hierarchical, weakly informative)

    log p(A) = Σ_i [log Γ(Σ e_{ij}) - Σ log Γ(e_{ij}) + Σ (e_{ij}-1) log a_{ij}]
    log p(μ) = -0.5 Σ_k μ_k² / σ²_prior  (up to constant)
    log p(σ²) = Σ_k [-(α+1) log σ²_k - β/σ²_k]  (up to constant)
    """
    log_p = 0.0

    # Dirichlet prior on A rows (Paper §3.3)
    e_diag = 4.0
    e_off = 1.0 / max(K - 1, 1)
    for i in range(K):
        e = np.full(K, e_off)
        e[i] = e_diag
        # Log-Dirichlet: Σ(e-1)*log(a) + log Γ(Σe) - Σlog Γ(e)
        log_p += float(gammaln(e.sum()) - np.sum(gammaln(e)))
        log_p += float(np.sum((e - 1.0) * np.log(np.maximum(A[i], 1e-300))))

    # Gaussian prior on μ: N(0, 1)
    sigma2_prior = 1.0
    log_p += -0.5 * np.sum(mu ** 2 / sigma2_prior)

    # Inverse-Gamma prior on σ²: IG(α=2, β=0.001)
    alpha_ig = 2.0
    beta_ig = 0.001
    for k in range(K):
        if sigma2[k] > 0:
            log_p += -(alpha_ig + 1) * np.log(sigma2[k]) - beta_ig / sigma2[k]
        else:
            return -np.inf

    return log_p


def _sample_dirichlet_rows(K: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a K×K transition matrix with Dirichlet rows (Paper §3.3 prior)."""
    e_diag = 4.0
    e_off = 1.0 / max(K - 1, 1)
    A = np.empty((K, K))
    for i in range(K):
        alpha = np.full(K, e_off)
        alpha[i] = e_diag
        A[i] = rng.dirichlet(alpha)
    return A


def _propose_transition_matrix(
    A: np.ndarray,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Propose a new transition matrix by perturbing in Dirichlet space.

    Adds noise to the log of each row, then re-normalizes.
    This ensures rows remain valid probability distributions.
    """
    K = A.shape[0]
    A_prop = np.empty_like(A)
    for i in range(K):
        log_row = np.log(np.maximum(A[i], 1e-300))
        log_row += rng.normal(0, scale, K)
        # Softmax to get valid probabilities
        log_row -= logsumexp(log_row)
        A_prop[i] = np.exp(log_row)
    return A_prop


def _ergodic_distribution(A: np.ndarray) -> np.ndarray:
    """Compute the stationary (ergodic) distribution of transition matrix A.

    Solves π·A = π, Σπ_k = 1.

    The paper (§3.3): "The vector π = {π_1, ..., π_K} of the initial states
    is drawn from the ergodic distribution of the hidden Markov chain."

    Uses eigendecomposition: the ergodic distribution is the left eigenvector
    of A corresponding to eigenvalue 1.
    """
    K = A.shape[0]
    if K == 1:
        return np.array([1.0])

    # Left eigenvectors: rows of A^T
    eigenvalues, eigenvectors = np.linalg.eig(A.T)

    # Find eigenvector for eigenvalue ≈ 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    pi = np.abs(pi)  # Ensure non-negative
    # Clip to avoid zero entries that would crash forward() (requires A, pi > 0)
    pi = np.maximum(pi, 1e-10)
    pi /= pi.sum()
    return pi


def bridge_sampling_log_marginal(
    samples_mu: np.ndarray,
    samples_sigma2: np.ndarray,
    observations: np.ndarray,
    K: int,
    A: np.ndarray,
    *,
    n_importance: int = 1000,
    random_state: int = 42,
) -> float:
    """Bridge sampling estimate of log p(ΔY | M_K) for model selection (Paper §3.3).

    Approximates the marginal likelihood using posterior MCMC draws and
    importance samples from an approximating density q(Θ):

        p̂(ΔY|M_K) = [L⁻¹ Σ_l ψ(Θ̃_l) p*(Θ̃_l|ΔY)] / [N⁻¹ Σ_n ψ(Θ̆_n) q(Θ̆_n)]

    where p* is the unnormalized posterior, q is a Gaussian approximation
    fitted to the MCMC draws, and ψ is a bridge function.

    For simplicity we use the harmonic mean estimator as a baseline:
        log p̂(ΔY|M_K) ≈ -log[N⁻¹ Σ_n 1/p(ΔY|Θ^(n))]

    This is biased but sufficient for relative model comparison.

    Parameters
    ----------
    samples_mu : np.ndarray, shape (N, K)
        Posterior MCMC draws for emission means (post burn-in).
    samples_sigma2 : np.ndarray, shape (N, K)
        Posterior MCMC draws for emission variances (post burn-in).
    observations : np.ndarray, shape (T,)
        Observed log-returns.
    K : int
        Number of hidden states for this model.
    A : np.ndarray, shape (K, K)
        Transition matrix (fixed at MAP or posterior mean).
    n_importance : int
        Number of draws to use from the posterior samples.
    random_state : int
        Seed for subsampling.

    Returns
    -------
    log_marginal : float
        Estimated log marginal likelihood log p(ΔY | M_K).
    """
    if len(samples_mu) == 0:
        raise ValueError("No posterior samples provided")

    rng = np.random.default_rng(random_state)
    N = min(n_importance, len(samples_mu))
    idx = rng.choice(len(samples_mu), size=N, replace=False)

    pi = _ergodic_distribution(A)

    # Harmonic mean estimator: -logsumexp(-log_liks) + log(N)
    log_liks = np.empty(N)
    for i, j in enumerate(idx):
        mu_j = samples_mu[j]
        s2_j = np.maximum(samples_sigma2[j], 1e-12)
        log_liks[i] = _safe_log_likelihood(observations, A, pi, mu_j, s2_j)

    # Filter out -inf
    valid = np.isfinite(log_liks)
    if not np.any(valid):
        return -np.inf

    log_liks_valid = log_liks[valid]
    N_valid = len(log_liks_valid)

    # Harmonic mean: log p(Y|M) ≈ -log(1/N Σ 1/p(Y|Θ)) = -log(1/N) - logsumexp(-log_liks)
    #              = log(N) - logsumexp(-log_liks)
    log_marginal = np.log(N_valid) - logsumexp(-log_liks_valid)
    return float(log_marginal)
