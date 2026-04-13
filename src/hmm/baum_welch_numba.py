"""Numba-accelerated Baum-Welch training and online inference for Gaussian HMMs.

Drop-in replacement for baum_welch.baum_welch + inference.run_inference when
T is large (e.g. 400k 1-min bars).  The forward, backward, and M-step loops
are compiled via @nb.njit, giving ~50-100x speedup over pure-Python/NumPy.

Key optimisation: we accumulate xi_sum (K x K) instead of materialising the
full xi array (T-1 x K x K), reducing memory from O(T K^2) to O(K^2).

Public API
----------
train_hmm_numba(obs, K, ...) -> dict, list[float]
    Baum-Welch with multiple restarts.  Returns best params + LL history.

run_inference_numba(obs, A, pi, mu, sigma2) -> predictions, state_probs
    Online predict-update filter.  Same interface as inference.run_inference.
"""

from __future__ import annotations

import numba as nb
import numpy as np


# ── Numba helpers ─────────────────────────────────────────────────────────────

@nb.njit(cache=True)
def _logsumexp(a):
    """Stable logsumexp for a 1-D array."""
    m = a[0]
    for i in range(1, a.shape[0]):
        if a[i] > m:
            m = a[i]
    if m == -np.inf:
        return -np.inf
    s = 0.0
    for i in range(a.shape[0]):
        s += np.exp(a[i] - m)
    return m + np.log(s)


@nb.njit(cache=True)
def _log_gauss(x, mu, sigma2):
    """log N(x; mu, sigma2) — univariate Gaussian log-PDF."""
    return -0.5 * np.log(2.0 * np.pi * sigma2) - 0.5 * (x - mu) ** 2 / sigma2


# ── Forward / Backward ───────────────────────────────────────────────────────

@nb.njit(cache=True)
def _forward(obs, log_A, log_pi, mu, sigma2):
    """Forward algorithm in log-space.  Returns (log_alpha, ll)."""
    T = obs.shape[0]
    K = mu.shape[0]
    log_alpha = np.empty((T, K))

    for k in range(K):
        log_alpha[0, k] = log_pi[k] + _log_gauss(obs[0], mu[k], sigma2[k])

    buf = np.empty(K)
    for t in range(1, T):
        for k in range(K):
            for i in range(K):
                buf[i] = log_alpha[t - 1, i] + log_A[i, k]
            log_alpha[t, k] = _logsumexp(buf) + _log_gauss(obs[t], mu[k], sigma2[k])

    final = np.empty(K)
    for k in range(K):
        final[k] = log_alpha[T - 1, k]
    ll = _logsumexp(final)

    return log_alpha, ll


@nb.njit(cache=True)
def _backward(obs, log_A, mu, sigma2):
    """Backward algorithm in log-space.  Returns log_beta."""
    T = obs.shape[0]
    K = mu.shape[0]
    log_beta = np.empty((T, K))

    for k in range(K):
        log_beta[T - 1, k] = 0.0

    buf = np.empty(K)
    for t in range(T - 2, -1, -1):
        for i in range(K):
            for j in range(K):
                buf[j] = (log_A[i, j]
                          + _log_gauss(obs[t + 1], mu[j], sigma2[j])
                          + log_beta[t + 1, j])
            log_beta[t, i] = _logsumexp(buf)

    return log_beta


# ── E-step: gamma + xi_sum ───────────────────────────────────────────────────

@nb.njit(cache=True)
def _e_step(obs, log_alpha, log_beta, log_A, mu, sigma2):
    """Compute gamma (T, K) and xi_sum (K, K) from forward-backward results.

    xi_sum[i, j] = sum_{t=0}^{T-2} xi_t(i, j)
    where xi_t(i,j) = p(m_t=i, m_{t+1}=j | y_{1:T}).
    """
    T, K = log_alpha.shape
    gamma = np.empty((T, K))
    xi_sum = np.zeros((K, K))

    # gamma
    buf = np.empty(K)
    for t in range(T):
        for k in range(K):
            buf[k] = log_alpha[t, k] + log_beta[t, k]
        norm = _logsumexp(buf)
        for k in range(K):
            gamma[t, k] = np.exp(buf[k] - norm)

    # xi_sum — accumulated without materialising (T-1, K, K)
    log_xi = np.empty((K, K))
    for t in range(T - 1):
        max_val = -np.inf
        for i in range(K):
            for j in range(K):
                v = (log_alpha[t, i] + log_A[i, j]
                     + _log_gauss(obs[t + 1], mu[j], sigma2[j])
                     + log_beta[t + 1, j])
                log_xi[i, j] = v
                if v > max_val:
                    max_val = v

        total = 0.0
        for i in range(K):
            for j in range(K):
                total += np.exp(log_xi[i, j] - max_val)
        log_norm = max_val + np.log(total)

        for i in range(K):
            for j in range(K):
                xi_sum[i, j] += np.exp(log_xi[i, j] - log_norm)

    return gamma, xi_sum


# ── M-step ────────────────────────────────────────────────────────────────────

@nb.njit(cache=True)
def _m_step(obs, gamma, xi_sum, min_var):
    """M-step: update (pi, A, mu, sigma2) from sufficient statistics."""
    T, K = gamma.shape

    # pi = gamma[0]
    pi_new = np.empty(K)
    pi_sum = 0.0
    for k in range(K):
        pi_new[k] = gamma[0, k]
        pi_sum += pi_new[k]
    for k in range(K):
        pi_new[k] /= pi_sum

    # A
    A_new = np.empty((K, K))
    for i in range(K):
        row_sum = 0.0
        for j in range(K):
            row_sum += xi_sum[i, j]
        for j in range(K):
            A_new[i, j] = xi_sum[i, j] / max(row_sum, 1e-300)

    # mu, sigma2
    mu_new = np.empty(K)
    sigma2_new = np.empty(K)
    for k in range(K):
        g_sum = 0.0
        w_sum = 0.0
        for t in range(T):
            g_sum += gamma[t, k]
            w_sum += gamma[t, k] * obs[t]
        mu_new[k] = w_sum / max(g_sum, 1e-300)

        v_sum = 0.0
        for t in range(T):
            d = obs[t] - mu_new[k]
            v_sum += gamma[t, k] * d * d
        sigma2_new[k] = max(v_sum / max(g_sum, 1e-300), min_var)

    return pi_new, A_new, mu_new, sigma2_new


# ── Single Baum-Welch run ────────────────────────────────────────────────────

@nb.njit(cache=True)
def _baum_welch_single(obs, pi0, A0, mu0, sigma20, max_iter, tol, min_var):
    """One EM run from given initial parameters.

    Returns (pi, A, mu, sigma2, final_ll, n_iters).
    """
    K = mu0.shape[0]
    pi = pi0.copy()
    A = A0.copy()
    mu = mu0.copy()
    sigma2 = sigma20.copy()

    prev_ll = -np.inf

    n_iters = 0
    for it in range(max_iter):
        n_iters = it + 1

        log_A = np.log(A)
        log_pi = np.log(pi)

        log_alpha, ll = _forward(obs, log_A, log_pi, mu, sigma2)

        # Catch degenerate runs
        if not np.isfinite(ll):
            break

        log_beta = _backward(obs, log_A, mu, sigma2)
        gamma, xi_sum = _e_step(obs, log_alpha, log_beta, log_A, mu, sigma2)
        pi, A, mu, sigma2 = _m_step(obs, gamma, xi_sum, min_var)

        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    return pi, A, mu, sigma2, ll, n_iters


# ── Online inference ──────────────────────────────────────────────────────────

@nb.njit(cache=True)
def _online_inference(obs, A, pi, mu, sigma2):
    """Online predict-update filter (Paper §6, Algorithm 4).

    Returns (predictions, state_probs) both shape (T,) / (T, K).
    predictions[t] = E[y_t | y_{1:t-1}] (one-step-ahead prediction).
    state_probs[t] = p(m_t | y_{1:t}).
    """
    T = obs.shape[0]
    K = mu.shape[0]
    predictions = np.empty(T)
    state_probs = np.empty((T, K))

    # t = 0: use prior pi
    pred0 = 0.0
    for k in range(K):
        pred0 += pi[k] * mu[k]
    predictions[0] = pred0

    # Update with first observation
    log_omega = np.empty(K)
    for k in range(K):
        log_omega[k] = np.log(pi[k]) + _log_gauss(obs[0], mu[k], sigma2[k])
    norm = _logsumexp(log_omega)
    omega = np.empty(K)
    for k in range(K):
        omega[k] = np.exp(log_omega[k] - norm)
    state_probs[0, :] = omega

    # t >= 1
    buf = np.empty(K)
    for t in range(1, T):
        # Predict: omega_pred[k] = sum_i omega[i] * A[i, k]
        omega_pred = np.empty(K)
        for k in range(K):
            s = 0.0
            for i in range(K):
                s += omega[i] * A[i, k]
            omega_pred[k] = s

        # One-step-ahead prediction
        pred = 0.0
        for k in range(K):
            pred += omega_pred[k] * mu[k]
        predictions[t] = pred

        # Update
        for k in range(K):
            buf[k] = np.log(max(omega_pred[k], 1e-300)) + _log_gauss(obs[t], mu[k], sigma2[k])
        norm = _logsumexp(buf)
        for k in range(K):
            omega[k] = np.exp(buf[k] - norm)
        state_probs[t, :] = omega

    return predictions, state_probs


# ── Public Python API ─────────────────────────────────────────────────────────

def train_hmm_numba(
    observations: np.ndarray,
    K: int,
    *,
    n_restarts: int = 10,
    max_iter: int = 200,
    tol: float = 1e-6,
    min_variance: float = 1e-8,
    random_state: int = 42,
    verbose: bool = False,
) -> tuple[dict, list[float]]:
    """Baum-Welch EM with multiple restarts (Numba-accelerated).

    Parameters
    ----------
    observations : 1-D float array, the observed sequence.
    K            : number of hidden states.
    n_restarts   : number of random restarts.
    max_iter     : max EM iterations per restart.
    tol          : convergence tolerance on LL change.
    min_variance : floor for emission variances.
    random_state : RNG seed.
    verbose      : print per-restart progress.

    Returns
    -------
    params : dict with keys "A", "pi", "mu", "sigma2"
        Best parameters (states sorted by ascending mu).
    history : list[float]
        LL value at the end of the best restart (single-element list for
        compatibility; the inner Numba loop does not store per-iteration LL).
    """
    obs = np.ascontiguousarray(observations, dtype=np.float64)
    if obs.ndim != 1 or obs.size == 0:
        raise ValueError("observations must be a non-empty 1D array")
    if K < 1:
        raise ValueError("K must be >= 1")

    rng = np.random.default_rng(random_state)
    obs_mean = float(np.mean(obs))
    obs_std = float(np.std(obs)) + 1e-12

    best_ll = -np.inf
    best_params = None

    for r in range(n_restarts):
        # Random initialisation (in Python — RNG not available inside njit)
        A0 = rng.dirichlet(alpha=np.ones(K), size=K).astype(np.float64)
        pi0 = rng.dirichlet(alpha=np.ones(K)).astype(np.float64)
        mu0 = (obs_mean + rng.normal(0.0, obs_std, size=K)).astype(np.float64)
        sigma20 = np.full(K, max(obs_std ** 2, min_variance), dtype=np.float64)

        pi, A, mu, sigma2, ll, n_it = _baum_welch_single(
            obs, pi0, A0, mu0, sigma20, max_iter, tol, min_variance,
        )

        if not np.isfinite(ll):
            continue

        if verbose:
            print(f"  restart {r+1}/{n_restarts}: LL={ll:.2f}, iters={n_it}")

        if ll > best_ll:
            best_ll = ll
            best_params = (pi.copy(), A.copy(), mu.copy(), sigma2.copy())

    if best_params is None:
        raise RuntimeError("All restarts failed (non-finite LL)")

    pi, A, mu, sigma2 = best_params

    # Sort states by ascending mu
    order = np.argsort(mu)
    mu = mu[order]
    sigma2 = sigma2[order]
    pi = pi[order]
    A = A[np.ix_(order, order)]

    # Clip and renormalize (same as sort_states)
    eps = 1e-12
    A = np.clip(A, eps, None)
    A /= A.sum(axis=1, keepdims=True)
    pi = np.clip(pi, eps, None)
    pi /= pi.sum()
    sigma2 = np.clip(sigma2, eps, None)

    params = {"A": A, "pi": pi, "mu": mu, "sigma2": sigma2}
    return params, [best_ll]


def run_inference_numba(
    observations: np.ndarray,
    A: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    sigma2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Online predict-update filter (Numba-accelerated).

    Same interface as inference.run_inference:
        predictions[t] = E[y_t | y_{1:t-1}]
        state_probs[t] = p(m_t | y_{1:t})

    Parameters
    ----------
    observations : (T,) array of returns.
    A            : (K, K) transition matrix.
    pi           : (K,) initial state distribution.
    mu           : (K,) emission means.
    sigma2       : (K,) emission variances.

    Returns
    -------
    predictions : (T,) one-step-ahead return predictions.
    state_probs : (T, K) filtered state posteriors.
    """
    obs = np.ascontiguousarray(observations, dtype=np.float64)
    A = np.ascontiguousarray(A, dtype=np.float64)
    pi = np.ascontiguousarray(pi, dtype=np.float64)
    mu = np.ascontiguousarray(mu, dtype=np.float64)
    sigma2 = np.ascontiguousarray(sigma2, dtype=np.float64)
    return _online_inference(obs, A, pi, mu, sigma2)
