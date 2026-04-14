# CLAUDE.md — HMM Momentum Trading Project

## Project Context

This is a **university master's project** for the course "Advanced Signal Processing: Tools and Applications" (ASPTA) at UPC Barcelona. The student must present this code orally to professors and defend every implementation choice. Technical debt, black-box code, or unexplainable behavior will be visible during the presentation and hurt the grade.

The project fully reproduces: Christensen, Turner & Godsill (2020), "Hidden Markov Models Applied To Intraday Momentum Trading With Side Information" (arXiv:2006.08307, Cambridge Signal Processing Lab). All five model variants (Default HMM, Baum-Welch HMM, MCMC HMM, Vol Ratio IOHMM, Seasonality IOHMM) are implemented on 1-min ES futures data, then compared against modern ML approaches (LSTM, XGBoost).

## Cardinal Rules

### 1. ONE FUNCTION AT A TIME

Never implement more than one function per interaction. The workflow is:
1. I ask for a specific function (e.g., "implement the forward algorithm")
2. You write ONLY that function with full docstring explaining the math
3. You write a test for that function
4. You run the test and show it passes
5. Only then do we move to the next function

Do NOT anticipate what comes next. Do NOT implement helper functions I didn't ask for. Do NOT refactor previous code unless I explicitly ask.

### 2. EVERY FUNCTION MUST HAVE A MATH DOCSTRING

Every function in `src/hmm/` must include in its docstring:
- The equation it implements (in plain text math notation)
- Which section/equation of the paper it corresponds to
- What each parameter means in the model
- What the return values represent

Example:
```python
def forward(observations, A, pi, mu, sigma2):
    """
    Forward algorithm (Paper §3.2, Algorithm 1 lines 6-9).

    Computes log α_t(k) for all t and k, where:
        α_1(k) = π_k · N(Δy_1; μ_k, σ²_k)
        α_t(k) = [Σ_i α_{t-1}(i) · A_{ik}] · N(Δy_t; μ_k, σ²_k)

    Uses log-space computation for numerical stability.

    Parameters:
        observations: np.ndarray, shape (T,)
            Log-returns Δy_1, ..., Δy_T
        A: np.ndarray, shape (K, K)
            Transition matrix. A[i,j] = p(m_t = j | m_{t-1} = i)
        pi: np.ndarray, shape (K,)
            Initial state distribution. pi[k] = p(m_1 = k)
        mu: np.ndarray, shape (K,)
            Emission means. mu[k] = mean return in state k
        sigma2: np.ndarray, shape (K,)
            Emission variances. sigma2[k] = variance of returns in state k

    Returns:
        log_alpha: np.ndarray, shape (T, K)
            Log forward variables. log_alpha[t, k] = log α_t(k)
        log_likelihood: float
            log p(ΔY | Θ) = log Σ_k α_T(k)
    """
```

### 3. NO CLASSES FOR CORE ALGORITHMS

The HMM algorithms (forward, backward, Baum-Welch, Viterbi) must be **pure functions**, not methods on a class. Reasons:
- A function is easier to read, test, and explain on a slide
- The math maps 1-to-1 to a function, not to a class hierarchy
- During the oral presentation, the professor can ask about one algorithm and the student opens one file

A thin wrapper class for convenience is acceptable ONLY in `experiments/` or `notebooks/`, NEVER in `src/hmm/`.

### 4. NUMPY ONLY — NO SCIPY.STATS FOR CORE MATH

For the Gaussian PDF computation, implement it directly:
```python
log_pdf = -0.5 * np.log(2 * np.pi * sigma2) - 0.5 * (x - mu)**2 / sigma2
```

Do NOT use `scipy.stats.norm.logpdf`. The student must be able to write the Gaussian PDF on a whiteboard. Using a library call hides this.

Exception: `scipy.special.logsumexp` is acceptable for the log-sum-exp trick (it's a numerical utility, not statistics). `scipy.interpolate` is acceptable for B-spline fitting (Paper §4).

### 5. VALIDATE BEFORE BUILDING ON TOP

Before moving from one function to the next in the dependency chain, we must verify correctness:

- **forward.py** → test against `hmmlearn` on synthetic data (log-likelihoods must match to 1e-6)
- **backward.py** → test that α · β gives the same likelihood as forward-only
- **forward_backward.py** → test that γ_t(k) sums to 1.0 for each t
- **baum_welch.py** → test that log-likelihood increases monotonically each iteration
- **baum_welch.py** → test parameter recovery on synthetic data (generate from known HMM, recover params)
- **viterbi.py** → test on synthetic data where the true state sequence is known
- **inference.py** → test that online predictions match the batch forward algorithm results

If a test fails, we fix the function before proceeding. No exceptions.

### 6. LOG-SPACE EVERYWHERE

All computations in `src/hmm/` must use log-probabilities. Products of probabilities become sums of log-probabilities. The log-sum-exp trick must be used for any summation of probabilities:

```python
from scipy.special import logsumexp
# log(Σ exp(a_i)) computed stably
result = logsumexp(log_values)
```

The reason: with T=1000 and K=3, the forward variables α_t(k) will underflow to 0.0 in regular probability space. This is not optional — it is required for correctness.

### 7. EXPERIMENT SCRIPTS ARE STANDALONE

Each file in `experiments/` must:
- Be runnable independently: `python experiments/03_baum_welch_training.py`
- Import only from `src/`
- Print clear output explaining what is happening
- Save any generated figures to `figures/`
- Not depend on other experiment scripts having been run first

### 8. COMMIT MESSAGES REFERENCE THE PAPER

When committing code, reference the relevant paper section:
```
feat: implement forward algorithm (Paper §3.2, Alg 1 lines 6-9)
feat: implement Baum-Welch M-step (Paper §3.2, Alg 1 lines 17-21)
test: verify forward against hmmlearn on 3-state synthetic data
```

### 9. NO PREMATURE OPTIMIZATION

Do not vectorize loops for clarity's sake in the first implementation. A clear double for-loop over t and k is better than a clever einsum that the student can't explain. Optimization (if needed) comes after correctness is verified and understood.

Exception: Numba `@njit` backends are acceptable for 1-min scale (400k+ bars) after the pure-Python version is verified.

### 10. ASK BEFORE ACTING ON AMBIGUITY

If an implementation choice is ambiguous (e.g., "how to initialize the transition matrix?", "tied vs untied variances?"), do NOT make a silent default. State the options, explain the trade-offs, and let me decide. The student must be able to justify every choice during the presentation.

### 11. VALIDATE INPUTS AT FUNCTION BOUNDARIES

Every public function in `src/` must guard against invalid inputs that would silently produce wrong results. Examples:
- `theta > 0` in a mean-reversion parameter → `raise ValueError`
- `tau` outside `[0, dt]` for a jump time → `raise ValueError`
- Negative variances, non-square matrices, dimension mismatches

Silent garbage-in-garbage-out is worse than a crash. Each guard must have a matching test (`pytest.raises`).

### 12. TEST BOUNDARY AND EDGE CASES, NOT JUST THE HAPPY PATH

Every test file must include tests at the boundaries of the parameter space, not only interior/typical values:
- Zero and extreme values (e.g., `sigma_obs = 0.0` exactly, not `1e-15` as a proxy)
- Boundary conditions (e.g., `tau = 0`, `tau = dt`, `T = 1`)
- Degenerate cases where algebra simplifies (e.g., `F = I`, `Q = 0`)
- Semigroup/identity checks: a no-op parameter (like `sigma_J = 0, mu_J = 0`) must match the no-jump path exactly

### 13. USE NUMERICALLY STABLE FORMULATIONS

When a numerically stable variant of a formula exists, use it from the start — do not write the naive version first:
- **Log-space**: already covered by Rule 6
- **Symmetry enforcement**: always symmetrize covariance matrices after computation: `C = (C + C.T) / 2`
- **Variance floors**: tie to tick size (α²/2) for discretized models, not arbitrary constants

### 14. DERIVE TEST TOLERANCES FROM PARAMETERS

Never hard-code magic tolerance numbers like `atol=1e-15` without justification. Instead, derive tolerances from the test parameters:
```python
# BAD: magic number, will break if sigma or dt changes
np.testing.assert_allclose(Q, np.zeros((2, 2)), atol=1e-15)

# GOOD: tolerance derived from expected magnitude
# Q[1,1] ≈ sigma^2 * dt, so use that as the scale
np.testing.assert_allclose(Q, np.zeros((2, 2)), atol=sigma**2 * dt * 10)
```

This prevents tests that pass by coincidence and break when parameters change.

## Tech Stack

```
Python 3.10+
numpy          — all core math
numba          — JIT compilation for performance-critical loops (Baum-Welch at 1-min scale)
scipy          — logsumexp, B-spline fitting (scipy.interpolate)
matplotlib     — all plotting
yfinance       — daily data download (SPY, QQQ, etc.)
databento      — 1-min CME futures data (parquet files in data/databento/)
hmmlearn       — validation/comparison ONLY (never used as the main implementation)
pytest         — testing
torch          — LSTM/GRU ML baseline (Phase 16 only)
xgboost        — gradient boosting ML baseline (Phase 16 only)
```

Do NOT use: sklearn (not needed for core), tensorflow, pandas (only in data loading layer if convenient).

## File Locations

- Source code: `src/`
- Experiments: `experiments/`
- Supplementary (RBPF): `experiments/supplementary/`
- Tests: `tests/`
- Figures output: `figures/` (gitignored PNGs)
- Reports output: `reports/` (gitignored TXT reports from experiments)
- Notebooks: `notebooks/`

## Shared Utilities

These helpers are used across multiple experiment scripts:

- `src/data/loader.extract_close_series(prices)` — extracts 1-D close Series from yfinance DataFrame
- `src/data/futures_loader.load_futures_1m(sym)` — loads 1-min CME futures from Databento parquet
- `src/data/futures_loader.filter_rth(df)` — filters to Regular Trading Hours only
- `src/hmm/utils.sort_states(params)` — reorders states by ascending emission mean, clips/renormalizes
- `src/hmm/utils.train_best_model(obs, K, ...)` — runs multiple single-restart EM fits, keeps best LL
- `src/hmm/baum_welch_numba.train_hmm_numba(obs, K, ...)` — Numba-accelerated Baum-Welch (~50-100x)
- `src/hmm/baum_welch_numba.run_inference_numba(obs, ...)` — Numba-accelerated online inference

## Implementation Order

```
Phase 1 — Data Layer ✅ COMPLETE
  1. src/data/loader.py
  2. src/data/features.py
  3. src/utils/metrics.py
  4. src/utils/plotting.py

Phase 2 — HMM Core ✅ COMPLETE
  5. src/hmm/forward.py        + tests/test_forward.py
  6. src/hmm/backward.py       + tests/test_backward.py
  7. src/hmm/forward_backward.py + tests/test_forward_backward.py
  8. src/hmm/baum_welch.py     + tests/test_baum_welch.py
  9. src/hmm/viterbi.py        + tests/test_viterbi.py
  10. src/hmm/model_selection.py + tests/test_model_selection.py
  11. src/hmm/inference.py      + tests/test_inference.py

Phase 3 — Strategy Layer ✅ COMPLETE
  12. src/strategy/signals.py
  13. src/strategy/backtest.py  + tests/test_backtest.py

Phase 4 — HMM Experiments (daily) ✅ COMPLETE
  14. experiments/01-05 (data exploration → backtest)

Phase 5 — Extensions (daily) ✅ COMPLETE
  15. experiments/06-11 (MCMC, multi-asset, K selection, rolling, refinement)

Phase 6-11 — Langevin/RBPF ✅ COMPLETE (supplementary, see experiments/supplementary/)

Phase 12 — Housekeeping & Diagnostics (#83, #84, #85, #89)
  16. Deprecate RBPF experiments → experiments/supplementary/
  17. experiments/21_autocorrelation_analysis.py  (frequency microstructure)
  18. src/hmm/discretize.py     + tests/test_discretize.py (tick-size grid)

Phase 13 — Rolling Retraining (#86)
  19. src/hmm/rolling.py        + tests/test_rolling.py
  20. experiments/22_rolling_hmm_1min.py

Phase 14 — Paper Model Variants (#87, #88)
  21. src/hmm/plr.py            + tests/test_plr.py (Default HMM, Paper §3.1)
  22. src/hmm/mcmc.py           + tests/test_mcmc.py (MCMC HMM, Paper §3.3)

Phase 15 — Side Information & IOHMM (#75, #76, #77, #78)
  23. src/hmm/side_info.py      + tests/test_side_info.py (Paper §4)
  24. src/hmm/iohmm.py          + tests/test_iohmm.py (Paper §5-6)

Phase 16 — ML Baselines (#79, #80, #81)
  25. src/ml/features.py        + tests/test_ml_features.py
  26. src/ml/lstm.py            + tests/test_lstm.py
  27. src/ml/boosting.py        + tests/test_boosting.py

Phase 17 — Final Experiments (#90, #82)
  28. experiments/23_full_paper_reproduction.py  (all 5 paper models, Figure 8)
  29. experiments/24_ml_vs_hmm_comparison.py     (7-way ML vs HMM vs IOHMM)
```

## What "Done" Looks Like For Each Function

A function is done when:
1. It has a complete docstring with math and paper reference
2. It has a passing test against synthetic data
3. It has a passing test against hmmlearn (where applicable)
4. The student (me) has read the code and understands every line
5. It handles edge cases (empty input, single observation, K=1)

Do NOT move to the next function until all 5 are satisfied.
