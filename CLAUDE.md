# CLAUDE.md — HMM Momentum Trading Project

## Project Context

This is a **university master's project** for the course "Advanced Signal Processing: Tools and Applications" (ASPTA) at UPC Barcelona. The student must present this code orally to professors and defend every implementation choice. Technical debt, black-box code, or unexplainable behavior will be visible during the presentation and hurt the grade.

The project reproduces results from: Christensen, Turner & Godsill (2020), "Hidden Markov Models Applied To Intraday Momentum Trading With Side Information" (arXiv:2006.08307, Cambridge Signal Processing Lab).

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

Exception: `scipy.special.logsumexp` is acceptable for the log-sum-exp trick (it's a numerical utility, not statistics).

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
- **Kalman covariance update**: use Joseph form `(I-KG) C (I-KG)' + K σ² K'`, not the standard form `(I-KG) C` which breaks PSD
- **Log-space**: already covered by Rule 6
- **Symmetry enforcement**: always symmetrize covariance matrices after computation: `C = (C + C.T) / 2`

The RBPF runs many Kalman filters in parallel — a single numerically unstable update can corrupt all downstream particles.

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
numba          — JIT compilation for performance-critical loops (RBPF, Baum-Welch)
scipy          — logsumexp only; optionally scipy.optimize for extensions
matplotlib     — all plotting
yfinance       — daily data download (SPY, QQQ, etc.)
databento      — 1-min CME futures data (parquet files in data/databento/)
hmmlearn       — validation/comparison ONLY (never used as the main implementation)
pytest         — testing
```

Do NOT install or use: sklearn (not needed), tensorflow, pytorch, pandas (only in data loading layer if convenient).

## File Locations

- Source code: `src/`
- Experiments: `experiments/`
- Tests: `tests/` (337 tests)
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
- `src/langevin/rbpf_numba.run_rbpf_numba(...)` — Numba-accelerated RBPF (~120x speedup)
- `src/langevin/utils.fir_momentum_signal(trend, n_taps)` — FIR smoothing (Paper §IV-D)
- `src/langevin/utils.igarch_volatility_scale(sig, returns, alpha)` — IGARCH scaling (Paper §IV-D)

## Implementation Order

```
Phase 1 — Data Layer ✅ COMPLETE
  1. src/data/loader.py
  2. src/data/features.py
  3. src/utils/metrics.py
  4. src/utils/plotting.py

Phase 2 — HMM Core ✅ COMPLETE (82 tests passing)
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

Phase 4 — Experiments ✅ COMPLETE (all verified with text reports)
  14. experiments/01_data_exploration.py
  15. experiments/02_model_selection.py
  16. experiments/03_baum_welch_training.py
  17. experiments/04_regime_detection.py
  18. experiments/05_backtest_comparison.py

Phase 5 — Extensions ✅ COMPLETE
  19. experiments/06_em_vs_mcmc.py
  20. experiments/07_multi_asset.py

Phase 6 — Langevin Model ✅ COMPLETE (Issue #41, PR #51)
  21. src/langevin/model.py     + tests/test_langevin_model.py (19 tests)

Phase 7 — Kalman Filter ✅ COMPLETE (Issue #42, PR #52-#53)
  22. src/langevin/kalman.py    + tests/test_kalman.py (18 tests)

Phase 8 — Standard Particle Filter ✅ COMPLETE (Issue #43, PR #54)
  23. src/langevin/particle.py  + tests/test_particle.py

Phase 9 — RBPF ✅ COMPLETE (Issues #44-#45, PR #55-#56)
  24. src/langevin/rbpf.py      + tests/test_rbpf.py
  25. src/langevin/utils.py

Phase 10 — RBPF Experiments ✅ COMPLETE (Issues #46-#50, PR #57-#62)
  26. experiments/12_kalman_filter_intro.py
  27. experiments/13_langevin_model.py
  28. experiments/14_particle_filter_baseline.py
  29. experiments/15_rbpf_trading.py
  30. experiments/16_hmm_vs_rbpf.py

Phase 11 — Intraday Experiments ✅ COMPLETE (Issues #63-#69, PR #70-#74)
  31. src/data/futures_loader.py      (Databento CME loader)
  32. src/langevin/rbpf_numba.py      (Numba RBPF, ~120x speedup)
  33. src/hmm/baum_welch_numba.py     (Numba Baum-Welch, ~50-100x speedup)
  34. experiments/17_rbpf_1min_es.py   (RBPF on 1-min ES, Table I params)
  35. experiments/18_rbpf_param_search.py (378-pt grid search)
  36. experiments/19_rbpf_portfolio.py  (26-contract portfolio)
  37. experiments/20_rbpf_per_contract.py (per-contract calibration)
  38. experiments/16_hmm_vs_rbpf.py     (rewritten: fair 1-min comparison)
```

## Key Results

### HMM (2020 paper) — daily yfinance data
- Model selection: AIC and BIC both favor K=4; K=3 used for interpretability
- 3 regimes: bearish (3.6% of time, ann. vol 56%), neutral (40%), bullish (56%, ann. vol 8.5%)
- Out-of-sample SPY (2022-2024): weighted vote Sharpe **+0.54** vs buy-and-hold +0.40
- Refined (EMA + no-trade): Sharpe **+0.71**, max DD 7.8%
- HMM on 1-min ES: Sharpe **-1.22** (captures microstructure noise, not regimes)

### Langevin / RBPF (2012 paper) — synthetic + daily + 1-min
- Kalman filter validated on synthetic data: 96.4% trend within 2σ, residuals ~N(0,1)
- RBPF jump detection on synthetic data: 75% detection rate, 6.2% FP rate
- RBPF on daily SPY: Sharpe -1.77 (excessive turnover ~1.37)
- RBPF on 1-min ES (Table I): Sharpe **-0.20** (FIR+IGARCH smoothing helps)
- 378-pt parameter grid search: best Sharpe -0.19 (sf_obs=35% always wins — filter ignores data)
- 26-contract uniform portfolio: Sharpe -3.38 (same params fail across assets)
- Per-contract calibrated portfolio (14 survivors): Sharpe -2.12 (overfit)
- Paper original (75 contracts, 2006-2011): Sharpe +1.82 (not reproduced)

### Main Result (Experiment 16 — fair 1-min comparison)
- On identical 1-min ES data: **RBPF (-0.20) beats HMM (-1.22)**, neither beats B&H (+0.10)
- Each model works in its native domain: HMM on daily (+0.54), RBPF on intraday (-0.20)
- The 2012 paper's result does not generalise to 2019-2024 — market microstructure has changed
- Better state estimation (higher LL) does not mean better trading signals

### Why the 2012 paper's Sharpe 1.82 was not reproduced
1. **Market regime**: 2006-2011 had persistent intraday trends (incl. 2008 crisis); 2019-2024 is HFT-dominated and mean-reverting
2. **Diversification**: 75 contracts vs our 26, lower cross-correlation in their period
3. **Parameters**: hand-tuned per asset class with domain expertise, not grid-searched

## What "Done" Looks Like For Each Function

A function is done when:
1. ✅ It has a complete docstring with math and paper reference
2. ✅ It has a passing test against synthetic data
3. ✅ It has a passing test against hmmlearn (where applicable)
4. ✅ The student (me) has read the code and understands every line
5. ✅ It handles edge cases (empty input, single observation, K=1)

Do NOT move to the next function until all 5 are satisfied.
