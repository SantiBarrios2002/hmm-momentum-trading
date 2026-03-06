# Project Architecture: HMM Momentum Trading

## Overview

This document explains the coding architecture so you understand every piece before, during, and after implementation. The project has **4 layers**, each building on the previous one. Nothing is a black box.

---

## The 4 Layers

```
Layer 4: Experiments & Extensions     ← what you present
Layer 3: Trading Strategy & Backtest  ← applies the model to financial data
Layer 2: HMM Engine                   ← the core algorithm (forward-backward, Baum-Welch, Viterbi)
Layer 1: Data & Utilities             ← data loading, feature computation, plotting
```

You build bottom-up: Layer 1 → 2 → 3 → 4. Each layer is testable independently.

---

## Layer 1: Data & Utilities

### What it does
Loads financial data, computes features, provides helper functions for plotting and evaluation.

### Files
```
src/
  data/
    loader.py          ← download/load price data (yfinance wrapper)
    features.py        ← compute EWMA volatility, log-returns, normalize
  utils/
    plotting.py        ← regime-colored price charts, cumulative returns
    metrics.py         ← Sharpe ratio, max drawdown, annualized return
```

### Key functions

**loader.py:**
- `load_daily_prices(ticker, start, end)` → DataFrame with Date, Open, High, Low, Close, Volume
- `load_multiple(tickers, start, end)` → dict of DataFrames

**features.py:**
- `log_returns(prices)` → Series of Δyₜ = log(yₜ/yₜ₋₁)
- `ewma_volatility(returns, lambda_param)` → Series of σ²ₜ₊₁|ₜ using the recursive IIR filter from paper Eq. (4)
- `normalize_returns(returns, window)` → standardized returns using rolling mean and vol

**metrics.py:**
- `sharpe_ratio(daily_returns)` → annualized Sharpe = √252 × mean/std
- `max_drawdown(cumulative_returns)` → maximum peak-to-trough decline
- `annualized_return(daily_returns)` → geometric annualized return

### Why this layer matters for your presentation
When a professor asks "where does your data come from?" or "how do you compute volatility?", you point here. Every feature computation is an explicit, readable function — no pandas chains hidden inside the HMM code.

---

## Layer 2: HMM Engine

This is the core. **Every algorithm from the paper becomes one function.** No classes wrapping things opaquely — just pure functions that take arrays and return arrays.

### Files
```
src/
  hmm/
    forward.py         ← forward algorithm (Algorithm 4 prediction step)
    backward.py        ← backward algorithm
    forward_backward.py ← combines both, returns γₜ(k)
    baum_welch.py      ← EM algorithm (Algorithm 1)
    viterbi.py         ← MAP state sequence decoding
    model_selection.py ← AIC, BIC computation
    inference.py       ← online predict-update loop (Algorithm 4)
```

### Key functions and their math

**forward.py:**
```python
def forward(observations, A, pi, mu, sigma2):
    """
    Forward algorithm. Computes log α_t(k) for all t, k.
    
    α₁(k) = π_k · N(Δy₁; μ_k, σ²_k)
    αₜ(k) = [Σᵢ αₜ₋₁(i) · Aᵢₖ] · N(Δyₜ; μ_k, σ²_k)
    
    Works entirely in log-space for numerical stability.
    
    Parameters:
        observations: array (T,) — the log-returns Δy₁, ..., ΔyT
        A: array (K, K) — transition matrix
        pi: array (K,) — initial state distribution
        mu: array (K,) — emission means per state
        sigma2: array (K,) — emission variances per state
    
    Returns:
        log_alpha: array (T, K) — log forward variables
        log_likelihood: float — log p(ΔY | Θ)
    """
```

**backward.py:**
```python
def backward(observations, A, mu, sigma2):
    """
    Backward algorithm. Computes log β_t(k) for all t, k.
    
    βT(k) = 1  (log βT(k) = 0)
    βₜ(k) = Σⱼ Aₖⱼ · N(Δyₜ₊₁; μⱼ, σ²ⱼ) · βₜ₊₁(j)
    
    Returns:
        log_beta: array (T, K) — log backward variables
    """
```

**forward_backward.py:**
```python
def compute_posteriors(observations, A, pi, mu, sigma2):
    """
    E-step of Baum-Welch. Combines forward + backward to get:
    
    γₜ(k) = αₜ(k) · βₜ(k) / p(ΔY|Θ)
    ξₜ(i,j) = αₜ₋₁(i) · Aᵢⱼ · N(Δyₜ; μⱼ, σ²ⱼ) · βₜ(j) / p(ΔY|Θ)
    
    Returns:
        gamma: array (T, K) — state posteriors (occupation probabilities)
        xi: array (T-1, K, K) — transition posteriors
        log_likelihood: float
    """
```

**baum_welch.py:**
```python
def m_step(observations, gamma, xi):
    """
    M-step of Baum-Welch. Updates parameters using occupation probabilities.
    
    π̂_k = γ₁(k)
    Âᵢⱼ = Σₜ ξₜ(i,j) / Σₜ γₜ(i)
    μ̂_k = Σₜ γₜ(k)·Δyₜ / Σₜ γₜ(k)
    σ̂²_k = Σₜ γₜ(k)·(Δyₜ - μ̂_k)² / Σₜ γₜ(k)
    
    Returns:
        A_new, pi_new, mu_new, sigma2_new
    """

def baum_welch(observations, K, max_iter=100, tol=1e-6, n_restarts=10):
    """
    Full Baum-Welch algorithm with multiple random restarts.
    
    For each restart:
        1. Random initialization of Θ
        2. Repeat until convergence:
            a. E-step: compute_posteriors() → γ, ξ, log_lik
            b. M-step: m_step() → Θ_new
            c. Check: |log_lik_new - log_lik_old| < tol?
    
    Keep the restart with highest final log-likelihood.
    
    Returns:
        best_params: dict with A, pi, mu, sigma2
        log_likelihood_history: list of log-likelihoods per iteration
        gamma: final state posteriors
    """
```

**viterbi.py:**
```python
def viterbi(observations, A, pi, mu, sigma2):
    """
    Viterbi algorithm. Finds MAP state sequence.
    
    δₜ(k) = max_i [δₜ₋₁(i) · Aᵢₖ] · N(Δyₜ; μ_k, σ²_k)
    ψₜ(k) = argmax_i [δₜ₋₁(i) · Aᵢₖ]
    
    Backtrack from T to 1 using ψ to recover the optimal path.
    
    Returns:
        states: array (T,) — optimal state sequence
        log_prob: float — log probability of the optimal path
    """
```

**model_selection.py:**
```python
def compute_aic(log_likelihood, K, d=1):
    """AIC = -2·log(L) + 2p, where p = K(K-1) + 2K + (K-1)"""

def compute_bic(log_likelihood, K, n_obs, d=1):
    """BIC = -2·log(L) + p·log(n)"""

def select_K(observations, K_range=range(1, 11), criterion='bic'):
    """Fit HMMs for each K, return scores and best K."""
```

**inference.py:**
```python
def predict_update_step(omega_prev, A, mu, sigma2, observation):
    """
    Single step of the online forward algorithm (Algorithm 4, one iteration).
    
    Predict: ω_{t|t-1,k} = Σ_{k'} a_{kk'} · ω_{t-1|t-1,k'}
    Update:  ω_{t|t,k} ∝ ω_{t|t-1,k} · N(Δyₜ; μ_k, σ²_k)
    Output:  Δŷₜ = Σ_k ω_{t|t-1,k} · μ_k
    
    Returns:
        omega_new: array (K,) — updated state distribution
        prediction: float — predicted return
    """

def run_inference(observations, A, pi, mu, sigma2):
    """
    Full inference loop over all observations.
    Calls predict_update_step repeatedly.
    
    Returns:
        predictions: array (T,) — predicted returns at each step
        state_probs: array (T, K) — filtering distributions over time
    """
```

### Why separate files per algorithm?
When a professor asks "show me your forward algorithm", you open `forward.py` — 40 lines, one function, the math in the docstring. No scrolling through a 500-line class. No "it's somewhere in the HMM object." Each file is a self-contained, testable unit that maps to one section of the paper.

---

## Layer 3: Trading Strategy & Backtest

### Files
```
src/
  strategy/
    signals.py         ← convert HMM predictions to trading signals
    backtest.py        ← simulate P&L with transaction costs
```

### Key functions

**signals.py:**
```python
def predictions_to_signal(predictions, transfer_fn='sign'):
    """
    Convert raw return predictions to trading signals in [-1, 1].
    
    'sign': signal = sign(prediction)  — long if positive, short if negative
    'linear': signal = clip(prediction / scale, -1, 1)
    """

def states_to_signal(state_probs, mu):
    """
    Alternative: use state probabilities directly.
    signal = Σ_k ω_k · sign(μ_k)  — weighted vote of states
    """
```

**backtest.py:**
```python
def backtest(returns, signals, transaction_cost_bps=5):
    """
    Simulate strategy returns.
    
    strategy_return_t = signal_{t-1} · return_t  (signal is lagged by 1!)
    transaction_cost_t = |signal_t - signal_{t-1}| · cost_bps / 10000
    net_return_t = strategy_return_t - transaction_cost_t
    
    The 1-period lag is critical — you trade on yesterday's signal,
    not today's. The paper explicitly notes this (§7).
    
    Returns:
        net_returns: array — daily net returns
        cumulative: array — cumulative strategy value
        metrics: dict — Sharpe, return, drawdown, turnover
    """
```

---

## Layer 4: Experiments & Extensions

### Files
```
experiments/
  01_data_exploration.py      ← load data, plot returns, check stationarity
  02_model_selection.py       ← reproduce Figure 2 (AIC/BIC vs K)
  03_baum_welch_training.py   ← train HMM, show convergence, print parameters
  04_regime_detection.py      ← plot detected regimes on price chart
  05_backtest_comparison.py   ← reproduce Figure 8 (strategy Sharpe ratios)
  06_em_vs_mcmc.py            ← Extension A: compare estimation methods
  07_multi_asset.py           ← Extension B: apply to multiple assets
```

Each experiment is a **standalone script** that imports from src/, runs one specific analysis, and produces one or two figures. Numbered so you (and the professor) see the logical progression.

---

## Directory Structure

```
hmm-momentum/
├── CLAUDE.md                 ← instructions for Claude Code
├── README.md                 ← project overview + how to run
├── requirements.txt          ← numpy, scipy, yfinance, matplotlib, etc.
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── features.py
│   ├── hmm/
│   │   ├── __init__.py
│   │   ├── forward.py
│   │   ├── backward.py
│   │   ├── forward_backward.py
│   │   ├── baum_welch.py
│   │   ├── viterbi.py
│   │   ├── model_selection.py
│   │   └── inference.py
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── signals.py
│   │   └── backtest.py
│   └── utils/
│       ├── __init__.py
│       ├── plotting.py
│       └── metrics.py
│
├── experiments/
│   ├── 01_data_exploration.py
│   ├── 02_model_selection.py
│   ├── 03_baum_welch_training.py
│   ├── 04_regime_detection.py
│   ├── 05_backtest_comparison.py
│   ├── 06_em_vs_mcmc.py
│   └── 07_multi_asset.py
│
├── tests/
│   ├── test_forward.py       ← verify forward algo against hmmlearn
│   ├── test_backward.py
│   ├── test_baum_welch.py    ← verify convergence on synthetic data
│   ├── test_viterbi.py       ← verify against known state sequence
│   └── test_metrics.py
│
└── notebooks/
    └── demo.ipynb            ← interactive walkthrough for presentation
```

---

## Implementation Order

Build in this exact order. Each step is verifiable before moving on.

```
Step 1:  src/data/loader.py          ← can you load SPY data?
Step 2:  src/data/features.py        ← can you compute log-returns and EWMA vol?
Step 3:  src/utils/metrics.py        ← can you compute Sharpe ratio?
Step 4:  src/hmm/forward.py          ← test against hmmlearn on synthetic data
Step 5:  src/hmm/backward.py         ← test against hmmlearn
Step 6:  src/hmm/forward_backward.py ← do γₜ(k) sum to 1 at each t?
Step 7:  src/hmm/baum_welch.py       ← does log-likelihood increase each iteration?
Step 8:  src/hmm/viterbi.py          ← does it recover known states from synthetic data?
Step 9:  src/hmm/model_selection.py  ← does BIC pick K=3 for 3-state synthetic data?
Step 10: src/hmm/inference.py        ← does online prediction match batch forward?
Step 11: src/strategy/signals.py     ← sanity check signal values
Step 12: src/strategy/backtest.py    ← does buy-and-hold match raw returns?
Step 13: experiments/01 through 07   ← one at a time
```

---

## Validation Strategy

At every step, you validate against **two independent checks:**

1. **Synthetic data:** Generate data FROM a known HMM (you pick A, μ, σ²), then verify your algorithm recovers the parameters. If Baum-Welch can't recover parameters from data it generated, something is broken.

2. **Library comparison:** Run the same data through `hmmlearn.GaussianHMM` and compare outputs (log-likelihood, state posteriors, Viterbi path). Your numbers should match to ~6 decimal places.

This is how you avoid technical debt. Every function is verified before the next one is built on top of it.
