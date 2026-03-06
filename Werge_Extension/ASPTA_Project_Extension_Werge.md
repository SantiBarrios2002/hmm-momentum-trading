# ASPTA Project Extension: Multi-Asset Validation via Werge (2021)

**Role in the project:** This is Extension B — applying the HMM framework built for the Christensen paper to the multi-asset daily setting described in Werge (2021), using Werge's published results as an independent benchmark.

**Primary paper:** Christensen, Turner & Godsill (2020) — arXiv:2006.08307
**Extension benchmark paper:** Werge (2021) — arXiv:2107.05535

---

## Why Werge as an Extension (Not the Primary Paper)

The Christensen paper is the core of the project because it comes from a signal processing lab, compares three estimation methods (PLR, Baum-Welch, MCMC), and connects explicitly to Kalman filtering and digital filters — all native to ASPTA.

But Christensen only tests on **one instrument** (ES futures) at **one frequency** (1-minute). That leaves a natural question: does this HMM framework generalize to daily data across multiple asset classes? Werge answers this — and provides detailed tables of exact performance metrics you can compare against.

The extension tests whether the codebase you built for Christensen (forward algorithm, Baum-Welch, inference loop) works on a completely different data regime, using Werge's published Sharpe ratios as ground truth.

---

## What Werge Adds That Christensen Doesn't

| Dimension | Christensen | Werge |
|---|---|---|
| Instruments | 1 (ES future) | 15 across 4 asset classes |
| Frequency | 1-minute intraday | Daily |
| Feature engineering | Raw returns | EWMM of order 1 and 2, with adjustable span s |
| Regime stickiness | Fixed via transition matrix | Controllable via feature span s ∈ {15, 30, 60} |
| Prediction metric | Raw return forecast Δŷₜ | Predicted Expected Sharpe Ratio (PESR) |
| Strategies tested | Momentum signal | Long-only and long/short with transaction costs |
| Published benchmarks | Sharpe ratios for 5 models (Figure 8) | Sharpe ratios for 15 instruments × 3 spans × 2 strategies (Tables 4–5) |

---

## What You Reproduce From Werge

You do NOT reimplement anything from scratch for this extension. You reuse your existing Christensen codebase (`src/hmm/`) and add only the Werge-specific feature engineering and evaluation metric.

### New code needed (minimal)

**src/data/features.py — add two functions:**

```
ewmm(returns, order, span)
```
Computes the Exponential Weighted Moving Moment of a given order. Order 1 = moving mean, order 2 = moving variance. The recursive formula from Werge §4.1:

> EWMM^i_t = λ · M^i_t + (1−λ) · EWMM^i_{t-1}, where λ = 2/(s+1)

This is a simple recursive filter — structurally identical to the EWMA you already implement for Christensen's volatility estimation.

```
extract_werge_features(returns, span)
```
Computes the feature vector f_s = (f¹_s, f²_s) by concatenating the normalized EWMM of order 1 and 2.

**src/strategy/signals.py — add one function:**

```
compute_pesr(mu, sigma, alpha, A, h=1)
```
Computes the Predicted Expected Sharpe Ratio from Werge Eq. (4.1):

> ESR_j = μ_j(f¹_s) / μ_j(f²_s) for each state j
> PESR = ESR^T · α_{n+h|n}, where α_{n+h|n} = α_{n|n} · A^h

This maps the HMM state probabilities to a single number representing the expected risk-adjusted return h steps ahead.

### Everything else is reused

- `src/hmm/baum_welch.py` — same Baum-Welch, now fed 2D features instead of 1D returns
- `src/hmm/forward.py` — same forward algorithm (just multivariate Gaussian emission instead of univariate)
- `src/hmm/inference.py` — same predict-update loop
- `src/hmm/model_selection.py` — same AIC/BIC
- `src/strategy/backtest.py` — same backtester
- `src/utils/metrics.py` — same Sharpe ratio computation

The only structural change is that the emission distribution becomes **2-dimensional** (mean feature + variance feature) instead of 1-dimensional (return). This means μ_k ∈ ℝ² and Σ_k ∈ ℝ²ˣ² instead of μ_k ∈ ℝ and σ²_k ∈ ℝ. The Baum-Welch update equations are identical — just with matrix operations instead of scalar.

---

## Data

Werge uses 15 futures contracts (anonymized). You can proxy them with ETFs:

| Werge instrument | Asset class | ETF proxy | Ticker |
|---|---|---|---|
| CO1 | Commodity | Oil (WTI crude) | USO |
| CO3 | Commodity | Gold | GLD |
| FX1 | Currency | EUR/USD | FXE |
| FX2 | Currency | GBP/USD | FXB |
| EQ1 | Equity | S&P 500 | SPY |
| EQ2 | Equity | Euro Stoxx 50 | FEZ |
| EQ4 | Equity | Nikkei 225 | EWJ |
| FI1 | Fixed Income | US 10yr Treasury | IEF |
| FI2 | Fixed Income | US 30yr Treasury | TLT |

You don't need all 15. A subset of 6–8 spanning all 4 asset classes is sufficient to test generalization.

**Date range:** Training up to 2012, validation 2012–2016, test 2016–2019. Extend the test period to 2024 as an additional contribution (the paper stops at October 2019 — testing through COVID and the 2022 bear market adds real value).

---

## What You Compare

### Reproduction targets (Werge Tables 4–5, span s=30)

For each instrument, the paper reports annualized return, annualized volatility, Sharpe ratio, maximum drawdown, and daily turnover for the long-only strategy. Your targets for the long-only PESR³₃₀(1):

| Instrument | Werge Sharpe | Your target |
|---|---|---|
| CO1 (oil) | 2.44 | Within same order of magnitude |
| EQ1 (S&P 500) | 1.73 | Similar direction and scale |
| EQ4 (Nikkei) | 2.01 | Similar direction and scale |
| FI1 (10yr) | 2.58 | Similar direction and scale |

You won't match exactly because you're using ETF proxies, not the exact futures contracts. What matters is: do you see the same pattern (Sharpe improvement over buy-and-hold across most instruments)?

### New contribution: extended test period (2019–2024)

Werge's test period ends October 2019. You extend to 2024, which includes:
- COVID crash (March 2020) — extreme regime change
- 2020–2021 bull run — strong momentum
- 2022 bear market / rate hiking cycle — another regime change
- 2023–2024 recovery

This is a genuine contribution: you test whether the regime-switching model survives through market conditions the original author didn't evaluate. If it does well during COVID (detecting the crash regime quickly), that's a strong result. If it fails (too slow to adapt), that motivates Extension A (online/adaptive EM) from the Christensen project idea.

---

## The Multivariate Gaussian Extension

This is the one piece of math that changes from Christensen to Werge. In Christensen, the emission is univariate:

> p(Δyₜ | mₜ = k) = N(Δyₜ; μₖ, σ²ₖ), where μₖ ∈ ℝ, σ²ₖ ∈ ℝ

In Werge, the feature vector is 2-dimensional, so the emission becomes multivariate:

> p(fₜ | zₜ = j) = N(fₜ; μⱼ, Σⱼ), where μⱼ ∈ ℝ², Σⱼ ∈ ℝ²ˣ²

The multivariate Gaussian log-PDF is:

> log N(f; μ, Σ) = −½ [d·log(2π) + log|Σ| + (f−μ)ᵀ Σ⁻¹ (f−μ)]

where d=2 is the dimension and |Σ| is the determinant.

The Baum-Welch M-step updates become:

> μ̂ⱼ = Σₜ γₜ(j) · fₜ / Σₜ γₜ(j) — same formula, but fₜ and μⱼ are vectors
> Σ̂ⱼ = Σₜ γₜ(j) · (fₜ − μ̂ⱼ)(fₜ − μ̂ⱼ)ᵀ / Σₜ γₜ(j) — outer product instead of squared scalar

Everything else (forward-backward, transition matrix update, Viterbi) stays identical. The state posteriors γₜ(k) are still scalars — only the emission computation changes.

### Implementation approach

Make `forward.py` and `baum_welch.py` dimension-agnostic from the start. The emission log-probability function should handle both d=1 and d=2:

```python
def gaussian_log_pdf(x, mu, sigma):
    """
    Works for both univariate (x scalar, sigma scalar)
    and multivariate (x vector, sigma matrix).
    """
    if np.isscalar(sigma) or sigma.ndim == 0:
        # Univariate case (Christensen)
        return -0.5 * np.log(2 * np.pi * sigma) - 0.5 * (x - mu)**2 / sigma
    else:
        # Multivariate case (Werge)
        d = len(mu)
        diff = x - mu
        sign, logdet = np.linalg.slogdet(sigma)
        return -0.5 * (d * np.log(2 * np.pi) + logdet + diff @ np.linalg.solve(sigma, diff))
```

This way, the same codebase handles both papers without duplication.

---

## Experiment Script

```
experiments/07_multi_asset.py
```

This script:
1. Downloads daily data for 6–8 ETF proxies via yfinance
2. Computes EWMM features with span s ∈ {15, 30, 60}
3. Trains a 3-state Gaussian HMM via Baum-Welch (using your existing code with 2D features)
4. Runs inference on the test period (2016–2019, then 2019–2024)
5. Computes PESR and maps to long-only holdings
6. Backtests with 5bps transaction costs
7. Prints a comparison table: your Sharpe vs Werge's published Sharpe
8. Generates cumulative return plots per instrument

---

## How This Fits the Presentation

In the final oral presentation (mid-June), the multi-asset extension is one slide:

**Slide title:** "Generalization to Multi-Asset Daily Data (Werge 2021)"

Content:
- Table comparing your Sharpe ratios to Werge's for 6–8 instruments
- One cumulative return plot for the most interesting instrument (e.g., S&P 500 through COVID)
- Key finding: "The regime-switching framework generalizes across asset classes and frequencies. Performance during COVID [describe result]."

This takes 2–3 minutes of the presentation and demonstrates that your implementation is robust beyond the single instrument in the primary paper.

---

## Connection to Other Extensions

The extensions reinforce each other:

**Extension A (EM vs MCMC, from Christensen):** If the fixed Baum-Welch model struggles during COVID (detected as a sharp regime change it adapts to slowly), this motivates the online/adaptive EM from Extension A. You can show: "the batch model missed the COVID regime change by 5 days; the online model detected it within 2 days."

**Extension C (GMM/k-means benchmark, from Christensen):** You can also run the GMM and k-means baselines on the Werge multi-asset data, answering: "does the Markov temporal structure matter across all asset classes, or only for some?"

The Werge extension creates a rich multi-asset dataset that makes every other extension more compelling.
