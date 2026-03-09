# Slide Deck Outline — HMM Momentum Trading

**Course:** ASPTA, UPC Barcelona
**Paper:** Christensen, Turner & Godsill (2020), arXiv:2006.08307

---

## Slide 1: Title

- Title: *Hidden Markov Models for Momentum Trading*
- Subtitle: Reproducing Christensen, Turner & Godsill (2020)
- Course, student name, date

---

## Slide 2: Motivation

- Financial returns exhibit **regime-switching** behavior (bull / bear markets)
- A single Gaussian badly fits the data: skewness = -0.80, excess kurtosis = 13.42
- **Goal:** detect latent regimes and trade on them
- **Figure:** `01_return_distribution.png`

---

## Slide 3: HMM Model Definition

- Hidden states: $m_t \in \{1, \ldots, K\}$ (market regimes)
- Observations: $\Delta y_t = \log P_t - \log P_{t-1}$ (log-returns)
- Transition: $A_{ij} = p(m_t = j \mid m_{t-1} = i)$
- Emission: $\Delta y_t \mid m_t = k \sim \mathcal{N}(\mu_k, \sigma^2_k)$
- Parameters: $\Theta = \{A, \pi, \mu, \sigma^2\}$

---

## Slide 4: Forward Algorithm

- Computes $\alpha_t(k) = p(\Delta y_{1:t}, m_t = k \mid \Theta)$
- Recursion (log-space):

$$\log \alpha_t(k) = \text{logsumexp}_i\left[\log \alpha_{t-1}(i) + \log A_{ik}\right] + \log \mathcal{N}(\Delta y_t; \mu_k, \sigma^2_k)$$

- Paper: Section 3.2, Algorithm 1 lines 6-9
- Validated against `hmmlearn` to 1e-6

---

## Slide 5: Backward Algorithm & Forward-Backward

- Backward: $\beta_t(k) = p(\Delta y_{t+1:T} \mid m_t = k, \Theta)$
- Posterior responsibilities:

$$\gamma_t(k) = \frac{\alpha_t(k) \cdot \beta_t(k)}{\sum_j \alpha_t(j) \cdot \beta_t(j)}$$

- $\gamma_t(k) = p(m_t = k \mid \Delta y_{1:T}, \Theta)$
- Paper: Section 3.2, Algorithm 1 lines 11-16

---

## Slide 6: Baum-Welch (EM) Algorithm

- **E-step:** compute $\gamma_t(k)$ and $\xi_t(i,j)$ via forward-backward
- **M-step:** update parameters:

$$\hat{\mu}_k = \frac{\sum_t \gamma_t(k) \Delta y_t}{\sum_t \gamma_t(k)}, \qquad \hat{\sigma}^2_k = \frac{\sum_t \gamma_t(k)(\Delta y_t - \hat{\mu}_k)^2}{\sum_t \gamma_t(k)}$$

- Paper: Section 3.2, Algorithm 1 lines 17-21
- Log-likelihood is **monotonically non-decreasing** (verified in tests)

---

## Slide 7: Model Selection

- AIC and BIC both favor **K=4**, with K=3 as close runner-up
- K=3 used for interpretability: bearish / neutral / bullish
- K=3 vs K=4 backtest shows virtually identical Sharpe (0.54 vs 0.53)
- **Figure:** `02_model_selection.png`

---

## Slide 8: EM Convergence & Learned Parameters

- Converges in ~73 iterations, +1003 nats improvement
- 3 regimes recovered:

| State | Ann. Return | Ann. Vol | Time in State |
|-------|-------------|----------|---------------|
| Bearish | -145% | 56.5% | 3.6% |
| Neutral | ~0% | 19.6% | 40.4% |
| Bullish | +28% | 8.5% | 56.0% |

- Bearish state has 7x volatility of bullish (leverage effect)
- **Figure:** `03_em_convergence.png`

---

## Slide 9: Regime Detection (Viterbi)

- Viterbi decoding: $m_{1:T}^* = \arg\max p(m_{1:T} \mid \Delta y_{1:T}, \Theta)$
- Bearish episodes align with COVID-19 (2020) and 2022 drawdown
- Bullish regime dominates uptrend periods
- **Figure:** `04_regime_prices.png`

---

## Slide 10: Online Inference for Trading

- Causal (real-time) filtering, not batch smoothing
- Predict-update loop (Paper Section 6, Algorithm 4):
  - Predict: $p(m_t \mid \Delta y_{1:t-1}) = \sum_i p(m_{t-1} = i \mid \Delta y_{1:t-1}) A_{ik}$
  - Update: $p(m_t \mid \Delta y_{1:t}) \propto p(m_t \mid \Delta y_{1:t-1}) \cdot \mathcal{N}(\Delta y_t; \mu_k, \sigma^2_k)$
- Produces filtered state probabilities $\omega_{t,k}$ at each time step

---

## Slide 11: Trading Strategy & Backtest

- **Weighted vote signal:** $s_t = \sum_k \omega_{t,k} \cdot \text{sign}(\mu_k)$
- One-period execution lag: $r_t^{\text{strat}} = s_{t-1} \cdot \Delta y_t$
- 5 bps transaction costs

| Strategy | Sharpe | Ann. Return | Max Drawdown |
|----------|--------|-------------|--------------|
| Weighted vote | 0.54 | 7.77% | 20.33% |
| Buy-and-hold | 0.40 | 5.57% | 27.06% |

- **Figure:** `05_backtest_comparison.png`

---

## Slide 12: Extension A — EM vs MCMC

- Gibbs sampler with conjugate priors (Dirichlet, Normal, Inverse-Gamma)
- FFBS for state sampling, 150 samples after burn-in
- EM and MCMC produce nearly identical results (Sharpe 0.54 vs 0.52)
- MCMC provides uncertainty estimates: bearish state has highest posterior std
- **Figure:** `06_mcmc_mu_posterior.png`

---

## Slide 13: Extension B — Signal Refinement

- **No-trade zone:** go flat when neutral posterior > threshold $\tau$
- **EMA smoothing:** $s''_t = \alpha s'_t + (1-\alpha) s''_{t-1}$
- Grid search over $(\tau, \alpha)$: best at $\tau = 0.6$, $\alpha = 0.1$
- Sharpe 0.71 (+31%), max drawdown 7.8% (-62% reduction)
- **Figure:** `10_signal_grid_sharpe.png`

---

## Slide 14: Robustness

- **Expanding-window backtest** (10 windows, 2020-2024): rolling Sharpe 0.71 vs BH 0.57
- **Multi-asset:** QQQ best (Sharpe 0.78), TLT bearish posterior negatively correlated with equities
- **Cross-ticker test** (6 configs): 50% win rate by Sharpe, 67% by max drawdown
- Model excels in drawdowns, lags in sustained bull markets
- **Figures:** `09_rolling_cumulative.png`, `11_robustness_sharpe.png`

---

## Slide 15: Conclusions & Limitations

**Conclusions:**
- HMM successfully identifies 3 economically meaningful regimes
- Weighted vote signal improves Sharpe by 35% and reduces max drawdown by 25%
- Signal refinement further boosts risk-adjusted returns
- EM point estimates are sufficient (MCMC confirms robustness)

**Limitations:**
- Underperforms in sustained bull markets (model's caution reduces exposure)
- Gaussian emissions may not capture extreme tail events
- Single-asset model; cross-asset dependencies not jointly modeled

**Future work:** multivariate HMM, non-Gaussian emissions, intraday data
