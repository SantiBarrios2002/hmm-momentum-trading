# ASPTA Project Idea: HMM for Intraday Momentum Trading

**Paper:** Christensen, Turner & Godsill (2020), "Hidden Markov Models Applied To Intraday Momentum Trading With Side Information"
arXiv:2006.08307 — Signal Processing and Communications Laboratory + Machine Learning Group, Cambridge University.

---

## Why This Paper Over the Werge Paper

This paper comes from a **signal processing lab**, not a finance department. The language, notation, and framing are native to your ASPTA training. The paper explicitly connects HMMs to digital filtering, Kalman filters, and particle filters — all ASPTA topics. It also covers **three** different parameter estimation methods (Piecewise Linear Regression, Baum-Welch/EM, and MCMC), giving you more estimation theory to discuss than the Werge paper. The model itself treats momentum as a **latent state to be estimated from noisy observations** — this is literally the core problem of statistical signal processing.

The trade-off: the paper uses intraday (1-minute) e-mini S&P500 futures data from 2011. You may not have access to identical data, but you can adapt the methodology to daily data (which is freely available) or use free intraday data sources.

---

## The Model in Signal Processing Language

The paper formulates a **state-space model** where:

- **Hidden state** mₜ ∈ {1, ..., K}: the latent momentum (trend) — discretized on a price grid
- **Observation** Δyₜ = log(yₜ/yₜ₋₁): the noisy log-return
- **Observation equation:** Δyₜ = μ_mₜ + εₜ, where εₜ ~ N(0, σ²_mₜ) — the return equals the trend plus Gaussian noise
- **State dynamics:** mₜ | mₜ₋₁ ~ Categorical(A[mₜ₋₁, :]) — the trend evolves as a Markov chain

This is identical in structure to a communications channel where you're trying to detect a transmitted symbol (the trend state) from a noisy received signal (the return). The paper explicitly notes: *"when the state variables become continuous and Gaussian, the problem can be solved by a Kalman filter, and when continuous and non-Gaussian, by a particle filter."*

---

## Mathematical Concepts: ASPTA ↔ Paper Mapping

### 1. Maximum Likelihood Estimation (Module 1.2)

**In the paper (§3.2, Algorithm 1):**
The Baum-Welch algorithm finds Θ̂ = argmax_Θ p(ΔY | Θ) — the MLE of the model parameters. Since the hidden states M are unobserved, direct maximization is intractable (requires enumerating S^T state sequences). Baum-Welch resolves this via EM.

**M-step update equations (Algorithm 1, lines 18–21):**
- μ̂(k) = Σₜ γₜ(k) · zₜ / Σₜ γₜ(k) — weighted sample mean
- σ̂²(k) = Σₜ γₜ(k) · (zₜ - μₖ)² / Σₜ γₜ(k) — weighted sample variance
- Â = normalized expected transition counts

These are the standard Gaussian MLE formulas weighted by the state posteriors γₜ(k) from the E-step.

**What to study:** MLE for Gaussian parameters. Understand that the Baum-Welch updates are simply "soft-assignment" versions of the standard MLE — instead of assigning each observation to one cluster, you weight by the posterior probability of belonging to each state.

---

### 2. EM Algorithm / Baum-Welch (Module 1.2)

**In the paper (§3.2):**
The EM algorithm iterates:
- **E-step:** Run the forward-backward algorithm to compute γₜ(k) = p(mₜ = k | Δy₁:T, Θ_old) — the smoothing distribution over hidden states.
- **M-step:** Update Θ = {A, φ} using the occupation probabilities γₜ(k) as weights in the MLE formulas.

The paper notes the key limitation: *"Baum-Welch does not always converge on the global maxima"* — handled by running from multiple initializations and keeping the best.

**What to study:** The Q-function Q(Θ, Θ̄) = E[log p(ΔY, M | Θ) | ΔY, Θ̄], the guarantee that likelihood is non-decreasing at each step (via Jensen's inequality), and the local convergence property.

---

### 3. Forward-Backward Algorithm — Bayesian Recursive Filtering (Module 1.3 + Module 2.2)

**In the paper (§3.2, Algorithm 1 lines 6–14; §6, Algorithm 4):**

The forward pass computes:
> αₜ(k) = Σ_{mₜ₋₁} p(Δyₜ | mₜ = k) · p(mₜ = k | mₜ₋₁) · αₜ₋₁(mₜ₋₁)

The backward pass computes:
> βₜ(k) = Σ_{mₜ₊₁} p(Δyₜ₊₁ | mₜ₊₁) · p(mₜ₊₁ | mₜ = k) · βₜ₊₁(mₜ₊₁)

The smoothing distribution is: γₜ(k) = αₜ(k) · βₜ(k) / p(Δy₁:T)

**This is exactly the predict-update cycle of a Bayesian filter** applied to discrete states. The paper's inference algorithm (Algorithm 4) makes this crystal clear:

> **Predict:** ωₜ|ₜ₋₁,ₖ = Σ_{k'} a_{kk'} · ωₜ₋₁|ₜ₋₁,ₖ' (propagate through transition matrix)
> **Update:** ωₜ|ₜ,ₖ ∝ ωₜ|ₜ₋₁,ₖ × N(Δyₜ; μₖ, σ²ₖ) (incorporate new observation via Bayes' rule)
> **Prediction:** Δŷₜ = Σₖ ωₜ|ₜ₋₁,ₖ × μ*ₖ (weighted expectation)

**Connection to Kalman filter (Module 2.2):** This is the discrete-state version of the Kalman predict-update equations. Replace discrete states with a continuous Gaussian state, and the transition matrix with a linear state model, and you recover the Kalman filter. The paper explicitly states this in §8.1.

**What to study:** Understand Bayesian filtering as predict → update. Review the Kalman filter equations and see the structural parallel. Understand log-space implementation for numerical stability.

---

### 4. MCMC / Bayesian Parameter Estimation (Module 1.3)

**In the paper (§3.3):**
As an alternative to MLE/EM, the paper estimates parameters using Metropolis-Hastings MCMC — a fully Bayesian approach:

> p(Θ | ΔY) ∝ p(ΔY | Θ) × p(Θ)

where p(Θ) is a proper prior (Dirichlet on transition rows, hierarchical prior on variances). MCMC draws samples from the posterior p(Θ | ΔY), giving a distributional estimate rather than a point estimate.

**MAP estimation:** The paper uses the posterior mode as the point estimate: Θ̂_MAP = argmax_Θ log p(ΔY | Θ) + log p(Θ), approximated by the MCMC draw with the highest posterior probability.

**Model selection via marginal likelihood:** p(ΔY | Mₖ) = ∫ p(ΔY | Θ, Mₖ) p(Θ | Mₖ) dΘ, computed via bridge sampling (Figure 3). This is the Bayesian alternative to AIC/BIC.

**Key finding:** MCMC performed worse than Baum-Welch in practice (Figure 8), likely due to prior sensitivity and poor mixing. This is an honest, practically useful result.

**What to study:** Bayes' theorem applied to parameter estimation. The concept of prior → posterior. How MAP relates to MLE (MAP = MLE with flat prior). You do NOT need to deeply understand MCMC implementation — understanding the conceptual framework is sufficient.

---

### 5. Model Selection / Information Criteria (Module 1.1)

**In the paper (§2.3.4, §3.2, Figure 2):**
Three methods for choosing K (number of hidden states):
- Cross-validation (PLR): K = 2
- AIC/BIC (Baum-Welch): K = 3 (Figure 2)
- Marginal likelihood via bridge sampling (MCMC): K = 3 (Figure 3)

BIC = −2 log(L) + p log(n), where p is the number of free parameters. The paper counts parameters explicitly: K states × K(K−1) transition parameters + K mean parameters + K variance parameters.

**Connection to CRLB:** More parameters → each is estimated with higher variance given fixed data. BIC penalizes complexity to avoid overfitting.

**What to study:** AIC vs BIC formulas. Understand that BIC has a stronger complexity penalty and is consistent for model selection.

---

### 6. Signal Detection (Module 3)

**In the paper (§2.2, §6):**
At each time step, the model performs a **detection problem**: given the noisy return Δyₜ, decide which of K momentum states generated it. The Viterbi algorithm finds the MAP state sequence (optimal joint detection), while the forward algorithm gives the marginal MAP at each time (filtering).

The paper frames this as **detecting a trend signal in noise** — the observation equation Δyₜ = μ_mₜ + εₜ is a classic detection problem: is the mean positive (uptrend), negative (downtrend), or zero (no trend)?

**What to study:** Hypothesis testing with known PDFs. Likelihood ratio tests. The connection between Bayesian filtering (soft detection via posterior probabilities) and hard detection (Viterbi).

---

### 7. Digital Filtering Connection (Module 2.1)

**In the paper (§1, §8.1):**
The paper explicitly compares the HMM approach to the MACD (Moving Average Convergence-Divergence) digital filter, which is the industry standard for trend detection. MACD uses **cascaded low-pass filters** to estimate the trend, but suffers from **time-lagging** due to the filter's frequency response. The HMM avoids this by modelling the trend as a latent state, allowing instantaneous regime changes.

The paper also connects EWMA volatility estimation (Equation 4) to recursive filtering:
> σ²ₜ₊₁|ₜ = (1−λ) Σ λ^τ Δy²ₜ₋τ

This is a first-order IIR filter on squared returns — structurally identical to Recursive Least Squares (Module 2.1).

**What to study:** Understand EWMA as a recursive filter with forgetting factor λ. The trade-off between reactivity and smoothness.

---

### 8. Input-Output HMM / Side Information (Extension of Module 1.2)

**In the paper (§5, Algorithm 3):**
The IOHMM extends the standard HMM by conditioning the transition matrix on external information:

> p(mₜ | mₜ₋₁, xₜ) instead of p(mₜ | mₜ₋₁)

Different transition matrices {A₁, ..., A_R} are learned for different values of the side information xₜ (discretized via spline roots). This is learned using a modified Baum-Welch where the forward-backward recursions become:

> αₜ(k) = Σ_{mₜ₋₁} p(Δyₜ | mₜ, xₜ) · p(mₜ | mₜ₋₁, xₜ₋₁) · αₜ₋₁(mₜ₋₁)

**What to study:** This is the most advanced concept in the paper. Understand it as: "the dynamics of the hidden state depend on an observable external signal." The estimation is still EM/Baum-Welch, just with per-bucket transition matrices.

---

## Summary Table

| Paper concept | Paper section | ASPTA Module | Core idea |
|---|---|---|---|
| State-space model (trend + noise) | §2.2, Eq. (2) | Foundation | Δyₜ = μ_mₜ + εₜ — detect latent trend from noisy returns |
| MLE via Baum-Welch (EM) | §3.2, Algorithm 1 | 1.2 | E-step: forward-backward for γₜ(k). M-step: weighted Gaussian MLE |
| MCMC / MAP estimation | §3.3 | 1.3 | Bayesian parameter estimation with Dirichlet/hierarchical priors |
| Model selection (AIC/BIC/marginal likelihood) | §2.3.4, Figures 2–3 | 1.1 | Penalized likelihood and Bayes factors for choosing K |
| Forward algorithm (Bayesian filtering) | §6, Algorithm 4 | 1.3 + 2.2 | Predict-update cycle: discrete analog of Kalman filter |
| Detection of trend states | §6 | 3.1 | Deciding which of K hypotheses generated the observation |
| EWMA volatility (recursive filter) | §4.2, Eq. (4) | 2.1 | First-order IIR filter on squared returns |
| Comparison to digital filters (MACD) | §1, §8.1 | 2.1 | HMM avoids time-lagging of cascaded low-pass filters |
| IOHMM with side information | §5, Algorithm 3 | 1.2 (extension) | Transition matrix conditioned on external observable signal |
| Kalman/particle filter connection | §8.1 | 2.2 / 2.3 | Continuous-state limit recovers Kalman; non-Gaussian → particle filter |

---

## What You'd Reproduce (Phase 1)

### Core reproduction target: Figures 2, 3, and 8

1. **Implement the 3-state Gaussian HMM** with Baum-Welch from scratch in Python. Observation equation: Δyₜ = μ_mₜ + εₜ, discretized Gaussian emission.
2. **Reproduce Figure 2:** Fit HMMs with K = 1, ..., 10. Compute AIC and BIC for each. Show K=3 is optimal.
3. **Reproduce PLR baseline:** Implement piecewise linear regression to find change points and initialize parameters (the "naive" baseline from §3.1).
4. **Reproduce Figure 8 (partial):** Compare the Default HMM, PLR HMM, and Baum-Welch HMM strategies on out-of-sample data. Report Sharpe ratios and cumulative returns.
5. **Reproduce the forward-algorithm inference** (Algorithm 4): the predict-update loop that generates trading signals.

**Data adaptation:** If you can't get 1-minute ES futures data, adapt the methodology to:
- Daily S&P 500 data (free via yfinance) — change the sampling frequency, adjust window sizes proportionally (23 days → keep as-is for daily data).
- Or use free intraday data from Yahoo Finance (5-minute bars for SPY) for a closer match.

**Key quantitative targets:** The paper reports pre-cost Sharpe ratios around 2.0 for the Baum-Welch HMM and IOHMM models. For daily data, expect lower Sharpe ratios (the momentum effect is weaker at daily frequency).

---

## How You'd Extend It (Phase 2)

### Extension A: Baum-Welch vs. MCMC — Why Does EM Win? ⭐ Recommended

**What:** The paper's most surprising finding is that Baum-Welch outperforms MCMC (Figure 8). The authors list several hypotheses (§7) but don't resolve the question. You can systematically investigate this.

**How:**
- Implement a basic Metropolis-Hastings sampler for the HMM parameters (doesn't need to be production-grade — use the PyMC library or write a simple MH loop).
- Test the paper's hypotheses: vary the prior strength (diffuse vs. informative Dirichlet), vary MCMC chain length, try initializing MCMC from the Baum-Welch solution.
- Compare the resulting trading Sharpe ratios.

**Why it's strong for the grade:** This is a genuine open question the paper poses. Your professor will appreciate systematic investigation of estimation method performance — it's the core of Module 1.

### Extension B: Adapt to Daily Data + Multiple Assets

**What:** The paper only uses one instrument (ES future) at one frequency (1-minute). Apply the same methodology to daily data across multiple assets (equity indices, FX, commodities) and evaluate whether the momentum HMM generalizes.

**How:** Download daily data for SPY, EEM, GLD, EUR/USD, etc. Apply the same pipeline. Compare regime detection and trading performance across asset classes.

**Why it's strong:** Tests the generalization of the approach. Easy to implement since you already have the codebase.

### Extension C: Online/Adaptive Baum-Welch

**What:** The paper uses batch learning (train on past data, then apply fixed parameters). Implement an online EM variant where parameters adapt as new data arrives.

**Why it's strong:** The paper uses rolling windows of 23 days for mean estimation — an online EM would formalize this and potentially improve adaptation.

### Extension D: Implement the IOHMM with Alternative Side Information

**What:** The paper uses volatility ratio and seasonality as side information. You could use alternative signals: VIX (implied volatility), trading volume, or cross-asset momentum (e.g., bond momentum predicting equity momentum).

**Why it's strong:** Tests the generality of the IOHMM framework with different predictors.

---

## Recommended Extension Combination

> **Reproduce Baum-Welch HMM on daily data (Phase 1) + Extension A (EM vs. MCMC investigation) + Extension B (multiple assets)**

This gives you:
1. Core EM/Baum-Welch implementation (Module 1.2)
2. Bayesian estimation comparison (Module 1.3)
3. Practical evaluation across assets (shows depth)
4. A genuine open question from the paper (shows initiative)

---

## Reading Order (Before the Paper)

1. **MLE for Gaussians** — derive μ̂ and σ̂² from scratch (30 min)
2. **EM algorithm** — ASPTA Module 1.2 slides. Focus on the Q-function (1 hour)
3. **Bayesian estimation / MAP** — ASPTA Module 1.3. Understand prior × likelihood = posterior (30 min)
4. **Rabiner (1989) Sections III–V** — the classic HMM tutorial. Covers forward-backward, Baum-Welch, and Viterbi (2 hours)
5. **Kalman filter predict-update** — ASPTA Module 2.2. See the structural parallel with the forward algorithm (30 min)
6. **Read the paper** — Sections 1–3 and 6–7 are essential. Section 4–5 (side information/IOHMM) can be skimmed initially (2–3 hours)

**Total prep: ~7 hours of focused reading.**

---

## Data Sources

| Source | What you get | Cost |
|---|---|---|
| yfinance (Python) | Daily OHLCV for SPY, ETFs, FX | Free |
| Yahoo Finance | 5-min intraday bars (last 60 days) | Free |
| Polygon.io | 1-min historical bars | Free tier (limited), $29/mo for more |
| Alpha Vantage | 1-min and daily data | Free (5 calls/min) |
| FirstRate Data | Historical 1-min ES futures | ~$20 one-time purchase |
