# Prerequisite Concepts Review
## For reading: Christensen, Turner & Godsill (2020) — HMM Momentum Trading

**How to use this document:** Work through each section in order. Each concept is presented as: what it is → the math → intuition → how it appears in the paper. Sections are ordered so that later ones build on earlier ones.

**Estimated time:** 8–10 hours of focused study. Skip what you already know.

---

## 1. Probability Foundations You'll Need Everywhere

### 1.1 Joint, Marginal, and Conditional Probability

The entire paper is about computing **conditional distributions** of hidden states given observations. Make sure these are reflexive:

**Joint:** p(x, z) — the probability of observing x AND being in state z simultaneously.

**Marginal:** p(x) = Σ_z p(x, z) — sum out (marginalize) the variable you don't care about. This is how you go from the joint distribution over (observations, states) to just the distribution over observations.

**Conditional:** p(z | x) = p(x, z) / p(x) — the probability of state z given that you observed x. Rearranging: p(x, z) = p(z | x) · p(x) = p(x | z) · p(z).

**Chain rule:** p(x₁, x₂, ..., xₙ) = p(x₁) · p(x₂|x₁) · p(x₃|x₁,x₂) · ... This is how the HMM joint distribution factorizes.

**Where in the paper:** Everywhere. The core quantity of interest is p(mₜ | Δy₁:ₜ) — the conditional probability of the momentum state given all observed returns up to now.

### 1.2 Bayes' Theorem

> p(θ | x) = p(x | θ) · p(θ) / p(x)

- **p(θ)** = prior — what you believe before seeing data
- **p(x | θ)** = likelihood — how probable the data is under parameter θ
- **p(θ | x)** = posterior — your updated belief after seeing data
- **p(x)** = evidence/marginal likelihood — normalizing constant

**Intuition:** You start with a prior belief, observe data, and update your belief proportionally to how well each hypothesis explains the data.

**Where in the paper:** The inference algorithm (Algorithm 4) is literally Bayes' theorem applied recursively. The update step is: p(mₜ = k | Δy₁:ₜ) ∝ p(mₜ = k | Δy₁:ₜ₋₁) × p(Δyₜ | mₜ = k). Prior × Likelihood.

### 1.3 Gaussian Distribution

The emission model in the paper is Gaussian:

> p(Δyₜ | mₜ = k) = N(Δyₜ; μₖ, σ²ₖ) = (1 / σₖ√(2π)) · exp(−(Δyₜ − μₖ)² / (2σ²ₖ))

Make sure you can:
- Write the PDF from memory
- Compute log N(x; μ, σ²) = −½ log(2π) − log(σ) − (x−μ)²/(2σ²)
- Understand that **the log-likelihood of n i.i.d. Gaussian observations** is: ℓ(μ,σ²) = −n/2 · log(2πσ²) − Σᵢ(xᵢ−μ)²/(2σ²)

**Where in the paper:** Equation (2) and (3) — the observation model. Equation (5) — the likelihood vector used in inference. Algorithm 1 lines 18–20 — the M-step updates.

---

## 2. Markov Chains

### 2.1 The Markov Property

A sequence of random variables m₁, m₂, ..., mₜ is a **Markov chain** if:

> p(mₜ | mₜ₋₁, mₜ₋₂, ..., m₁) = p(mₜ | mₜ₋₁)

The future depends on the past only through the present. This is the key assumption that makes HMMs tractable.

### 2.2 Transition Matrix

For a chain with K states, the dynamics are described by a K×K **transition matrix** A:

> Aᵢⱼ = p(mₜ = j | mₜ₋₁ = i)

Each row sums to 1 (it's a probability distribution over next states). The diagonal elements Aᵢᵢ are the **self-transition** (persistence) probabilities — how likely the chain is to stay in state i.

**Key operation — state propagation:**
If ω is a row vector of current state probabilities [p(mₜ=1), ..., p(mₜ=K)], then the predicted state distribution at t+1 is:

> ω_{t+1|t} = ω_{t|t} · A

This is matrix-vector multiplication. To predict h steps ahead: ω_{t+h|t} = ω_{t|t} · Aʰ.

**Where in the paper:** §2.3.1. The transition matrix governs how the momentum state evolves. In the IOHMM extension (§5), A becomes dependent on external information: p(mₜ | mₜ₋₁, xₜ), giving multiple transition matrices {A₁, ..., A_R}.

### 2.3 Initial Distribution

The chain starts from an **initial distribution** π:

> πₖ = p(m₁ = k)

The paper draws π from the **ergodic distribution** of the chain (the stationary distribution satisfying π = π · A).

---

## 3. Hidden Markov Models

### 3.1 The Generative Model

An HMM is a Markov chain where you **can't observe the states directly**. Instead, each state generates an observation through an emission distribution. The full generative process:

1. Draw initial state: m₁ ~ Categorical(π)
2. For t = 1, ..., T:
   - Emit observation: Δyₜ ~ p(Δyₜ | mₜ) = N(Δyₜ; μ_{mₜ}, σ²_{mₜ})
   - Transition to next state: mₜ₊₁ ~ Categorical(A[mₜ, :])

The **joint distribution** factorizes as:

> p(ΔY, M) = p(m₁) · [∏ₜ₌₂ᵀ p(mₜ | mₜ₋₁)] · [∏ₜ₌₁ᵀ p(Δyₜ | mₜ)]

This is the equation in §2.2 of the paper. The three terms are: initial state × transitions × emissions.

### 3.2 The Three Fundamental Problems

The paper (§2.2) and Rabiner (1989) identify three problems:

**Problem 1 — Evaluation:** Given observations ΔY and parameters Θ, compute p(ΔY | Θ).
→ Solved by the **forward algorithm**.

**Problem 2 — Decoding:** Given ΔY and Θ, find the most likely state sequence M* = argmax_M p(M | ΔY, Θ).
→ Solved by the **Viterbi algorithm**.

**Problem 3 — Learning:** Find Θ* = argmax_Θ p(ΔY | Θ).
→ Solved by **Baum-Welch (EM)**.

### 3.3 Why Direct MLE is Intractable

To compute the likelihood p(ΔY | Θ) you need to sum over ALL possible state sequences:

> p(ΔY | Θ) = Σ_M p(ΔY, M | Θ)

With K states and T time steps, there are K^T possible sequences. For K=3, T=1000: 3^1000 ≈ 10^477 terms. Impossible.

The forward algorithm and EM both avoid this enumeration using the Markov property to decompose the computation into T steps of K² operations each (i.e., O(TK²) complexity).

---

## 4. The Forward-Backward Algorithm

This is the computational heart of everything in the paper. Understand this well.

### 4.1 Forward Pass (Filtering)

Define the **forward variable:**

> αₜ(k) = p(Δy₁, Δy₂, ..., Δyₜ, mₜ = k | Θ)

This is the joint probability of having observed the first t returns AND being in state k at time t.

**Initialization:**
> α₁(k) = πₖ · p(Δy₁ | m₁ = k) = πₖ · N(Δy₁; μₖ, σ²ₖ)

**Recursion (for t = 2, ..., T):**
> αₜ(k) = [Σᵢ αₜ₋₁(i) · Aᵢₖ] · p(Δyₜ | mₜ = k)

Read this as two steps:
1. **Predict:** Σᵢ αₜ₋₁(i) · Aᵢₖ — propagate previous forward variables through the transition matrix
2. **Update:** multiply by the emission probability p(Δyₜ | mₜ = k) — incorporate the new observation

**The likelihood** falls out as a byproduct:
> p(ΔY | Θ) = Σₖ αₜ(k)

**Where in the paper:** Algorithm 1 lines 6–9. Algorithm 4 is the forward-only version used for real-time inference.

### 4.2 Backward Pass (Smoothing)

Define the **backward variable:**

> βₜ(k) = p(Δyₜ₊₁, ..., ΔyT | mₜ = k, Θ)

The probability of the future observations given that you're in state k at time t.

**Initialization:** βT(k) = 1 for all k.

**Recursion (for t = T−1, ..., 1):**
> βₜ(k) = Σⱼ Aₖⱼ · p(Δyₜ₊₁ | mₜ₊₁ = j) · βₜ₊₁(j)

**Where in the paper:** Algorithm 1 lines 11–14.

### 4.3 State Posteriors (Occupation Probabilities)

Combining forward and backward:

> γₜ(k) = p(mₜ = k | ΔY, Θ) = αₜ(k) · βₜ(k) / p(ΔY | Θ)

This is the **posterior probability** of being in state k at time t, given ALL the data (past and future). These γₜ(k) are the "soft assignments" used in the EM M-step.

**Where in the paper:** Algorithm 1 line 16.

### 4.4 Numerical Stability: Log-Space

In practice, αₜ(k) and βₜ(k) become astronomically small (products of hundreds of probabilities < 1). You must work in **log-space:**

> log αₜ(k) = log [Σᵢ exp(log αₜ₋₁(i) + log Aᵢₖ)] + log p(Δyₜ | k)

The inner sum uses the **log-sum-exp** trick: log Σᵢ exp(aᵢ) = max(a) + log Σᵢ exp(aᵢ − max(a)). The paper notes this in §3.2, citing Kingsbury & Rayner (1971).

---

## 5. Maximum Likelihood Estimation

### 5.1 MLE for a Gaussian

Given n observations x₁, ..., xₙ from N(μ, σ²):

> μ̂_ML = (1/n) Σᵢ xᵢ
> σ̂²_ML = (1/n) Σᵢ (xᵢ − μ̂)²

Derived by setting ∂ℓ/∂μ = 0 and ∂ℓ/∂σ² = 0. You should be able to do this derivation.

### 5.2 MLE with Soft Assignments (Weighted MLE)

When each observation has a **weight** γᵢ (the probability of belonging to a particular component), the MLE becomes:

> μ̂ₖ = Σᵢ γᵢ(k) · xᵢ / Σᵢ γᵢ(k)
> σ̂²ₖ = Σᵢ γᵢ(k) · (xᵢ − μ̂ₖ)² / Σᵢ γᵢ(k)

These are the **Baum-Welch M-step equations** — Algorithm 1 lines 18–19. The γₜ(k) are the occupation probabilities from the forward-backward algorithm.

**Intuition:** It's the same as regular MLE, but instead of counting observations assigned to cluster k, you weight by the probability of belonging to cluster k.

---

## 6. The EM Algorithm

### 6.1 The General Framework

EM handles MLE when there are **latent variables** (the hidden states M). Since we can't maximize p(ΔY | Θ) directly (it requires summing over all M), EM introduces an auxiliary function:

> Q(Θ, Θ_old) = E_{M|ΔY,Θ_old} [log p(ΔY, M | Θ)]

This is the **expected complete-data log-likelihood**, where the expectation is taken over the posterior p(M | ΔY, Θ_old) computed using the old parameters.

**E-step:** Compute Q(Θ, Θ_old) — in practice, this means computing the posteriors γₜ(k) using forward-backward.

**M-step:** Find Θ_new = argmax_Θ Q(Θ, Θ_old) — in practice, the weighted MLE formulas above.

### 6.2 Why It Works

**Key theorem:** At each EM iteration, p(ΔY | Θ_new) ≥ p(ΔY | Θ_old). The observed-data likelihood never decreases.

**Proof sketch:** Uses Jensen's inequality on the KL divergence between the true posterior and the approximation. You don't need to prove this, but understand the statement.

**Caveat:** EM converges to a **local** maximum, not necessarily the global one. The paper handles this by running Baum-Welch from multiple random initializations (§3.2).

### 6.3 Baum-Welch = EM for HMMs

Baum-Welch is just EM where:
- The E-step is the forward-backward algorithm → produces γₜ(k)
- The M-step is the weighted MLE formulas → updates μₖ, σ²ₖ, Aᵢⱼ

The complete algorithm is Algorithm 1 in the paper.

---

## 7. Bayesian Estimation and MCMC

### 7.1 MAP Estimation

Instead of maximizing the likelihood alone, MAP adds a prior:

> Θ̂_MAP = argmax_Θ [log p(ΔY | Θ) + log p(Θ)]

When p(Θ) is uniform (flat prior), MAP = MLE. With an informative prior, MAP is "regularized MLE."

### 7.2 The Dirichlet Prior

The paper places a **Dirichlet prior** on each row of the transition matrix A. The Dirichlet is the conjugate prior for categorical distributions:

> Dir(a₁, ..., aₖ) — parameterized by "pseudo-counts" eᵢⱼ

The paper uses eᵢᵢ = 4 (high self-transition) and eᵢⱼ = 1/(K−1) for i≠j, encoding the belief that the chain is **sticky** (tends to stay in the same state).

**What you need to know:** The Dirichlet is like having "virtual observations" of transitions before seeing any real data. Large eᵢᵢ means "I believe state i is persistent."

### 7.3 MCMC (Metropolis-Hastings) — Conceptual Understanding

MCMC generates samples Θ⁽¹⁾, Θ⁽²⁾, ..., Θ⁽ⁿ⁾ from the posterior p(Θ | ΔY). The Metropolis-Hastings algorithm:

1. Start at some Θ_current
2. Propose Θ_new from a proposal distribution q(Θ_new | Θ_current)
3. Compute acceptance ratio: α = [p(ΔY | Θ_new) · p(Θ_new) · q(Θ_current | Θ_new)] / [p(ΔY | Θ_current) · p(Θ_current) · q(Θ_new | Θ_current)]
4. Accept Θ_new with probability min(1, α); otherwise keep Θ_current
5. Repeat

After a **burn-in** period (the paper uses 2,000 draws), the samples approximate the posterior.

**You don't need to implement this from scratch** for the project (you can use PyMC or emcee). But understand: MCMC gives you a distribution over Θ, while EM gives you a single point estimate. The paper's key finding is that the EM point estimate works better in practice (§7, Figure 8).

### 7.4 Marginal Likelihood and Model Selection

The **marginal likelihood** (or evidence) for model Mₖ with K states:

> p(ΔY | Mₖ) = ∫ p(ΔY | Θ, Mₖ) · p(Θ | Mₖ) dΘ

This integral is intractable. The paper approximates it via **bridge sampling** (§3.3, Figure 3). The Bayes factor between two models is B = p(ΔY | M₁) / p(ΔY | M₂).

**You mainly need to understand this conceptually** — the marginal likelihood balances fit (how well the model explains the data) against complexity (how many parameters it has), analogous to BIC but more principled.

---

## 8. Model Selection: AIC and BIC

### 8.1 The Formulas

> AIC = −2 log L̂ + 2p
> BIC = −2 log L̂ + p log(n)

where L̂ is the maximized likelihood, p is the number of free parameters, and n is the number of observations.

**Lower is better** — you want high likelihood (good fit) with few parameters (low complexity).

### 8.2 Parameter Counting for an HMM

For a K-state HMM with univariate Gaussian emissions:
- Transition matrix A: K(K−1) parameters (each row sums to 1, so K−1 free per row)
- Emission means: K parameters
- Emission variances: K parameters
- Initial distribution π: K−1 parameters

Total: p = K(K−1) + 2K + (K−1) = K² + K − 1

**Where in the paper:** Figure 2 — AIC and BIC are computed for K = 1 to 50. The maximum (best model) is at K = 3.

---

## 9. The Predict-Update Cycle (Kalman Filter Connection)

### 9.1 The General Principle

All recursive Bayesian filters share the same structure:

**Predict:** Use the system dynamics to propagate the belief forward in time.
> p(mₜ | Δy₁:ₜ₋₁) = Σ_{mₜ₋₁} p(mₜ | mₜ₋₁) · p(mₜ₋₁ | Δy₁:ₜ₋₁)

**Update:** Incorporate the new observation via Bayes' rule.
> p(mₜ | Δy₁:ₜ) ∝ p(Δyₜ | mₜ) · p(mₜ | Δy₁:ₜ₋₁)

### 9.2 In the HMM (Paper Algorithm 4)

> **Predict:** ωₜ|ₜ₋₁,ₖ = Σ_{k'} aₖₖ' · ωₜ₋₁|ₜ₋₁,ₖ'
> **Update:** ωₜ|ₜ,ₖ = ωₜ|ₜ₋₁,ₖ · N(Δyₜ; μₖ, σ²ₖ) / Σ_{k'} ωₜ|ₜ₋₁,ₖ' · N(Δyₜ; μₖ', σ²ₖ')
> **Output:** Δŷₜ = Σₖ ωₜ|ₜ₋₁,ₖ · μ*ₖ

### 9.3 In the Kalman Filter (ASPTA Module 2.2)

For a linear-Gaussian state-space model xₜ = F·xₜ₋₁ + wₜ, yₜ = H·xₜ + vₜ:

> **Predict:** x̂ₜ|ₜ₋₁ = F · x̂ₜ₋₁|ₜ₋₁, Pₜ|ₜ₋₁ = F · Pₜ₋₁|ₜ₋₁ · Fᵀ + Q
> **Update:** Kₜ = Pₜ|ₜ₋₁ · Hᵀ · (H · Pₜ|ₜ₋₁ · Hᵀ + R)⁻¹, x̂ₜ|ₜ = x̂ₜ|ₜ₋₁ + Kₜ(yₜ − H · x̂ₜ|ₜ₋₁)

### 9.4 The Structural Parallel

| HMM forward algorithm | Kalman filter |
|---|---|
| State space: discrete {1, ..., K} | State space: continuous ℝⁿ |
| State distribution: categorical vector ω | State distribution: Gaussian (x̂, P) |
| Predict: ω · A (matrix-vector multiply) | Predict: F · x̂, F·P·Fᵀ + Q |
| Update: ω ⊙ likelihood, then normalize | Update: Kalman gain K, x̂ + K·innovation |
| Emission: N(Δyₜ; μₖ, σ²ₖ) per state | Emission: N(yₜ; H·xₜ, R) |

The paper explicitly states (§8.1): *"when the state variables become continuous and Gaussian, the problem can be solved by a Kalman filter."*

---

## 10. Detection Theory Connection

### 10.1 Hypothesis Testing

At each time step, the model decides between K hypotheses:
- H₁: mₜ = 1 (negative trend), observation from N(μ₁, σ²₁)
- H₂: mₜ = 2 (zero trend), observation from N(μ₂, σ²₂)
- H₃: mₜ = 3 (positive trend), observation from N(μ₃, σ²₃)

The **Bayesian detector** computes posterior probabilities p(Hₖ | Δyₜ) and selects the hypothesis with highest posterior — this is exactly what the forward algorithm does.

### 10.2 Viterbi Algorithm

Viterbi solves the **MAP sequence detection** problem: find the single best path through all K^T possible state sequences. It uses dynamic programming:

> δₜ(k) = max_{m₁:ₜ₋₁} p(m₁, ..., mₜ₋₁, mₜ = k, Δy₁:ₜ | Θ)

Recursion: δₜ(k) = max_i [δₜ₋₁(i) · Aᵢₖ] · p(Δyₜ | k)

This is like the forward algorithm, but with **max** replacing **sum**. In communications, this is the same algorithm used for decoding convolutional codes.

---

## 11. EWMA and Recursive Filtering

### 11.1 Exponential Weighted Moving Average

The paper uses EWMA for volatility estimation (§4.2, Eq. 4):

> σ²ₜ₊₁|ₜ = (1−λ) Σ_{τ=0}^{ψ} λ^τ · Δy²ₜ₋τ

Recursive form: σ²ₜ₊₁|ₜ = λ · σ²ₜ|ₜ₋₁ + (1−λ) · Δy²ₜ

This is a **first-order IIR filter** on squared returns. The parameter λ ∈ (0,1) controls the forgetting factor:
- λ close to 1 → slow adaptation, long memory, smooth estimate
- λ close to 0 → fast adaptation, short memory, noisy estimate

The paper sets λ = 0.79 for 1-minute data (more reactive than the standard 0.94 for daily data).

### 11.2 Connection to Module 2.1

This is a special case of **Recursive Least Squares** with a forgetting factor. The general RLS recursion:

> θ̂ₜ = θ̂ₜ₋₁ + Kₜ(yₜ − xₜᵀθ̂ₜ₋₁)

For the univariate case of estimating a variance, this reduces to the EWMA formula.

---

## 12. Splines and Side Information (Skim-Level)

The paper uses B-splines (§4.1) to capture non-linear relationships between external predictors (volatility ratio, time-of-day) and returns. You don't need deep spline theory for the project — just understand:

- A spline fits a smooth, non-linear curve through noisy (x, y) data
- The spline is evaluated at new x values to produce predictions
- The roots of the spline define natural "buckets" for discretizing the predictor
- Each bucket gets its own transition matrix in the IOHMM

This section is the most skippable for the core project. Become familiar with it if you plan Extension D (alternative side information).

---

## Recommended Reading Sequence

| Order | Topic | Source | Time |
|---|---|---|---|
| 1 | Gaussian MLE derivation | Kay Vol. I Ch. 7, or any stats textbook | 30 min |
| 2 | Markov chains + transition matrices | Any probability textbook (Blitzstein Ch. 11) | 45 min |
| 3 | HMM structure + three problems | Rabiner (1989) §II–III | 1.5 hours |
| 4 | Forward-backward algorithm | Rabiner (1989) §III.A–B, with pen and paper | 2 hours |
| 5 | EM algorithm (general) | ASPTA Module 1.2 slides, or Bishop Ch. 9.2–9.3 | 1 hour |
| 6 | Baum-Welch = EM for HMMs | Rabiner (1989) §IV, or Bishop Ch. 13.2 | 1 hour |
| 7 | MAP estimation + priors | ASPTA Module 1.3 slides | 30 min |
| 8 | AIC/BIC | Any model selection reference (1 page sufficient) | 15 min |
| 9 | Kalman filter predict-update | ASPTA Module 2.2 slides (focus on structure, not derivation) | 45 min |
| 10 | **Read the paper** | Sections 1–3, 6–7 are essential. §4–5 can be skimmed | 2–3 hours |

**Key reference:** Rabiner, L.R. (1989). "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition." Proceedings of the IEEE, 77(2), 257–286. This is the single most important preparatory reading — it covers problems 1–3, forward-backward, Baum-Welch, and Viterbi with clear notation and examples.
