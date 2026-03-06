# Mathematical Concepts: ASPTA ↔ Paper Mapping

## What to read for — and where you'll find it in the paper

This document maps every ASPTA concept the paper uses, gives you the equation/section to look at, and tells you what to study beforehand so you can read the paper fluently.

---

## 1. Maximum Likelihood Estimation (Module 1.2)

### In the paper
The core problem is: given observations **x** = (x₁, ..., xₙ), find model parameters Λ = {π, A, μ, Σ} that maximize P(**x** | Λ). This is textbook MLE applied to a latent-variable model.

**Paper Section 2.2, Problem 3:** *"How do we adjust the HMM parameters Λ to maximize P(x|Λ)?"*

### What you need to know beforehand
- **MLE fundamentals:** Given i.i.d. observations, the MLE is θ̂ = argmax_θ Σ log p(xₜ | θ). You should be comfortable deriving MLEs for Gaussian parameters (mean and variance).
- **Log-likelihood:** Why we work with log P(x|Λ) instead of P(x|Λ) — numerical stability, turns products into sums.
- **The incomplete-data problem:** When there are latent variables z (the hidden states), you can't directly maximize P(x|Λ) because it requires marginalizing over all possible state sequences: P(x|Λ) = Σ_z P(x,z|Λ). This sum has S^n terms — intractable. This motivates the EM algorithm.

### ASPTA reference
- Kay Vol. I, Chapter 7 (Maximum Likelihood Estimation)
- ASPTA Module 1.2 lectures on MLE

---

## 2. The EM Algorithm / Baum-Welch (Module 1.2)

### In the paper
**Paper Eq. (2.1):** The E-step computes the expected complete-data log-likelihood:

> Q(Λ, Λ̄) = E[ log P(x, z | Λ) | x, Λ̄ ]

The M-step maximizes Q with respect to Λ. The algorithm iterates E→M→E→M until convergence. The key property is P(x|Λ) ≥ P(x|Λ̄) at each step — the likelihood never decreases.

**Baum-Welch is simply EM applied to HMMs.** The E-step uses the forward-backward algorithm (see below) to compute two quantities:
- **γₜ(j)** = P(zₜ = j | x, Λ̄) — the posterior probability of being in state j at time t
- **ξₜ(i,j)** = P(zₜ₋₁ = i, zₜ = j | x, Λ̄) — the posterior probability of transitioning from state i to j at time t

The M-step updates are then:
- **π̂ⱼ** = γ₁(j)   (initial state probability = posterior at t=1)
- **Âᵢⱼ** = Σₜ ξₜ(i,j) / Σₜ γₜ(i)   (transition prob = expected transitions from i to j / expected time in state i)
- **μ̂ⱼ** = Σₜ γₜ(j) xₜ / Σₜ γₜ(j)   (weighted mean of observations)
- **Σ̂ⱼ** = Σₜ γₜ(j) (xₜ - μ̂ⱼ)(xₜ - μ̂ⱼ)ᵀ / Σₜ γₜ(j)   (weighted covariance)

These are just weighted versions of the standard Gaussian MLE formulas, where the weights are the posterior state probabilities.

### What you need to know beforehand
- **EM general framework:** E-step computes Q(θ, θ_old), M-step maximizes Q. Understand why it converges (Jensen's inequality argument).
- **Complete vs. incomplete data:** If we knew z, MLE would be trivial (just count transitions, compute means per state). EM "fills in" the missing z using posterior expectations.
- **Convergence properties:** EM converges to a local maximum, not necessarily global. Paper handles this by running multiple random initializations and keeping the best.

### ASPTA reference
- ASPTA Module 1.2 (EM algorithm lecture)
- Kay Vol. I, Section 7.9 or Dempster, Laird & Rubin (1977)

---

## 3. Bayesian / MAP Estimation (Module 1.3)

### In the paper
**Paper Eq. (2.2):** The paper extends pure MLE to Maximum a Posteriori (MAP) estimation by adding a prior term:

> maximize Q(Λ, Λ̄) + log G(Λ)

where G(Λ) is the prior distribution on model parameters. This is Bayesian estimation where instead of maximizing the likelihood P(x|Λ), you maximize the posterior P(Λ|x) ∝ P(x|Λ) · G(Λ).

The paper uses conjugate priors (Dirichlet for transition probabilities, Normal-Inverse-Wishart for Gaussian parameters), which give closed-form MAP updates that look like the MLE updates but with "pseudo-counts" added.

### What you need to know beforehand
- **Bayes' theorem applied to estimation:** p(θ|x) ∝ p(x|θ) · p(θ). The posterior is proportional to likelihood × prior.
- **MAP estimator:** θ̂_MAP = argmax_θ p(θ|x) = argmax_θ [log p(x|θ) + log p(θ)]
- **How MAP differs from MLE:** MLE can overfit with limited data. MAP adds regularization through the prior. With infinite data, MAP → MLE.
- **Conjugate priors:** The Dirichlet is conjugate to the Categorical (used for π and rows of A). Normal-Inverse-Wishart is conjugate to the multivariate Gaussian (used for μⱼ, Σⱼ).

### ASPTA reference
- ASPTA Module 1.3 (Bayesian estimation)
- Kay Vol. I, Chapter 10–12

---

## 4. Forward-Backward Algorithm (Modules 1.2 / 2.2)

### In the paper
**Paper Section 2.2:** The forward-backward algorithm is how you compute P(x|Λ) efficiently and obtain the state posteriors γₜ(j) needed for the E-step.

**Forward pass** computes αₜ(j) = P(x₁,...,xₜ, zₜ = j | Λ):
- α₁(j) = πⱼ · p(x₁ | zₜ = j)
- αₜ(j) = [ Σᵢ αₜ₋₁(i) · Aᵢⱼ ] · p(xₜ | zₜ = j)

**Backward pass** computes βₜ(j) = P(xₜ₊₁,...,xₙ | zₜ = j, Λ):
- βₙ(j) = 1
- βₜ(j) = Σⱼ' Aⱼⱼ' · p(xₜ₊₁ | zₜ₊₁ = j') · βₜ₊₁(j')

**Posteriors:**
- γₜ(j) = αₜ(j) · βₜ(j) / P(x|Λ)
- ξₜ(i,j) = αₜ₋₁(i) · Aᵢⱼ · p(xₜ|zₜ=j) · βₜ(j) / P(x|Λ)

This is a form of recursive Bayesian filtering — structurally similar to the Kalman filter (Module 2.2) but for discrete states.

### What you need to know beforehand
- **Recursive structure:** Each step combines a prediction (propagating through transition matrix A) and an update (incorporating the new observation via the emission probability). This predict-update cycle is the same concept as in Kalman filtering.
- **Log-space implementation:** In practice, you work in log-space to avoid numerical underflow (products of many small probabilities → zero in floating point).
- **Relationship to Kalman filter:** The forward algorithm is the discrete-state analog of the Kalman filter's prediction + update. If you replace discrete states with continuous Gaussian state, and the transition matrix with a linear state model, you get the Kalman filter.

### ASPTA reference
- ASPTA Module 2.1–2.2 (Kalman filter has the same predict/update structure)
- Rabiner (1989) tutorial (the classic HMM reference)

---

## 5. Model Selection / Information Criteria (Module 1.1)

### In the paper
**Paper Section 2.4:** To choose the number of hidden states S, the paper uses penalized likelihood criteria:

> AIC = −2 log(L) + 2p
> BIC = −2 log(L) + p log(n)

where p = S(S + cd) is the number of free parameters, and c = d + d(d+1)/2 for a d-dimensional Gaussian.

This is directly connected to the bias-variance tradeoff and the Cramér-Rao bound. More parameters (larger S) → better fit to training data (lower −2 log L) but risk of overfitting (higher penalty term).

### What you need to know beforehand
- **Cramér-Rao Lower Bound (CRLB):** The minimum variance any unbiased estimator can achieve. With more parameters, each individual parameter is estimated with more variance given fixed data.
- **Sufficient statistics:** The EWMM features the paper computes (mean and variance) are related to sufficient statistics of the Gaussian distribution.
- **Model complexity vs. fit:** AIC/BIC formalize the tradeoff. BIC penalizes complexity more heavily and is consistent (selects the true model as n → ∞).

### ASPTA reference
- ASPTA Module 1.1 (Cramér-Rao bound, sufficient statistics)
- Kay Vol. I, Chapters 2–3

---

## 6. Viterbi Algorithm (Detection theory parallel — Module 3)

### In the paper
**Paper Section 2.2, Problem 2:** Finding the optimal hidden state sequence z* = argmax_z P(z|x, Λ). Solved by the Viterbi algorithm (dynamic programming).

This is structurally a **detection problem** — at each time step, you're deciding which of S states generated the observation. It connects to ASPTA Module 3 (Detection Theory): given an observation, decide between S hypotheses, each with a known probability model.

### What you need to know beforehand
- **MAP sequence detection:** Viterbi finds the most probable entire sequence, not just the most probable state at each time. This is like jointly detecting all symbols in a communications sequence.
- **Dynamic programming:** Viterbi avoids brute-force enumeration (S^n paths) by exploiting the Markov property — the optimal path to state j at time t only depends on the optimal paths to all states at time t-1.

### ASPTA reference
- ASPTA Module 3.1–3.2 (Detection of deterministic/random signals)

---

## 7. Exponential Smoothing / Recursive Estimation (Module 2.1)

### In the paper
**Paper Section 4.1:** The features are computed using Exponential Weighted Moving Moments:

> EWMM_t = λ · M_t + (1 − λ) · EWMM_{t-1}

where λ = 2/(s+1). This is a first-order recursive filter — the simplest form of adaptive estimation, directly related to Recursive Least Squares (Module 2.1).

### What you need to know beforehand
- **Recursive estimation:** The idea that you can update an estimate by combining the old estimate with the new observation, without reprocessing all past data.
- **Forgetting factor:** λ controls how much weight recent vs. old data gets. Small λ → more smoothing, long memory. Large λ → more reactive, short memory.
- **Connection to RLS:** RLS is the generalization of this to multivariate regression problems.

### ASPTA reference
- ASPTA Module 2.1 (Recursive Least Squares)

---

## Summary Table

| Paper concept | Paper section | ASPTA Module | What to study |
|---|---|---|---|
| MLE for parametric models | §2.2, Problem 3 | 1.2 | MLE derivation for Gaussian, log-likelihood |
| EM algorithm (Baum-Welch) | §2.2, Eq. (2.1) | 1.2 | E-step/M-step, convergence, Q-function |
| MAP estimation with priors | §2.2, Eq. (2.2) | 1.3 | Bayes' theorem, conjugate priors, MAP vs MLE |
| Forward-backward algorithm | §2.2 | 1.2 / 2.2 | Recursive Bayesian filtering, predict-update |
| Model selection (AIC/BIC) | §2.4 | 1.1 | CRLB, sufficient statistics, bias-variance |
| Viterbi (MAP sequence detection) | §2.2, Problem 2 | 3.1 | Hypothesis testing, dynamic programming |
| Exponential smoothing | §4.1 | 2.1 | Recursive estimation, forgetting factor |
| State prediction via A^h | §2.3, Eq. (2.3) | 2.2 | Markov chain propagation (Kalman prediction analog) |

---

## Reading Order

Before diving into the paper, review these in order (you probably know most of it, but solidify the gaps):

1. **MLE for Gaussians** — Derive μ̂_ML and Σ̂_ML from scratch. 30 minutes.
2. **EM algorithm general** — Read ASPTA Module 1.2 slides on EM. Focus on the Q-function and why the likelihood is non-decreasing. 1 hour.
3. **Bayes' theorem for estimation** — Review MAP vs MLE. Understand conjugate priors conceptually (you don't need to derive them). 30 minutes.
4. **Rabiner (1989) Sections III-IV** — The classic HMM tutorial. Covers forward-backward and Baum-Welch with clear notation. 2 hours.
5. **Read the Werge paper** — Now you'll understand everything. Focus on Sections 2 and 4. 2 hours.

**Total prep time: ~6 hours of focused reading.**
