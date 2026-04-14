# HMM Intraday Momentum Trading with Side Information

Full reproduction of Christensen, Turner & Godsill (2020), *"Hidden Markov Models Applied To Intraday Momentum Trading With Side Information"* (arXiv:2006.08307, Cambridge Signal Processing Lab), plus a comparison against modern ML approaches.

Master's project for the course Advanced Signal Processing: Tools and Applications (ASPTA) at UPC Barcelona.

## Overview

The project implements all five model variants from the paper — Default HMM (PLR), Baum-Welch HMM, MCMC HMM, Volatility Ratio IOHMM, and Seasonality IOHMM — on 1-minute CME E-mini S&P 500 (ES) futures data using rolling-window daily retraining. As an extension, the HMM variants are benchmarked against LSTM and XGBoost baselines.

```
Layer 4: Final Comparison              <- ML vs classical SP showdown
Layer 3: IOHMM + Side Information      <- Paper §4-5: splines, bucket training
Layer 2: HMM Engine                    <- forward, backward, Baum-Welch, Viterbi, MCMC
Layer 1: Data & Utilities              <- data loading, feature computation, backtest
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## Running Tests

```bash
pytest -v
```

## Project Structure

```
src/
  data/
    loader.py              # yfinance wrapper + extract_close_series helper
    futures_loader.py      # Databento CME futures loader (1-min, RTH filter)
    features.py            # log-returns, EWMA volatility, normalization
  hmm/
    forward.py             # forward algorithm (Paper §3.2, Alg 1 lines 6-9)
    backward.py            # backward algorithm (Paper §3.2, Alg 1 lines 11-14)
    forward_backward.py    # E-step: gamma and xi posteriors
    baum_welch.py          # EM training with random restarts (Paper §3.2, Alg 1)
    baum_welch_numba.py    # Numba-accelerated Baum-Welch + inference (~50-100x)
    viterbi.py             # MAP state sequence decoding
    model_selection.py     # AIC / BIC for choosing K
    inference.py           # online predict-update loop (Paper §6, Alg 4)
    utils.py               # shared helpers: sort_states, train_best_model
    side_info.py           # volatility ratio + seasonality splines (Paper §4)
    iohmm.py               # IOHMM bucket training + inference (Paper §5-6)
    plr.py                 # piecewise linear regression initialization (Paper §3.1)
    mcmc.py                # MCMC HMM via Metropolis-Hastings (Paper §3.3)
    rolling.py             # rolling-window daily retraining (Paper §7)
    discretize.py          # return discretization to tick-size grid (Paper §2.1)
  ml/
    features.py            # feature engineering pipeline for ML models
    lstm.py                # LSTM/GRU regime predictor
    boosting.py            # XGBoost gradient boosting baseline
  strategy/
    signals.py             # convert predictions to trading signals
    backtest.py            # simulate P&L with transaction costs
  utils/
    metrics.py             # Sharpe ratio, max drawdown, annualized return
    plotting.py            # regime-colored charts, cumulative returns

experiments/
  # Phase 4-5: HMM on daily data (yfinance)
  01_data_exploration.py   # load data, plot returns, basic stats
  02_model_selection.py    # AIC/BIC vs K (reproduce paper Figure 2)
  03_baum_welch_training.py # train HMM, show convergence
  04_regime_detection.py   # Viterbi decoding overlaid on price
  05_backtest_comparison.py # HMM strategy vs buy-and-hold
  06_em_vs_mcmc.py         # Extension A: EM vs MCMC comparison
  07_multi_asset.py        # Extension B: multi-asset analysis
  08_k3_vs_k4.py           # K=3 vs K=4 model comparison
  09_rolling_backtest.py   # expanding-window robustness backtest
  10_signal_refinement.py  # no-trade zone + EMA smoothing grid search
  11_robustness_test.py    # robustness across tickers and periods
  # Phase 11: HMM vs RBPF comparison on 1-min data
  16_hmm_vs_rbpf.py        # fair head-to-head on 1-min ES
  # Phase 12-17: Full paper reproduction + ML (1-min data)
  21_autocorrelation_analysis.py  # frequency-dependent microstructure analysis
  22_rolling_hmm_1min.py          # rolling-window BW HMM at 1-min
  23_full_paper_reproduction.py   # all 5 paper models (Figure 8)
  24_ml_vs_hmm_comparison.py      # 7-way ML vs HMM vs IOHMM showdown

experiments/supplementary/         # Langevin/RBPF (2012 paper, deprioritized)
  12_kalman_filter_intro.py
  13_langevin_model.py
  14_particle_filter_baseline.py
  15_rbpf_trading.py
  17_rbpf_1min_es.py
  18_rbpf_param_search.py
  19_rbpf_portfolio.py
  20_rbpf_per_contract.py

tests/                     # pytest suite
data/databento/            # 1-min CME futures parquet files (gitignored)
docs/                      # paper PDFs, architecture docs, math mappings
figures/                   # output directory for experiment plots
reports/                   # output directory for experiment text reports
```

## Implementation Principles

- **Pure functions, not classes** -- each HMM algorithm is one function in one file
- **Log-space everywhere** -- all probability computations use log-probabilities with `logsumexp`
- **NumPy only for core math** -- Gaussian PDF implemented directly, no `scipy.stats`
- **Validated at every layer** -- each function tested against synthetic data and `hmmlearn`

## Key Algorithms

| Algorithm | File | Paper Reference |
|-----------|------|----------------|
| Forward | `src/hmm/forward.py` | §3.2, Algorithm 1 lines 6-9 |
| Backward | `src/hmm/backward.py` | §3.2, Algorithm 1 lines 11-14 |
| Forward-Backward | `src/hmm/forward_backward.py` | §3.2, Algorithm 1 line 16 |
| Baum-Welch (EM) | `src/hmm/baum_welch.py` | §3.2, Algorithm 1 lines 17-21 |
| Numba Baum-Welch | `src/hmm/baum_welch_numba.py` | Same, JIT-compiled (~50-100x) |
| Viterbi | `src/hmm/viterbi.py` | §2.2, Problem 2 |
| Online Inference | `src/hmm/inference.py` | §6, Algorithm 4 |
| PLR Initialization | `src/hmm/plr.py` | §3.1 |
| MCMC (MH) | `src/hmm/mcmc.py` | §3.3 |
| Side Info Splines | `src/hmm/side_info.py` | §4 |
| IOHMM | `src/hmm/iohmm.py` | §5-6 |
| Return Discretization | `src/hmm/discretize.py` | §2.1, Eq. 3 |
| Rolling Retraining | `src/hmm/rolling.py` | §7 |

## Running Experiments

Each experiment is standalone and saves figures to `figures/` and text reports to `reports/`:

```bash
# HMM on daily data (~minutes each)
python experiments/01_data_exploration.py
python experiments/02_model_selection.py
python experiments/03_baum_welch_training.py
python experiments/04_regime_detection.py
python experiments/05_backtest_comparison.py
python experiments/06_em_vs_mcmc.py
python experiments/07_multi_asset.py

# Full paper reproduction on 1-min data (requires Databento parquet files)
PYTHONUNBUFFERED=1 python experiments/21_autocorrelation_analysis.py
PYTHONUNBUFFERED=1 python experiments/22_rolling_hmm_1min.py
PYTHONUNBUFFERED=1 python experiments/23_full_paper_reproduction.py
PYTHONUNBUFFERED=1 python experiments/24_ml_vs_hmm_comparison.py
```

## References

- Christensen, H.L., Turner, R.E. & Godsill, S.J. (2020). *Hidden Markov Models Applied To Intraday Momentum Trading With Side Information*. arXiv:2006.08307.
- Rabiner, L.R. (1989). *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition*. Proceedings of the IEEE, 77(2), 257-286.
