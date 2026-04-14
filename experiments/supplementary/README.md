# Supplementary Experiments — Langevin/RBPF (2012 Paper)

These experiments implement the Rao-Blackwellized Particle Filter (RBPF) from:
Christensen, Godsill & Turner (2012), *"Bayesian Methods for Jump-Diffusion Langevin Models"*.

They are **supplementary** to the main project, which focuses on the 2020 HMM/IOHMM paper.
The RBPF work is complete and functional but produced negative results on 2019-2024 data
(best Sharpe -0.20 on ES, portfolio Sharpe -2.12 to -3.38). See the main project README
for context on why the project was refocused.

## Experiments

| Script | Description |
|--------|-------------|
| `12_kalman_filter_intro.py` | Kalman filter on synthetic Langevin data |
| `13_langevin_model.py` | RBPF jump detection on synthetic data (Paper Figure 3) |
| `14_particle_filter_baseline.py` | Standard bootstrap PF baseline on SPY |
| `15_rbpf_trading.py` | RBPF trading: jumps ON vs OFF |
| `17_rbpf_1min_es.py` | RBPF on 1-min ES with Table I params |
| `18_rbpf_param_search.py` | 378-point parameter grid search |
| `19_rbpf_portfolio.py` | 26-contract uniform-param portfolio |
| `20_rbpf_per_contract.py` | Per-contract calibration + selective portfolio |

## Running

All experiments are still standalone and import from `src/`:

```bash
python experiments/supplementary/12_kalman_filter_intro.py
```

Intraday experiments (17-20) require Databento parquet files in `data/databento/`.

## Source Code

The Langevin/RBPF source code remains in `src/langevin/` with all tests passing.
