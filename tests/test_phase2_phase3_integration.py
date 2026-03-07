import numpy as np

from src.hmm import baum_welch, run_inference
from src.strategy import backtest, predictions_to_signal, states_to_signal


def _train_and_infer(sample_hmm, T=240):
    A_true = np.array([[0.95, 0.05], [0.08, 0.92]], dtype=float)
    pi_true = np.array([0.5, 0.5], dtype=float)
    mu_true = np.array([-0.01, 0.01], dtype=float)
    sigma2_true = np.array([0.0004, 0.0004], dtype=float)

    _, observations = sample_hmm(T, A_true, pi_true, mu_true, sigma2_true, seed=21)

    params, _, _ = baum_welch(
        observations,
        K=2,
        max_iter=25,
        tol=1e-6,
        n_restarts=1,
        random_state=21,
    )
    predictions, state_probs = run_inference(
        observations,
        params["A"],
        params["pi"],
        params["mu"],
        params["sigma2"],
    )
    return observations, params, predictions, state_probs


def test_inference_to_predictions_to_signal_pipeline(sample_hmm):
    observations, _, predictions, _ = _train_and_infer(sample_hmm)

    signals = predictions_to_signal(predictions)

    assert signals.shape == (observations.size,)
    assert np.all(np.isin(signals, np.array([-1.0, 0.0, 1.0], dtype=float)))


def test_inference_to_states_to_signal_pipeline(sample_hmm):
    observations, params, _, state_probs = _train_and_infer(sample_hmm)

    signals = states_to_signal(state_probs, params["mu"])

    assert signals.shape == (observations.size,)
    assert np.all(signals >= -1.0)
    assert np.all(signals <= 1.0)


def test_full_pipeline_inference_to_backtest(sample_hmm):
    observations, _, predictions, _ = _train_and_infer(sample_hmm)
    signals = predictions_to_signal(predictions)

    result = backtest(observations, signals)

    assert set(result.keys()) == {"net_returns", "cumulative", "metrics"}
    assert result["net_returns"].shape == (observations.size,)
    assert result["cumulative"].shape == (observations.size,)
    assert result["cumulative"][0] >= 0.0

    metrics = result["metrics"]
    assert set(metrics.keys()) == {
        "sharpe",
        "annualized_return",
        "max_drawdown",
        "turnover",
    }
    assert all(np.isfinite(float(value)) for value in metrics.values())
