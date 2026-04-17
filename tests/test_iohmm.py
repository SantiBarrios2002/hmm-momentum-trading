"""Tests for IOHMM — bucket-based training + inference (Paper §5-6)."""

import numpy as np
import pytest

from src.hmm.iohmm import (
    _log_gauss_vec,
    run_inference_iohmm,
    train_iohmm,
)

# ── Shared fixtures ──────────────────────────────────────────────────────────

# Small synthetic data with two regimes correlated with side info
RNG = np.random.default_rng(42)
T_SYNTH = 500
# Side info: linearly increasing from 0 to 1
SIDE_INFO_SYNTH = np.linspace(0.2, 0.8, T_SYNTH)
# Returns: bearish when side_info < 0.5, bullish when side_info >= 0.5
OBS_SYNTH = np.where(
    SIDE_INFO_SYNTH < 0.5,
    RNG.normal(-0.002, 0.005, T_SYNTH),
    RNG.normal(+0.002, 0.005, T_SYNTH),
)

# BW settings for fast tests
FAST_BW = dict(n_restarts=2, max_iter=50, tol=1e-4, min_variance=1e-8)


# ── _log_gauss_vec ───────────────────────────────────────────────────────────


class TestLogGaussVec:
    def test_known_values(self):
        """Standard normal at x=0 should give log(1/√(2π))."""
        result = _log_gauss_vec(0.0, np.array([0.0]), np.array([1.0]))
        expected = -0.5 * np.log(2 * np.pi)
        np.testing.assert_allclose(result, [expected], rtol=1e-12)

    def test_vector_output(self):
        mu = np.array([-1.0, 0.0, 1.0])
        sigma2 = np.array([0.5, 1.0, 2.0])
        result = _log_gauss_vec(0.5, mu, sigma2)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))
        # All log-densities should be negative (density < 1 for these params)
        assert np.all(result < 0)


# ── train_iohmm ─────────────────────────────────────────────────────────────


class TestTrainIohmm:
    def test_output_keys(self):
        """All expected keys must be present in the output dict."""
        result = train_iohmm(
            OBS_SYNTH, SIDE_INFO_SYNTH, K=2, n_knots=4, **FAST_BW,
        )
        expected_keys = {"spline", "boundaries", "bucket_params", "bucket_ll", "K", "R"}
        assert set(result.keys()) == expected_keys

    def test_bucket_count_consistent(self):
        """R must equal len(boundaries) + 1."""
        result = train_iohmm(
            OBS_SYNTH, SIDE_INFO_SYNTH, K=2, n_knots=4, **FAST_BW,
        )
        assert result["R"] == len(result["boundaries"]) + 1
        assert len(result["bucket_params"]) == result["R"]
        assert len(result["bucket_ll"]) == result["R"]

    def test_bucket_params_valid_hmms(self):
        """Each bucket's parameters must form a valid HMM."""
        result = train_iohmm(
            OBS_SYNTH, SIDE_INFO_SYNTH, K=2, n_knots=4, **FAST_BW,
        )
        K = result["K"]
        for r, params in enumerate(result["bucket_params"]):
            # Check shapes
            assert params["A"].shape == (K, K), f"bucket {r}: A wrong shape"
            assert params["pi"].shape == (K,), f"bucket {r}: pi wrong shape"
            assert params["mu"].shape == (K,), f"bucket {r}: mu wrong shape"
            assert params["sigma2"].shape == (K,), f"bucket {r}: sigma2 wrong shape"

            # Transition matrix: rows sum to 1
            row_sums = params["A"].sum(axis=1)
            np.testing.assert_allclose(
                row_sums, np.ones(K),
                atol=K * np.finfo(float).eps,
                err_msg=f"bucket {r}: A rows don't sum to 1",
            )

            # Initial distribution sums to 1
            assert params["pi"].sum() == pytest.approx(1.0, abs=1e-10)

            # Positive variances
            assert np.all(params["sigma2"] > 0), f"bucket {r}: non-positive sigma2"

            # States sorted by ascending mu
            assert np.all(np.diff(params["mu"]) >= 0), f"bucket {r}: mu not sorted"

    def test_K_preserved(self):
        result = train_iohmm(
            OBS_SYNTH, SIDE_INFO_SYNTH, K=3, n_knots=4, **FAST_BW,
        )
        assert result["K"] == 3
        for params in result["bucket_params"]:
            assert params["mu"].shape == (3,)

    def test_fallback_for_sparse_bucket(self):
        """Buckets with too few observations should use global fallback."""
        # Use very high min_obs_per_bucket so ALL buckets fall back
        result = train_iohmm(
            OBS_SYNTH, SIDE_INFO_SYNTH, K=2, n_knots=4,
            min_obs_per_bucket=T_SYNTH + 1,  # impossible threshold
            **FAST_BW,
        )
        # All buckets should have identical parameters (global fallback)
        for r in range(1, result["R"]):
            np.testing.assert_array_equal(
                result["bucket_params"][r]["mu"],
                result["bucket_params"][0]["mu"],
            )

    def test_bucket_ll_finite(self):
        result = train_iohmm(
            OBS_SYNTH, SIDE_INFO_SYNTH, K=2, n_knots=4, **FAST_BW,
        )
        for ll in result["bucket_ll"]:
            assert np.isfinite(ll)

    def test_input_validation_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            train_iohmm(np.array([1.0, 2.0]), np.array([1.0]), K=2)

    def test_input_validation_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            train_iohmm(np.array([]), np.array([]), K=2)

    def test_input_validation_K(self):
        with pytest.raises(ValueError, match="K must be"):
            train_iohmm(OBS_SYNTH, SIDE_INFO_SYNTH, K=0)


# ── run_inference_iohmm ─────────────────────────────────────────────────────


class TestRunInferenceIohmm:
    @pytest.fixture()
    def trained_model(self):
        return train_iohmm(
            OBS_SYNTH, SIDE_INFO_SYNTH, K=2, n_knots=4, **FAST_BW,
        )

    def test_output_shapes(self, trained_model):
        preds, sprobs = run_inference_iohmm(
            OBS_SYNTH, SIDE_INFO_SYNTH, trained_model,
        )
        assert preds.shape == (T_SYNTH,)
        assert sprobs.shape == (T_SYNTH, 2)

    def test_predictions_finite(self, trained_model):
        preds, _ = run_inference_iohmm(
            OBS_SYNTH, SIDE_INFO_SYNTH, trained_model,
        )
        assert np.all(np.isfinite(preds))

    def test_state_probs_sum_to_one(self, trained_model):
        _, sprobs = run_inference_iohmm(
            OBS_SYNTH, SIDE_INFO_SYNTH, trained_model,
        )
        row_sums = sprobs.sum(axis=1)
        np.testing.assert_allclose(
            row_sums, np.ones(T_SYNTH),
            atol=1e-10,
        )

    def test_state_probs_nonnegative(self, trained_model):
        _, sprobs = run_inference_iohmm(
            OBS_SYNTH, SIDE_INFO_SYNTH, trained_model,
        )
        assert np.all(sprobs >= 0.0)

    def test_works_on_new_data(self, trained_model):
        """Inference should work on data not used for training."""
        rng = np.random.default_rng(99)
        T_new = 100
        new_obs = rng.normal(0, 0.005, T_new)
        new_side = np.linspace(0.3, 0.7, T_new)
        preds, sprobs = run_inference_iohmm(new_obs, new_side, trained_model)
        assert preds.shape == (T_new,)
        assert sprobs.shape == (T_new, 2)
        assert np.all(np.isfinite(preds))

    def test_input_validation_length_mismatch(self, trained_model):
        with pytest.raises(ValueError, match="same length"):
            run_inference_iohmm(
                np.array([1.0, 2.0]), np.array([1.0]), trained_model,
            )

    def test_input_validation_empty(self, trained_model):
        with pytest.raises(ValueError, match="non-empty"):
            run_inference_iohmm(
                np.array([]), np.array([]), trained_model,
            )


# ── Integration ──────────────────────────────────────────────────────────────


class TestIntegration:
    def test_train_then_infer_end_to_end(self):
        """Full pipeline: side info → spline → buckets → train → infer."""
        rng = np.random.default_rng(123)
        T = 400

        # Two-regime data: low side info → negative returns, high → positive
        side = np.linspace(0.2, 0.8, T)
        obs = np.where(
            side < 0.5,
            rng.normal(-0.003, 0.004, T),
            rng.normal(+0.003, 0.004, T),
        )

        model = train_iohmm(obs, side, K=2, n_knots=4, **FAST_BW)
        preds, sprobs = run_inference_iohmm(obs, side, model)

        assert preds.shape == (T,)
        assert sprobs.shape == (T, 2)
        assert np.all(np.isfinite(preds))
        assert np.all(np.isfinite(sprobs))
        # State probs sum to 1
        np.testing.assert_allclose(sprobs.sum(axis=1), np.ones(T), atol=1e-10)

    def test_single_bucket_degeneracy(self):
        """If spline has no roots → 1 bucket → IOHMM degenerates to HMM."""
        rng = np.random.default_rng(77)
        T = 300
        # Constant returns → flat spline → likely 1 bucket
        obs = rng.normal(0, 0.005, T)
        side = np.linspace(0.3, 0.7, T)

        model = train_iohmm(obs, side, K=2, n_knots=4, **FAST_BW)

        # Even with 1 bucket, everything should work
        preds, sprobs = run_inference_iohmm(obs, side, model)
        assert preds.shape == (T,)
        assert sprobs.shape == (T, 2)
        assert np.all(np.isfinite(preds))

    def test_side_info_outside_training_range(self):
        """Side info values outside training range should be handled."""
        rng = np.random.default_rng(55)
        T = 300
        obs = rng.normal(0, 0.005, T)
        side_train = np.linspace(0.3, 0.7, T)

        model = train_iohmm(obs, side_train, K=2, n_knots=4, **FAST_BW)

        # Inference with side info outside training range
        side_test = np.linspace(0.0, 1.0, T)  # wider range
        preds, sprobs = run_inference_iohmm(obs, side_test, model)
        assert np.all(np.isfinite(preds))
        assert np.all(np.isfinite(sprobs))
