"""Unit tests for XGBoost champion, LightGBM challenger, and SHAP explainer."""
from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
import pytest

from aegis_ml.models.fraud_models import (
    pos_weight,
    shap_explainer,
    top_reason_codes,
    train_lightgbm,
    train_xgboost,
)


# --- 1. pos_weight tests ---


def test_pos_weight_balanced():
    """pos_weight on balanced labels (50/50) returns exactly 1.0."""
    y = np.array([0, 1, 0, 1])
    assert pos_weight(y) == 1.0


def test_pos_weight_imbalanced():
    """pos_weight on imbalanced labels calculates neg / pos correctly."""
    # 95 negatives, 5 positives -> 95 / 5 = 19.0
    y = np.r_[np.zeros(95, dtype=int), np.ones(5, dtype=int)]
    assert pos_weight(y) == pytest.approx(19.0)

    # 97 negatives, 3 positives -> 97 / 3 = 32.333...
    y2 = pd.Series([0] * 97 + [1] * 3)
    assert pos_weight(y2) == pytest.approx(97.0 / 3.0)


def test_pos_weight_zero_positives():
    """pos_weight with zero positives clamps positive count to 1 to avoid division by zero."""
    y = np.zeros(20, dtype=int)
    # (20 - 1) / 1 = 19.0
    assert pos_weight(y) == 19.0

    y_single = np.array([0])
    # (1 - 1) / 1 = 0.0
    assert pos_weight(y_single) == 0.0


def test_pos_weight_all_positives():
    """pos_weight with all positives returns 0.0."""
    y = np.ones(10, dtype=int)
    assert pos_weight(y) == 0.0


def test_pos_weight_accepts_booleans():
    """pos_weight accepts boolean inputs."""
    y = [False, True, False, False]
    assert pos_weight(y) == 3.0


def test_pos_weight_rejects_empty():
    """pos_weight rejects empty input."""
    with pytest.raises(ValueError, match="cannot be empty"):
        pos_weight([])
    with pytest.raises(ValueError, match="cannot be empty"):
        pos_weight(np.array([]))


def test_pos_weight_rejects_multidimensional():
    """pos_weight rejects multidimensional arrays."""
    with pytest.raises(ValueError, match="must be 1D"):
        pos_weight(np.array([[0, 1], [1, 0]]))


def test_pos_weight_rejects_invalid_values():
    """pos_weight rejects non-binary numbers, non-finite values, and strings."""
    # Non-binary continuous floats
    with pytest.raises(ValueError, match="binary labels"):
        pos_weight([0.0, 0.5, 1.0])

    # Out of range integers
    with pytest.raises(ValueError, match="binary labels"):
        pos_weight([0, 1, 2])
    with pytest.raises(ValueError, match="binary labels"):
        pos_weight([0, 1, -1])

    # Non-finite values
    with pytest.raises(ValueError, match="binary labels"):
        pos_weight([0, 1, np.nan])
    with pytest.raises(ValueError, match="binary labels"):
        pos_weight([0, 1, np.inf])

    # String arrays
    with pytest.raises(ValueError, match="binary labels"):
        pos_weight(["0", "1"])


def test_pos_weight_does_not_mutate_input():
    """pos_weight does not mutate caller data."""
    orig = np.array([0, 1, 0, 1])
    orig_copy = orig.copy()
    _ = pos_weight(orig)
    np.testing.assert_array_equal(orig, orig_copy)


# --- 2. Model training fixtures ---


@pytest.fixture
def synthetic_data():
    """Deterministic synthetic tabular dataset for model tests."""
    rng = np.random.default_rng(42)
    n_train = 120
    n_val = 40
    n_features = 6

    # Informative linear combination with noise
    w = np.array([1.5, -1.0, 0.8, -0.5, 0.0, 0.2])

    X_tr = rng.normal(0, 1, (n_train, n_features))
    logits_tr = X_tr @ w
    p_tr = 1 / (1 + np.exp(-logits_tr))
    y_tr = (p_tr > 0.65).astype(int)
    # Ensure both classes exist
    if y_tr.sum() == 0:
        y_tr[0] = 1

    X_v = rng.normal(0, 1, (n_val, n_features))
    logits_v = X_v @ w
    p_v = 1 / (1 + np.exp(-logits_v))
    y_v = (p_v > 0.65).astype(int)
    if y_v.sum() == 0:
        y_v[0] = 1

    cols = [f"feat_{i}" for i in range(n_features)]
    df_train = pd.DataFrame(X_tr, columns=cols)
    df_val = pd.DataFrame(X_v, columns=cols)

    return df_train, y_tr, df_val, y_v, cols


# --- 3. XGBoost Champion tests ---


def test_train_xgboost_execution_and_outputs(synthetic_data):
    """train_xgboost fits model and returns probabilities on validation set."""
    X_train, y_train, X_val, y_val, cols = synthetic_data
    fast_params = {
        "n_estimators": 10,
        "max_depth": 3,
        "early_stopping_rounds": 5,
        "seed": 42,
        "n_jobs": 1,
    }

    model, probs = train_xgboost(X_train, y_train, X_val, y_val, fast_params)

    # Check return shapes and types
    assert hasattr(model, "predict_proba")
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (len(X_val),)

    # Probabilities must be finite and bounded within [0, 1]
    assert np.all(np.isfinite(probs))
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)

    # Model attributes match expected configuration
    assert model.max_depth == 3
    assert model.n_estimators == 10
    assert model.learning_rate == 0.05
    assert model.subsample == 0.8
    assert model.colsample_bytree == 0.8
    assert model.min_child_weight == 5
    assert model.reg_lambda == 1.0
    assert model.tree_method == "hist"
    assert model.eval_metric == "aucpr"
    assert model.scale_pos_weight == pytest.approx(pos_weight(y_train))


def test_train_xgboost_default_parameters():
    """train_xgboost applies default hyperparameters from the implementation guide."""
    import xgboost as xgb

    X_train = np.random.randn(20, 2)
    y_train = np.array([0] * 18 + [1] * 2)
    X_val = np.random.randn(10, 2)
    y_val = np.array([0] * 9 + [1] * 1)

    # Calling with empty dict / None
    model, probs = train_xgboost(X_train, y_train, X_val, y_val, params={})
    assert model.n_estimators == 800
    assert model.max_depth == 6
    assert model.learning_rate == 0.05
    assert model.subsample == 0.8
    assert model.colsample_bytree == 0.8
    assert model.min_child_weight == 5
    assert model.reg_lambda == 1.0
    assert model.tree_method == "hist"
    assert model.eval_metric == "aucpr"
    assert model.early_stopping_rounds == 60
    assert model.random_state == 42
    assert model.n_jobs == 4
    assert model.scale_pos_weight == pytest.approx(pos_weight(y_train))


def test_train_xgboost_mismatched_lengths():
    """train_xgboost validates that feature and label dimensions match."""
    X = np.random.randn(20, 2)
    y_short = np.array([0, 1])
    y_ok = np.array([0] * 10 + [1] * 10)

    with pytest.raises(ValueError, match="X_train length"):
        train_xgboost(X, y_short, X, y_ok)

    with pytest.raises(ValueError, match="X_val length"):
        train_xgboost(X, y_ok, X, y_short)


def test_train_xgboost_deterministic(synthetic_data):
    """train_xgboost produces deterministic predictions for identical seeds."""
    X_train, y_train, X_val, y_val, _ = synthetic_data
    params = {"n_estimators": 10, "max_depth": 3, "seed": 42, "n_jobs": 1}

    _, probs1 = train_xgboost(X_train, y_train, X_val, y_val, params)
    _, probs2 = train_xgboost(X_train, y_train, X_val, y_val, params)

    np.testing.assert_allclose(probs1, probs2, rtol=1e-5)


# --- 4. LightGBM Challenger tests ---


def test_train_lightgbm_execution_and_outputs(synthetic_data):
    """train_lightgbm fits model and returns probabilities on validation set."""
    X_train, y_train, X_val, y_val, cols = synthetic_data
    fast_params = {
        "n_estimators": 10,
        "num_leaves": 15,
        "min_child_samples": 5,
        "early_stopping_rounds": 5,
        "seed": 42,
        "n_jobs": 1,
    }

    model, probs = train_lightgbm(X_train, y_train, X_val, y_val, fast_params)

    # Check return shapes and types
    assert hasattr(model, "predict_proba")
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (len(X_val),)

    # Probabilities must be finite and bounded within [0, 1]
    assert np.all(np.isfinite(probs))
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)

    # Model attributes match expected configuration
    assert model.objective == "binary"
    assert model.n_estimators == 10
    assert model.learning_rate == 0.05
    assert model.num_leaves == 15
    assert model.min_child_samples == 5
    assert model.subsample == 0.8
    assert model.subsample_freq == 1
    assert model.colsample_bytree == 0.8
    assert model.scale_pos_weight == pytest.approx(pos_weight(y_train))


def test_train_lightgbm_default_parameters():
    """train_lightgbm applies default hyperparameters from the implementation guide."""
    X_train = np.random.randn(30, 2)
    y_train = np.array([0] * 25 + [1] * 5)
    X_val = np.random.randn(10, 2)
    y_val = np.array([0] * 9 + [1] * 1)

    model, probs = train_lightgbm(X_train, y_train, X_val, y_val, params=None)
    assert model.objective == "binary"
    assert model.n_estimators == 800
    assert model.learning_rate == 0.05
    assert model.num_leaves == 63
    assert model.min_child_samples == 50
    assert model.subsample == 0.8
    assert model.subsample_freq == 1
    assert model.colsample_bytree == 0.8
    assert model.random_state == 42
    assert model.n_jobs == 4
    assert model.verbose == -1
    assert model.scale_pos_weight == pytest.approx(pos_weight(y_train))


def test_train_lightgbm_mismatched_lengths():
    """train_lightgbm validates that feature and label dimensions match."""
    X = np.random.randn(20, 2)
    y_short = np.array([0, 1])
    y_ok = np.array([0] * 10 + [1] * 10)

    with pytest.raises(ValueError, match="X_train length"):
        train_lightgbm(X, y_short, X, y_ok)

    with pytest.raises(ValueError, match="X_val length"):
        train_lightgbm(X, y_ok, X, y_short)


def test_train_lightgbm_deterministic(synthetic_data):
    """train_lightgbm produces deterministic predictions for identical seeds."""
    X_train, y_train, X_val, y_val, _ = synthetic_data
    params = {"n_estimators": 10, "num_leaves": 15, "min_child_samples": 5, "seed": 42, "n_jobs": 1}

    _, probs1 = train_lightgbm(X_train, y_train, X_val, y_val, params)
    _, probs2 = train_lightgbm(X_train, y_train, X_val, y_val, params)

    np.testing.assert_allclose(probs1, probs2, rtol=1e-5)


# --- 5. Inputs Not Mutated ---


def test_training_does_not_mutate_caller_inputs(synthetic_data):
    """Neither XGBoost nor LightGBM training modifies caller DataFrames or label arrays."""
    X_train, y_train, X_val, y_val, _ = synthetic_data
    X_tr_copy = X_train.copy()
    y_tr_copy = y_train.copy()
    X_v_copy = X_val.copy()
    y_v_copy = y_val.copy()

    params = {"n_estimators": 5, "max_depth": 2, "early_stopping_rounds": 2, "n_jobs": 1}
    _ = train_xgboost(X_train, y_train, X_val, y_val, params)
    pd.testing.assert_frame_equal(X_train, X_tr_copy)
    np.testing.assert_array_equal(y_train, y_tr_copy)
    pd.testing.assert_frame_equal(X_val, X_v_copy)
    np.testing.assert_array_equal(y_val, y_v_copy)

    _ = train_lightgbm(X_train, y_train, X_val, y_val, {"n_estimators": 5, "num_leaves": 8, "n_jobs": 1})
    pd.testing.assert_frame_equal(X_train, X_tr_copy)
    np.testing.assert_array_equal(y_train, y_tr_copy)


# --- 6. SHAP Explainer tests ---


def test_shap_explainer_construction_and_pickle(synthetic_data):
    """shap_explainer constructs TreeExplainer suitable for pickle serialization."""
    import shap

    X_train, y_train, X_val, y_val, _ = synthetic_data
    params = {"n_estimators": 10, "max_depth": 3, "early_stopping_rounds": 5, "seed": 42, "n_jobs": 1}
    model, _ = train_xgboost(X_train, y_train, X_val, y_val, params)

    explainer = shap_explainer(model)
    assert isinstance(explainer, shap.TreeExplainer)
    assert explainer.feature_perturbation == "tree_path_dependent"
    assert explainer.data is None

    # Test pickle roundtrip
    dumped = pickle.dumps(explainer)
    loaded = pickle.loads(dumped)
    assert isinstance(loaded, shap.TreeExplainer)

    # Both original and deserialized compute identical SHAP values
    sv_orig = explainer.shap_values(X_val.iloc[0:1])
    sv_loaded = loaded.shap_values(X_val.iloc[0:1])
    np.testing.assert_allclose(sv_orig, sv_loaded)


def test_shap_explainer_with_lightgbm(synthetic_data):
    """shap_explainer constructs TreeExplainer for LightGBM model."""
    import shap

    X_train, y_train, X_val, y_val, _ = synthetic_data
    params = {"n_estimators": 10, "num_leaves": 15, "min_child_samples": 5, "seed": 42, "n_jobs": 1}
    model, _ = train_lightgbm(X_train, y_train, X_val, y_val, params)

    explainer = shap_explainer(model, X_background=X_train)
    assert isinstance(explainer, shap.TreeExplainer)
    sv = explainer.shap_values(X_val.iloc[0:1])
    assert np.asarray(sv).size > 0


# --- 7. Top Reason Codes tests ---


def test_top_reason_codes_positive_only_and_ordering():
    """top_reason_codes returns only positive contributions in descending order."""
    class MockExplainer:
        def shap_values(self, X):
            # 5 features: contributions include positive, negative, and zero
            return np.array([[0.15, -0.42, 0.85, 0.0, 0.35]])

    explainer = MockExplainer()
    feature_names = ["feat_A", "feat_B", "feat_C", "feat_D", "feat_E"]

    # Top-3 should return feat_C (0.85), feat_E (0.35), feat_A (0.15)
    reasons = top_reason_codes(explainer, np.zeros((1, 5)), feature_names, k=3)
    assert len(reasons) == 3
    assert reasons[0] == {"feature": "feat_C", "contribution": pytest.approx(0.85)}
    assert reasons[1] == {"feature": "feat_E", "contribution": pytest.approx(0.35)}
    assert reasons[2] == {"feature": "feat_A", "contribution": pytest.approx(0.15)}

    # Requesting k=5 returns only the 3 positive features (negative and 0 are excluded)
    reasons_all = top_reason_codes(explainer, np.zeros((1, 5)), feature_names, k=5)
    assert len(reasons_all) == 3


def test_top_reason_codes_no_positive_contributions():
    """top_reason_codes returns empty list when all features have <= 0 contribution."""
    class MockExplainer:
        def shap_values(self, X):
            return np.array([[-0.1, -0.5, 0.0, -0.3]])

    explainer = MockExplainer()
    feature_names = ["f1", "f2", "f3", "f4"]
    reasons = top_reason_codes(explainer, np.zeros((1, 4)), feature_names, k=5)
    assert reasons == []


def test_top_reason_codes_k_zero_or_negative():
    """top_reason_codes returns empty list when k <= 0."""
    class MockExplainer:
        def shap_values(self, X):
            return np.array([[0.5, 0.3]])

    explainer = MockExplainer()
    assert top_reason_codes(explainer, np.zeros((1, 2)), ["a", "b"], k=0) == []
    assert top_reason_codes(explainer, np.zeros((1, 2)), ["a", "b"], k=-2) == []


def test_top_reason_codes_feature_length_mismatch():
    """top_reason_codes raises ValueError if feature_names length does not match SHAP values."""
    class MockExplainer:
        def shap_values(self, X):
            return np.array([[0.5, 0.3, 0.1]])

    explainer = MockExplainer()
    with pytest.raises(ValueError, match="Length of feature_names"):
        top_reason_codes(explainer, np.zeros((1, 3)), ["f1", "f2"], k=5)


def test_top_reason_codes_supports_multiple_input_types(synthetic_data):
    """top_reason_codes transparently handles 1D Series, 2D DataFrame, 1D array, and 2D array."""
    X_train, y_train, X_val, y_val, cols = synthetic_data
    params = {"n_estimators": 10, "max_depth": 3, "early_stopping_rounds": 5, "seed": 42, "n_jobs": 1}
    model, _ = train_xgboost(X_train, y_train, X_val, y_val, params)
    explainer = shap_explainer(model)

    # 1. 2D DataFrame row
    res_df = top_reason_codes(explainer, X_val.iloc[0:1], cols, k=3)
    # 2. 1D Series row
    res_series = top_reason_codes(explainer, X_val.iloc[0], cols, k=3)
    # 3. 2D NumPy row
    res_np2d = top_reason_codes(explainer, X_val.iloc[0:1].to_numpy(), cols, k=3)
    # 4. 1D NumPy row
    res_np1d = top_reason_codes(explainer, X_val.iloc[0].to_numpy(), cols, k=3)

    assert res_df == res_series
    assert res_df == res_np2d
    assert res_df == res_np1d
    assert all(r["contribution"] > 0 for r in res_df)
