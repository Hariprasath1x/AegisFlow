"""Champion (XGBoost) and alternative (LightGBM) fraud models.

Both take the same matrix and return raw probabilities; calibration is a separate
step so the two are comparable before and after it. scale_pos_weight, not
resampling: SMOTE on transaction data invents customers who never existed, and the
invented ones are exactly where the model then makes its mistakes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_binary_labels(y, name: str = "y") -> np.ndarray:
    """Validate that labels are 1D binary (0 or 1) numeric or boolean."""
    try:
        arr = np.asarray(y)
    except Exception as exc:
        raise ValueError(f"{name} must be a valid array-like: {exc}") from exc

    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.ndim}D array")
    if len(arr) == 0:
        raise ValueError(f"{name} cannot be empty")

    if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain only binary labels (0 or 1), non-finite found")
        if not np.all(np.isin(arr, [0, 1])):
            raise ValueError(f"{name} must contain only binary labels (0 or 1)")
        return arr.astype(int)
    elif np.issubdtype(arr.dtype, np.bool_):
        return arr.astype(int)
    else:
        raise ValueError(f"{name} must contain only numeric binary labels (0 or 1), got dtype {arr.dtype}")


def pos_weight(y) -> float:
    """Calculate positive class weight: negative samples / positive samples.

    Clamps positive count to at least 1 to prevent division by zero.
    """
    y_arr = _validate_binary_labels(y, "y")
    pos = max(int(y_arr.sum()), 1)
    return float((len(y_arr) - pos) / pos)


def train_xgboost(X_train, y_train, X_val, y_val, params: dict | None = None):
    """Train XGBoost champion model with early stopping on validation set."""
    import xgboost as xgb

    params = {} if params is None else params
    y_tr = _validate_binary_labels(y_train, "y_train")
    y_v = _validate_binary_labels(y_val, "y_val")

    if len(X_train) != len(y_tr):
        raise ValueError(f"X_train length ({len(X_train)}) != y_train length ({len(y_tr)})")
    if len(X_val) != len(y_v):
        raise ValueError(f"X_val length ({len(X_val)}) != y_val length ({len(y_v)})")

    es_rounds = params.get("early_stopping_rounds", 60)
    model = xgb.XGBClassifier(
        n_estimators=params.get("n_estimators", 800),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        min_child_weight=params.get("min_child_weight", 5),
        reg_lambda=params.get("reg_lambda", 1.0),
        scale_pos_weight=pos_weight(y_tr),
        eval_metric="aucpr",
        tree_method="hist",
        early_stopping_rounds=es_rounds if es_rounds and es_rounds > 0 else None,
        random_state=params.get("seed", params.get("random_state", 42)),
        n_jobs=params.get("n_jobs", 4),
    )
    model.fit(X_train, y_tr, eval_set=[(X_val, y_v)], verbose=False)
    return model, model.predict_proba(X_val)[:, 1]


def train_lightgbm(X_train, y_train, X_val, y_val, params: dict | None = None):
    """Train LightGBM challenger model with early stopping on validation set."""
    import lightgbm as lgb

    params = {} if params is None else params
    y_tr = _validate_binary_labels(y_train, "y_train")
    y_v = _validate_binary_labels(y_val, "y_val")

    if len(X_train) != len(y_tr):
        raise ValueError(f"X_train length ({len(X_train)}) != y_train length ({len(y_tr)})")
    if len(X_val) != len(y_v):
        raise ValueError(f"X_val length ({len(X_val)}) != y_val length ({len(y_v)})")

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=params.get("n_estimators", 800),
        learning_rate=params.get("learning_rate", 0.05),
        num_leaves=params.get("num_leaves", 63),
        min_child_samples=params.get("min_child_samples", 50),
        subsample=params.get("subsample", 0.8),
        subsample_freq=1,
        colsample_bytree=params.get("colsample_bytree", 0.8),
        scale_pos_weight=pos_weight(y_tr),
        random_state=params.get("seed", params.get("random_state", 42)),
        n_jobs=params.get("n_jobs", 4),
        verbose=-1,
    )
    callbacks = []
    es_rounds = params.get("early_stopping_rounds", 60)
    if es_rounds and es_rounds > 0:
        callbacks.append(lgb.early_stopping(es_rounds, verbose=False))

    model.fit(
        X_train,
        y_tr,
        eval_set=[(X_val, y_v)],
        eval_metric="average_precision",
        callbacks=callbacks,
    )
    return model, model.predict_proba(X_val)[:, 1]


def shap_explainer(model, X_background=None):
    """Built at training time and saved with the model. M4's scoring service loads
    it once at startup so a per-request explanation costs about a millisecond
    instead of rebuilding the explainer on every call."""
    import shap

    return shap.TreeExplainer(model, data=None, feature_perturbation="tree_path_dependent")


def top_reason_codes(explainer, X_row, feature_names, k: int = 5) -> list[dict]:
    """The top-k features pushing this transaction's score UP, which is what a
    compliance officer has to be able to read (spec objective 5.3)."""
    if k <= 0:
        return []

    if isinstance(X_row, pd.Series):
        X_input = X_row.to_frame().T
    elif hasattr(X_row, "ndim") and X_row.ndim == 1:
        X_input = np.asarray(X_row).reshape(1, -1)
    elif isinstance(X_row, (list, tuple)) and (
        len(X_row) == 0 or not isinstance(X_row[0], (list, tuple, np.ndarray, pd.Series))
    ):
        X_input = np.asarray(X_row).reshape(1, -1)
    else:
        X_input = X_row

    raw = explainer.shap_values(X_input)
    if isinstance(raw, list) and len(raw) == 2:
        raw = raw[1]
    arr = np.asarray(raw)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        arr = arr[0, :, 1]
    values = arr.reshape(-1)

    feature_names = list(feature_names)
    if len(feature_names) != len(values):
        raise ValueError(
            f"Length of feature_names ({len(feature_names)}) does not match "
            f"number of SHAP values ({len(values)})"
        )

    order = np.argsort(-values, kind="stable")[:k]
    return [
        {"feature": feature_names[i], "contribution": float(values[i])}
        for i in order
        if values[i] > 0
    ]
