"""Isotonic calibration and the reliability check (spec section 14).

A raw XGBoost margin is a ranking, not a probability. Cost-based decisioning
multiplies the score by a transaction amount, so a score of 0.9 has to mean roughly
a 90% chance of fraud or every cost number downstream is nonsense. Isotonic
regression is fitted on a held-out slice, never on the training rows.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def _validate_binary_labels(y, name: str = "y_true") -> np.ndarray:
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


def _validate_scores(scores, name: str = "scores", allow_empty: bool = False) -> np.ndarray:
    """Validate that scores are 1D finite numeric values."""
    try:
        arr = np.asarray(scores, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be numeric: {exc}") from exc

    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.ndim}D array")
    if len(arr) == 0 and not allow_empty:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite numeric values")
    return arr


def _validate_probs(probs, name: str = "probs") -> np.ndarray:
    """Validate that probabilities are 1D finite numeric values bounded in [0, 1]."""
    arr = _validate_scores(probs, name=name, allow_empty=False)
    if not np.all((arr >= 0.0) & (arr <= 1.0)):
        raise ValueError(f"{name} must be bounded in [0.0, 1.0]")
    return arr


class Calibrator:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.fitted = False

    def fit(self, scores, y_true) -> "Calibrator":
        s = _validate_scores(scores, "scores")
        y = _validate_binary_labels(y_true, "y_true")
        if len(s) != len(y):
            raise ValueError(f"scores length ({len(s)}) != y_true length ({len(y)})")
        self.iso.fit(s, y)
        self.fitted = True
        return self

    def transform(self, scores) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("calibrator not fitted")
        s = _validate_scores(scores, "scores", allow_empty=True)
        if len(s) == 0:
            return np.array([], dtype=float)
        return np.clip(self.iso.predict(s), 0.0, 1.0)


def reliability(y_true, probs, bins: int = 10) -> list[dict]:
    """Per-bin predicted vs observed fraud rate - the table behind the reliability
    curve in the HTML report."""
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins <= 0:
        raise ValueError(f"bins must be a positive integer, got {bins}")

    y = _validate_binary_labels(y_true, "y_true")
    p = _validate_probs(probs, "probs")
    if len(y) != len(p):
        raise ValueError(f"y_true length ({len(y)}) != probs length ({len(p)})")

    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not m.any():
            continue
        out.append({
            "bin": f"[{lo:.1f},{hi:.1f})",
            "n": int(m.sum()),
            "predicted": float(p[m].mean()),
            "observed": float(y[m].mean()),
        })
    return out


def expected_calibration_error(y_true, probs, bins: int = 10) -> float:
    """Weighted mean gap between predicted and observed rates. Spec section 14 asks
    for calibration; this is the single number that summarises it."""
    rows = reliability(y_true, probs, bins)
    n = sum(r["n"] for r in rows) or 1
    return float(sum(r["n"] * abs(r["predicted"] - r["observed"]) for r in rows) / n)
