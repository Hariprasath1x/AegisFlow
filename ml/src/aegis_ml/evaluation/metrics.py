"""Fraud metrics, cost curves, and cohort slices.

Accuracy is never computed here: with a 3% fraud rate a model that approves
everything scores 97%. Everything below is imbalance-aware.
Conforms to AegisFlow_M3_Implementation_Guide.pdf Step 3.6.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_curve

__all__ = [
    "recall_at_fpr",
    "pr_auc",
    "lift_over",
    "cost_curve",
    "best_threshold",
    "cohort_slices",
    "amount_bands",
]


def _validate_binary_labels(y, name: str = "y_true") -> np.ndarray:
    """Validate that input is a 1D array containing strictly binary labels (0 or 1)."""
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {arr.shape}")
    if len(arr) == 0:
        return np.array([], dtype=int)

    # Reject string, bytes, or object containers with invalid types
    if arr.dtype.kind in ("U", "S", "O"):
        for val in arr:
            if isinstance(val, (str, bytes)):
                raise ValueError(f"{name} must contain only binary labels (0 or 1), got string")
            if not isinstance(val, (int, float, np.integer, np.floating, bool, np.bool_)):
                raise ValueError(
                    f"{name} must contain only binary labels (0 or 1), got {type(val).__name__}"
                )

    if arr.dtype.kind in ("f", "c"):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain only binary labels (0 or 1), non-finite found")

    try:
        is_zero = arr == 0
        is_one = arr == 1
        if not np.all(is_zero | is_one):
            raise ValueError(f"{name} must contain only binary labels (0 or 1)")
    except Exception as exc:
        raise ValueError(f"{name} must contain only binary labels (0 or 1)") from exc

    return arr.astype(int)


def _validate_1d_numeric(arr, name: str) -> np.ndarray:
    """Validate that input is a 1D array of finite numeric floats."""
    try:
        a = np.asarray(arr, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc

    if a.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {a.shape}")
    if len(a) > 0 and not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain only finite numeric values (no NaNs or Infs)")
    return a


def recall_at_fpr(y_true, scores, max_fpr: float = 0.01) -> dict:
    """The headline metric (spec objective 5.2): how much fraud is caught while
    falsely flagging at most max_fpr of legitimate traffic.

    Returns the operating point, not just the number, because M4's threshold API
    needs the threshold.
    """
    if not np.isfinite(max_fpr) or max_fpr < 0.0 or max_fpr > 1.0:
        raise ValueError(f"max_fpr must be a finite float between 0.0 and 1.0, got {max_fpr}")

    y_int = _validate_binary_labels(y_true, "y_true")
    s_arr = _validate_1d_numeric(scores, "scores")

    if len(y_int) != len(s_arr):
        raise ValueError(f"Length mismatch: y_true ({len(y_int)}) vs scores ({len(s_arr)})")

    if len(y_int) == 0 or y_int.sum() == 0 or (y_int == 0).sum() == 0:
        return {
            "recall": 0.0,
            "fpr": 0.0,
            "threshold": float("inf"),
            "max_fpr": float(max_fpr),
        }

    fpr, tpr, thr = roc_curve(y_int, s_arr)
    ok = fpr <= max_fpr
    i = int(np.argmax(np.where(ok, tpr, -1.0)))
    return {
        "recall": float(tpr[i]),
        "fpr": float(fpr[i]),
        "threshold": float(thr[i]),
        "max_fpr": float(max_fpr),
    }


def pr_auc(y_true, scores) -> float:
    """Area under the Precision-Recall curve (average precision)."""
    y_int = _validate_binary_labels(y_true, "y_true")
    s_arr = _validate_1d_numeric(scores, "scores")

    if len(y_int) != len(s_arr):
        raise ValueError(f"Length mismatch: y_true ({len(y_int)}) vs scores ({len(s_arr)})")

    if len(y_int) == 0 or y_int.sum() == 0 or (y_int == 0).sum() == 0:
        return 0.0

    return float(average_precision_score(y_int, s_arr))


def lift_over(baseline_recall: float, model_recall: float) -> float:
    """Relative improvement in recall at the same FPR - the number the spec's 30%
    objective is stated in.
    """
    if not np.isfinite(baseline_recall) or not np.isfinite(model_recall):
        return 0.0
    return 0.0 if baseline_recall <= 0 else float((model_recall - baseline_recall) / baseline_recall)


def cost_curve(
    y_true,
    scores,
    amounts,
    review_cost: float = 3.0,
    friction_cost: float = 12.0,
    grid: int = 60,
) -> list[dict]:
    """Expected loss per threshold. Three costs, all real:
    - a missed fraud costs the transaction amount (chargeback),
    - a declined legitimate customer costs friction_cost (lost margin plus churn),
    - every flagged transaction costs review_cost of analyst time.

    The optimum is almost never 0.5, which is exactly the point.
    """
    if not np.isfinite(review_cost) or review_cost < 0.0:
        raise ValueError("review_cost must be a finite non-negative number")
    if not np.isfinite(friction_cost) or friction_cost < 0.0:
        raise ValueError("friction_cost must be a finite non-negative number")
    if not isinstance(grid, (int, np.integer)) or grid <= 0:
        raise ValueError("grid must be a positive integer")

    y = _validate_binary_labels(y_true, "y_true")
    s = _validate_1d_numeric(scores, "scores")
    amt = _validate_1d_numeric(amounts, "amounts")

    if not (len(y) == len(s) == len(amt)):
        raise ValueError(
            f"Length mismatch: y_true ({len(y)}), scores ({len(s)}), amounts ({len(amt)})"
        )

    if len(y) == 0:
        return []

    s_min = float(s.min())
    s_max = float(s.max())
    if s_min == s_max:
        thresholds = np.array([s_min], dtype=float)
    else:
        thresholds = np.linspace(s_min, s_max, int(grid))

    out = []
    n = len(y)
    for t in thresholds:
        flag = s >= t
        missed = float(amt[(~flag) & (y == 1)].sum())
        friction = float(friction_cost * np.sum(flag & (y == 0)))
        review = float(review_cost * flag.sum())
        out.append(
            {
                "threshold": float(t),
                "missed_fraud_value": missed,
                "friction_cost": friction,
                "review_cost": review,
                "total_cost": missed + friction + review,
                "flag_rate": float(flag.mean()) if n > 0 else 0.0,
            }
        )
    return out


def best_threshold(curve: list[dict]) -> dict:
    """Return the operating point dictionary with minimum total_cost."""
    if not curve:
        raise ValueError("curve must not be empty")
    return min(curve, key=lambda r: r["total_cost"])


def cohort_slices(y_true, scores, cohorts: dict, max_fpr: float = 0.01) -> dict:
    """recall@FPR inside each cohort (amount band, channel, new vs established).

    A model can hit its headline number and still be blind to one channel; this is
    how you find that before a fraud ring does.
    """
    if not np.isfinite(max_fpr) or max_fpr < 0.0 or max_fpr > 1.0:
        raise ValueError(f"max_fpr must be a finite float between 0.0 and 1.0, got {max_fpr}")
    if not isinstance(cohorts, dict):
        raise TypeError(f"cohorts must be a dict, got {type(cohorts).__name__}")

    y_arr = _validate_binary_labels(y_true, "y_true")
    s_arr = _validate_1d_numeric(scores, "scores")

    if len(y_arr) != len(s_arr):
        raise ValueError(f"Length mismatch: y_true ({len(y_arr)}) vs scores ({len(s_arr)})")

    out = {}
    for name, mask in cohorts.items():
        m_raw = np.asarray(mask)
        if m_raw.ndim != 1:
            raise ValueError(f"Cohort mask '{name}' must be 1D, got shape {m_raw.shape}")
        if len(m_raw) != len(y_arr):
            raise ValueError(
                f"Cohort mask '{name}' length ({len(m_raw)}) does not match y_true ({len(y_arr)})"
            )
        if m_raw.dtype.kind not in ("b", "i", "u", "f"):
            raise ValueError(f"Cohort mask '{name}' contains invalid non-boolean elements")
        if m_raw.dtype.kind in ("f", "i", "u"):
            if not np.all((m_raw == 0) | (m_raw == 1)):
                raise ValueError(f"Cohort mask '{name}' must be boolean or binary indicators")

        m = m_raw.astype(bool)
        n_cohort = int(m.sum())
        if n_cohort < 50 or int(y_arr[m].sum()) == 0:
            out[name] = {
                "n": n_cohort,
                "recall": None,
                "note": "too few rows or no fraud in cohort",
            }
            continue
        r = recall_at_fpr(y_arr[m], s_arr[m], max_fpr)
        out[name] = {"n": n_cohort, **r}
    return out


def amount_bands(amounts, edges=(0, 50, 250, 1000, np.inf)) -> dict[str, np.ndarray]:
    """Partition transaction amounts into boolean cohort masks by bin edges."""
    amt = _validate_1d_numeric(amounts, "amounts")
    if len(edges) < 2:
        raise ValueError("edges must contain at least two boundary values")
    edges_arr = np.asarray(edges, dtype=float)
    if not np.all(np.diff(edges_arr) > 0):
        raise ValueError("edges must be strictly monotonically increasing")

    return {
        f"amount_{int(lo)}_{'inf' if np.isinf(hi) else int(hi)}": (amt >= lo) & (amt < hi)
        for lo, hi in zip(edges[:-1], edges[1:])
    }
