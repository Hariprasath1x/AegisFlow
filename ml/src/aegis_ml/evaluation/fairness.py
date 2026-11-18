"""Disparate-impact screens (spec section 14, 'fairness screens').

This is not a legal compliance tool and the guide says so out loud. It answers one
narrow question: does the model decline some cohorts at a much higher rate than
others at the SAME operating threshold, and is that gap explained by the fraud rate
in the cohort or not? An unexplained gap is a finding for the model card, not an
automatic failure.
"""
from __future__ import annotations

import math
import numpy as np

FOUR_FIFTHS = 0.80


def _validate_binary_array(arr, name: str) -> np.ndarray:
    """Validate 1D binary or boolean array."""
    try:
        a = np.asarray(arr)
    except Exception as exc:
        raise ValueError(f"{name} must be array-like: {exc}") from exc

    if a.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {a.ndim}D array")
    if len(a) == 0:
        raise ValueError(f"{name} cannot be empty")

    if np.issubdtype(a.dtype, np.floating) or np.issubdtype(a.dtype, np.integer):
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must contain only binary labels (0 or 1), non-finite found")
        if not np.all(np.isin(a, [0, 1])):
            raise ValueError(f"{name} must contain only binary labels (0 or 1)")
        return a.astype(int)
    elif np.issubdtype(a.dtype, np.bool_):
        return a.astype(int)
    else:
        raise ValueError(f"{name} must contain only numeric binary labels (0 or 1), got dtype {a.dtype}")


def group_rates(y_true, flags, groups: dict) -> dict:
    """Calculate screening rates per cohort group."""
    if not isinstance(groups, dict):
        raise ValueError("groups must be a dictionary")

    y = _validate_binary_array(y_true, "y_true")
    f = _validate_binary_array(flags, "flags")
    if len(y) != len(f):
        raise ValueError(f"y_true length ({len(y)}) != flags length ({len(f)})")

    out = {}
    for name, mask in groups.items():
        try:
            m = np.asarray(mask)
        except Exception as exc:
            raise ValueError(f"mask for group '{name}' must be array-like: {exc}") from exc

        if m.ndim != 1:
            raise ValueError(f"mask for group '{name}' must be 1D, got {m.ndim}D")
        if len(m) != len(y):
            raise ValueError(f"mask for group '{name}' length ({len(m)}) != y_true length ({len(y)})")

        if np.issubdtype(m.dtype, np.floating) or np.issubdtype(m.dtype, np.integer):
            if not np.all(np.isfinite(m)) or not np.all(np.isin(m, [0, 1])):
                raise ValueError(f"mask for group '{name}' must be boolean or binary")
            m_bool = m.astype(bool)
        elif np.issubdtype(m.dtype, np.bool_):
            m_bool = m.copy()
        else:
            raise ValueError(f"mask for group '{name}' must be boolean or binary, got dtype {m.dtype}")

        if m_bool.sum() < 50:
            out[name] = {"n": int(m_bool.sum()), "note": "cohort too small to screen"}
            continue

        legit = m_bool & (y == 0)
        out[name] = {
            "n": int(m_bool.sum()),
            "flag_rate": float(f[m_bool].mean()),
            "false_positive_rate": float(f[legit].mean()) if legit.any() else None,
            "fraud_rate": float(y[m_bool].mean()),
        }
    return out


def disparate_impact(rates: dict, key: str = "flag_rate") -> dict:
    """Ratio of the lowest to the highest cohort rate. Below 0.8 is the classic
    four-fifths flag used in US adverse-impact analysis."""
    if not isinstance(rates, dict):
        raise ValueError("rates must be a dictionary")
    if not isinstance(key, str):
        raise ValueError("key must be a string")

    vals = {}
    for k, v in rates.items():
        if isinstance(v, dict) and v.get(key) is not None:
            val = v[key]
            if isinstance(val, (int, float, np.floating, np.integer)) and not isinstance(val, bool):
                if not math.isnan(val) and not math.isinf(val):
                    vals[k] = float(val)

    if len(vals) < 2:
        return {"ratio": None, "note": "need at least two screenable cohorts"}

    lo_k, lo = min(vals.items(), key=lambda kv: kv[1])
    hi_k, hi = max(vals.items(), key=lambda kv: kv[1])
    ratio = (lo / hi) if hi > 0 else None
    return {
        "ratio": ratio,
        "lowest": lo_k,
        "highest": hi_k,
        "flag": ratio is not None and ratio < FOUR_FIFTHS,
        "note": "a gap tracking the cohort fraud rate is expected; an unexplained gap goes in the model card",
    }
