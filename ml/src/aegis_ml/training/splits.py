"""Time-based splits. There is exactly one correct way to split fraud data.

Fraud is adversarial and seasonal: rings appear, get blocked, and are replaced.
A random split lets the model see next week's ring while training, so it scores
beautifully offline and collapses in production. Train on weeks 1-6, test on 7-8
(spec section 14), and never shuffle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRAIN_WEEKS = 6
TEST_WEEKS = 2


def _normalize_as_of(as_of) -> pd.Timestamp:
    """Normalize as_of to a UTC-aware Timestamp."""
    ts = pd.Timestamp(as_of)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def week_index(ts: pd.Series, origin=None) -> pd.Series:
    """0-based week number from the first transaction in the dataset."""
    if origin is not None:
        orig = _normalize_as_of(origin)
        if pd.isna(orig):
            raise ValueError("origin cannot be missing (NaT)")
    else:
        orig = None

    if len(ts) == 0:
        return pd.Series(dtype=int, index=getattr(ts, "index", None))

    ts_dt = pd.to_datetime(ts, utc=True)
    if ts_dt.isna().any():
        raise ValueError("ts contains missing timestamps (NaT)")

    if orig is None:
        orig = ts_dt.min()

    diff_sec = (ts_dt - orig).dt.total_seconds()
    return (diff_sec // (7 * 86400)).astype(int)


def time_split(
    ts: pd.Series,
    train_weeks: int = TRAIN_WEEKS,
    test_weeks: int = TEST_WEEKS,
    embargo_hours: int = 24,
) -> pd.Series:
    """'train' / 'test' / 'drop'. The embargo removes the hours straddling the
    boundary: a transaction scored just before the cut shares velocity windows with
    transactions just after it, which is leakage through the feature store rather
    than through the label."""
    if train_weeks <= 0:
        raise ValueError(f"train_weeks must be positive, got {train_weeks}")
    if test_weeks <= 0:
        raise ValueError(f"test_weeks must be positive, got {test_weeks}")
    if embargo_hours < 0:
        raise ValueError(f"embargo_hours must be non-negative, got {embargo_hours}")

    if len(ts) == 0:
        return pd.Series(dtype=object, index=getattr(ts, "index", None))

    w = week_index(ts)
    ts_dt = pd.to_datetime(ts, utc=True)
    cut = ts_dt.min() + pd.Timedelta(weeks=train_weeks)
    out = pd.Series("drop", index=ts.index, dtype=object)
    out[(w < train_weeks) & (ts_dt < cut - pd.Timedelta(hours=embargo_hours))] = "train"
    out[(w >= train_weeks) & (w < train_weeks + test_weeks)] = "test"
    return out


def check_no_time_leak(train_ts, test_ts) -> list[str]:
    """Empty list means clean: every training timestamp is strictly before every
    test timestamp."""
    tr = pd.to_datetime(train_ts, utc=True)
    te = pd.to_datetime(test_ts, utc=True)
    problems = []
    if len(tr) == 0 or len(te) == 0:
        problems.append("train or test split is empty")
        return problems
    if tr.isna().any():
        problems.append("train split contains missing timestamps (NaT)")
    if te.isna().any():
        problems.append("test split contains missing timestamps (NaT)")
    if problems:
        return problems
    if tr.max() >= te.min():
        problems.append(
            f"train ends {tr.max()} but test starts {te.min()} - the split overlaps"
        )
    return problems


def label_delay_mask(ts, reported_at, as_of, horizon_days: int = 30) -> np.ndarray:
    """Which rows had a KNOWN label at `as_of`. Chargebacks arrive weeks late, so a
    model 'evaluated' on labels that did not exist yet is reading the future. Rows
    whose label had not landed are excluded from training, not silently marked legit."""
    if horizon_days < 0:
        raise ValueError(f"horizon_days must be non-negative, got {horizon_days}")

    ts_s = pd.Series(ts) if not isinstance(ts, pd.Series) else ts
    rep_s = pd.Series(reported_at) if not isinstance(reported_at, pd.Series) else reported_at

    if len(ts_s) != len(rep_s):
        raise ValueError(f"Length mismatch: ts ({len(ts_s)}) vs reported_at ({len(rep_s)})")

    if len(ts_s) == 0:
        return np.array([], dtype=bool)

    rep_s = pd.Series(rep_s.to_numpy(), index=ts_s.index)
    ts_dt = pd.to_datetime(ts_s, utc=True)
    rep_dt = pd.to_datetime(rep_s, utc=True)
    as_of_ts = _normalize_as_of(as_of)

    # Legitimate transactions (no chargeback) mature once horizon_days have elapsed without dispute.
    matured = rep_dt.isna() & (ts_dt <= as_of_ts - pd.Timedelta(days=horizon_days))

    # Fraud transactions are known if and only if the chargeback has landed on or before as_of.
    # If a chargeback is pending/unlanded (rep_dt > as_of_ts), it is excluded rather than treated as mature clean.
    known = rep_dt.notna() & (rep_dt <= as_of_ts)

    return (matured | known).to_numpy()
