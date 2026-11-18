"""Deterministic six-rule fraud baseline for AegisFlow Milestone 3.

Provides auditable rule hits, weighted scoring, decisions, and explanations.
Conforms to AegisFlow_M3_Implementation_Guide.pdf Step 3.5.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rule:
    """Deterministic rule definition."""

    id: str
    description: str
    predicate: Callable[[pd.DataFrame], np.ndarray]
    weight: float = 1.0


def _pred_r001(df: pd.DataFrame) -> np.ndarray:
    """R001: amount above 50,000."""
    if "amount" not in df.columns:
        raise KeyError("Missing required column: 'amount'")
    amt = pd.to_numeric(df["amount"], errors="coerce").to_numpy(dtype=np.float64, copy=True)
    res = (amt > 50_000.0) & ~np.isnan(amt)
    return res.astype(bool)


def _pred_r002(df: pd.DataFrame) -> np.ndarray:
    """R002: new device for this account (device_age_days < 1, default 99)."""
    if "device_age_days" in df.columns:
        s = pd.to_numeric(df["device_age_days"], errors="coerce").fillna(99.0)
        vals = s.to_numpy(dtype=np.float64, copy=True)
    else:
        vals = np.full(len(df), 99.0, dtype=np.float64)
    res = (vals < 1.0) & ~np.isnan(vals)
    return res.astype(bool)


def _pred_r003(df: pd.DataFrame) -> np.ndarray:
    """R003: more than 5 transactions in the last hour (txn_count_1h > 5, default 0)."""
    if "txn_count_1h" in df.columns:
        s = pd.to_numeric(df["txn_count_1h"], errors="coerce").fillna(0.0)
        vals = s.to_numpy(dtype=np.float64, copy=True)
    else:
        vals = np.full(len(df), 0.0, dtype=np.float64)
    res = (vals > 5.0) & ~np.isnan(vals)
    return res.astype(bool)


def _pred_r004(df: pd.DataFrame) -> np.ndarray:
    """R004: amount > 5 * max(amount_mean_30d, 1.0) (default 1.0)."""
    if "amount" not in df.columns:
        raise KeyError("Missing required column: 'amount'")
    amt = pd.to_numeric(df["amount"], errors="coerce").to_numpy(dtype=np.float64, copy=True)
    if "amount_mean_30d" in df.columns:
        s = pd.to_numeric(df["amount_mean_30d"], errors="coerce").fillna(1.0)
        mean_val = s.to_numpy(dtype=np.float64, copy=True)
    else:
        mean_val = np.full(len(df), 1.0, dtype=np.float64)
    denom = np.maximum(mean_val, 1.0)
    threshold = 5.0 * denom
    res = (amt > threshold) & ~np.isnan(amt) & ~np.isnan(threshold)
    return res.astype(bool)


def _pred_r005(df: pd.DataFrame) -> np.ndarray:
    """R005: impossible travel: two countries within an hour (distinct_countries_1h > 1, default 1)."""
    if "distinct_countries_1h" in df.columns:
        s = pd.to_numeric(df["distinct_countries_1h"], errors="coerce").fillna(1.0)
        vals = s.to_numpy(dtype=np.float64, copy=True)
    else:
        vals = np.full(len(df), 1.0, dtype=np.float64)
    res = (vals > 1.0) & ~np.isnan(vals)
    return res.astype(bool)


def _pred_r006(df: pd.DataFrame) -> np.ndarray:
    """R006: device shared by more than 3 accounts (device_account_degree > 3, default 1)."""
    if "device_account_degree" in df.columns:
        s = pd.to_numeric(df["device_account_degree"], errors="coerce").fillna(1.0)
        vals = s.to_numpy(dtype=np.float64, copy=True)
    else:
        vals = np.full(len(df), 1.0, dtype=np.float64)
    res = (vals > 3.0) & ~np.isnan(vals)
    return res.astype(bool)


R001 = Rule(
    id="R001",
    description="amount above 50,000",
    predicate=_pred_r001,
    weight=1.0,
)
R002 = Rule(
    id="R002",
    description="new device for this account",
    predicate=_pred_r002,
    weight=1.0,
)
R003 = Rule(
    id="R003",
    description="more than 5 transactions in the last hour",
    predicate=_pred_r003,
    weight=1.0,
)
R004 = Rule(
    id="R004",
    description="amount more than 5x the account's 30-day average",
    predicate=_pred_r004,
    weight=1.0,
)
R005 = Rule(
    id="R005",
    description="impossible travel: two countries within an hour",
    predicate=_pred_r005,
    weight=2.0,
)
R006 = Rule(
    id="R006",
    description="device shared by more than 3 accounts",
    predicate=_pred_r006,
    weight=1.0,
)

RULES: list[Rule] = [R001, R002, R003, R004, R005, R006]

DEFAULT_OPTIONAL_COLUMNS: dict[str, float] = {
    "device_age_days": 99.0,
    "txn_count_1h": 0.0,
    "amount_mean_30d": 1.0,
    "distinct_countries_1h": 1.0,
    "device_account_degree": 1.0,
}


def hits(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Evaluate each rule independently and return rule ID to boolean NumPy array.

    Preserves row order and array length. Fails closed if 'amount' is missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    if "amount" not in df.columns:
        raise KeyError("Missing required column: 'amount'")

    return {rule.id: rule.predicate(df) for rule in RULES}


def score(df: pd.DataFrame) -> np.ndarray:
    """Calculate the weighted sum of fired rules for each row."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    if "amount" not in df.columns:
        raise KeyError("Missing required column: 'amount'")

    h = hits(df)
    n = len(df)
    total = np.zeros(n, dtype=np.float64)
    for rule in RULES:
        total += h[rule.id].astype(np.float64) * float(rule.weight)
    return total


def decisions(df: pd.DataFrame, min_weight: float = 1.0) -> np.ndarray:
    """Return boolean array where score(df) >= min_weight."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    s = score(df)
    return (s >= min_weight).astype(bool)


def explain(df: pd.DataFrame) -> list[list[str]]:
    """Return the ordered IDs of fired rules for every transaction."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    if "amount" not in df.columns:
        raise KeyError("Missing required column: 'amount'")

    h = hits(df)
    n = len(df)
    explanations: list[list[str]] = [[] for _ in range(n)]
    for rule in RULES:
        fired_mask = h[rule.id]
        for i in range(n):
            if fired_mask[i]:
                explanations[i].append(rule.id)
    return explanations
