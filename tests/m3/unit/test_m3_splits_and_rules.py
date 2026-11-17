"""Time splits, label delay and the embargo - the three places a fraud model
quietly reads the future or claims a lift it did not earn.

Unit tests for AegisFlow Milestone 3 - Step 3.2.
(Note: Rules baseline unit tests belong to Step 3.5 and are intentionally omitted here).
"""
from __future__ import annotations

import datetime
import numpy as np
import pandas as pd
import pytest

from aegis_ml.training.splits import (
    TRAIN_WEEKS,
    TEST_WEEKS,
    check_no_time_leak,
    label_delay_mask,
    time_split,
    week_index,
)

START = pd.Timestamp("2026-01-05", tz="UTC")


def frame(days: int = 70) -> pd.DataFrame:
    """Generate hourly transaction timestamps starting from START."""
    ts = pd.date_range(START, periods=days * 24, freq="h", tz="UTC")
    return pd.DataFrame({"ts": ts})


def test_week_index_counts_from_the_first_row() -> None:
    """1. Week indexing starts at zero and indexes 7-day increments."""
    ts = frame(21)["ts"]
    w = week_index(ts)
    assert w.min() == 0 and w.max() == 2
    assert (w[ts < START + pd.Timedelta(days=7)] == 0).all()
    assert (w[(ts >= START + pd.Timedelta(days=7)) & (ts < START + pd.Timedelta(days=14))] == 1).all()
    assert (w[ts >= START + pd.Timedelta(days=14)] == 2).all()


def test_week_index_explicit_origin() -> None:
    """2. Explicit origin produces correct week indices across inputs."""
    ts = pd.Series(pd.to_datetime(["2026-01-10", "2026-01-18"], utc=True))
    w = week_index(ts, origin="2026-01-01")
    # 2026-01-10 is 9 days after 2026-01-01 -> week 1
    # 2026-01-18 is 17 days after 2026-01-01 -> week 2
    assert list(w) == [1, 2]

    # Empty series
    empty = pd.Series([], dtype="datetime64[ns, UTC]")
    assert len(week_index(empty)) == 0


def test_time_split_windows_and_embargo() -> None:
    """3 & 4 & 5 & 6. Training window (weeks 1-6), embargo (24h drop), test window (weeks 7-8), and post-window drop."""
    df = frame(70)
    df["split"] = time_split(df["ts"], train_weeks=6, test_weeks=2, embargo_hours=24)
    tr = df[df.split == "train"]
    te = df[df.split == "test"]
    dr = df[df.split == "drop"]

    # 6 weeks total = 1008 hours. Embargo removes 24 hours -> 984 train hours
    assert len(tr) == (6 * 7 * 24) - 24
    # 2 weeks total = 336 test hours
    assert len(te) == 2 * 7 * 24
    # 70 days total = 1680 hours. Dropped = 24 (embargo) + (70 - 56)*24 (post-window 14 days) = 360 hours
    assert len(dr) == 24 + 336

    # Verify no leak between train and test
    assert check_no_time_leak(tr["ts"], te["ts"]) == []

    # Verify exact embargo gap
    embargo_gap = te["ts"].min() - tr["ts"].max()
    assert embargo_gap >= pd.Timedelta(hours=24)
    # The last train timestamp is 23:00 on day 41, test starts 00:00 on day 43 -> 25 hours difference
    assert embargo_gap == pd.Timedelta(hours=25)


def test_exact_24_hour_embargo_dropped() -> None:
    """4. Exact 24 hours immediately before the cut are assigned 'drop'."""
    df = frame(70)
    df["split"] = time_split(df["ts"], train_weeks=6, test_weeks=2, embargo_hours=24)
    cut = START + pd.Timedelta(weeks=6)
    embargo_start = cut - pd.Timedelta(hours=24)

    embargo_slice = df[(df["ts"] >= embargo_start) & (df["ts"] < cut)]
    assert len(embargo_slice) == 24
    assert (embargo_slice["split"] == "drop").all()


def test_leak_checker_fires_on_empty_and_overlapping_split() -> None:
    """7 & 8. Leak checker rejects empty and overlapping splits."""
    df = frame(70)

    # Empty inputs
    assert check_no_time_leak(df["ts"], [])[0].startswith("train or test")
    assert check_no_time_leak([], df["ts"])[0].startswith("train or test")
    assert check_no_time_leak([], [])[0].startswith("train or test")

    # Identical / overlapping inputs
    problems = check_no_time_leak(df["ts"], df["ts"])
    assert len(problems) > 0 and "overlaps" in problems[0]

    # Inverted order
    inverted = check_no_time_leak(df["ts"][200:], df["ts"][:100])
    assert len(inverted) > 0 and "overlaps" in inverted[0]

    # Boundary touch (train ends exactly when test starts)
    t_boundary = pd.Timestamp("2026-02-01", tz="UTC")
    touch = check_no_time_leak([t_boundary], [t_boundary])
    assert len(touch) > 0 and "overlaps" in touch[0]


def test_label_delay_guide_baseline_fixture() -> None:
    """9 & 10. Exact guide test fixture from page 9."""
    ts = pd.Series(pd.to_datetime(["2026-03-01", "2026-03-20", "2026-03-20"], utc=True))
    reported = pd.Series(pd.to_datetime([None, "2026-03-22", None], utc=True))
    known = label_delay_mask(ts, reported, as_of="2026-04-02", horizon_days=30)
    # Row 0: 32 days old, clean -> matured (True)
    # Row 1: 13 days old, fraud reported on 2026-03-22 <= 2026-04-02 -> known (True)
    # Row 2: 13 days old, clean -> unmatured (False)
    assert list(known) == [True, True, False]


def test_label_delay_excludes_unlanded_fraud_past_horizon() -> None:
    """10. A fraud transaction whose chargeback has NOT landed by as_of is excluded even if ts was > 30 days ago."""
    # Transaction occurred 35 days before as_of, but chargeback lands 10 days in the future relative to as_of
    ts = pd.Series(pd.to_datetime(["2026-03-01"], utc=True))
    reported = pd.Series(pd.to_datetime(["2026-04-15"], utc=True))
    as_of = "2026-04-05"

    known = label_delay_mask(ts, reported, as_of=as_of, horizon_days=30)
    # Must NOT mature as clean, and chargeback is unlanded -> excluded
    assert list(known) == [False]


def test_label_delay_includes_fraud_when_chargeback_lands() -> None:
    """9. A fraud transaction is included if chargeback has landed on or before as_of, regardless of age."""
    # Transaction 5 days ago, chargeback reported today (on as_of)
    ts = pd.Series(pd.to_datetime(["2026-04-01"], utc=True))
    reported = pd.Series(pd.to_datetime(["2026-04-06"], utc=True))
    as_of = "2026-04-06"

    known = label_delay_mask(ts, reported, as_of=as_of, horizon_days=30)
    assert list(known) == [True]


def test_timezone_naive_inputs_normalized_to_utc() -> None:
    """11. Timezone-naive inputs are consistently normalized to UTC across functions."""
    naive_ts = pd.Series(["2026-01-05 00:00:00", "2026-01-12 00:00:00"])
    w = week_index(naive_ts)
    assert list(w) == [0, 1]

    # Leak checker with naive strings
    assert check_no_time_leak(["2026-01-01"], ["2026-01-02"]) == []
    assert len(check_no_time_leak(["2026-01-05"], ["2026-01-02"])) > 0

    # Label delay mask with naive strings
    m = label_delay_mask(["2026-03-01"], [None], as_of="2026-04-02", horizon_days=30)
    assert list(m) == [True]


def test_boundary_and_invalid_inputs() -> None:
    """12. Boundary conditions, parameter validation, and empty inputs."""
    # Negative horizon
    with pytest.raises(ValueError, match="horizon_days must be non-negative"):
        label_delay_mask(["2026-03-01"], [None], as_of="2026-04-02", horizon_days=-1)

    # Length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        label_delay_mask(["2026-03-01", "2026-03-02"], [None], as_of="2026-04-02")

    # Invalid time_split arguments
    with pytest.raises(ValueError, match="train_weeks must be positive"):
        time_split(pd.Series(["2026-01-01"]), train_weeks=0)
    with pytest.raises(ValueError, match="test_weeks must be positive"):
        time_split(pd.Series(["2026-01-01"]), test_weeks=-1)
    with pytest.raises(ValueError, match="embargo_hours must be non-negative"):
        time_split(pd.Series(["2026-01-01"]), embargo_hours=-5)

    # Empty inputs return empty results
    assert len(time_split(pd.Series([], dtype="datetime64[ns]"))) == 0
    assert len(label_delay_mask([], [], as_of="2026-04-02")) == 0


def test_week_index_rejects_nat_inputs() -> None:
    """Fail-closed validation: week_index rejects missing timestamps in ts or explicit origin."""
    # NaT in ts series
    with pytest.raises(ValueError, match="ts contains missing timestamps \\(NaT\\)"):
        week_index(pd.Series(["2026-01-05", None]))

    # NaT in ts datetime series
    with pytest.raises(ValueError, match="ts contains missing timestamps \\(NaT\\)"):
        week_index(pd.Series([pd.Timestamp("2026-01-05"), pd.NaT]))

    # Explicit origin is NaT
    with pytest.raises(ValueError, match="origin cannot be missing \\(NaT\\)"):
        week_index(pd.Series(["2026-01-05"]), origin=pd.NaT)

    with pytest.raises(ValueError, match="origin cannot be missing \\(NaT\\)"):
        week_index(pd.Series(["2026-01-05"]), origin="NaT")


def test_check_no_time_leak_detects_nat_timestamps() -> None:
    """Fail-closed validation: check_no_time_leak detects NaT in train and test splits."""
    valid_train = ["2026-01-01", "2026-01-02"]
    valid_test = ["2026-01-10", "2026-01-11"]

    # NaT in train
    p_train = check_no_time_leak(["2026-01-01", None], valid_test)
    assert len(p_train) == 1
    assert "train split contains missing timestamps (NaT)" in p_train[0]

    # NaT in test
    p_test = check_no_time_leak(valid_train, ["2026-01-10", pd.NaT])
    assert len(p_test) == 1
    assert "test split contains missing timestamps (NaT)" in p_test[0]

    # NaT in both
    p_both = check_no_time_leak([pd.NaT], [pd.NaT])
    assert len(p_both) == 2
    assert "train split contains missing timestamps (NaT)" in p_both[0]
    assert "test split contains missing timestamps (NaT)" in p_both[1]
