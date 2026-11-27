"""Time splits, label delay, embargo, and deterministic six-rule fraud baseline.

Unit tests for AegisFlow Milestone 3 - Step 3.2 and Step 3.5.
"""
from __future__ import annotations

import dataclasses
import datetime
import numpy as np
import pandas as pd
import pytest

from aegis_ml.evaluation.rules_baseline import (
    Rule,
    R001,
    R002,
    R003,
    R004,
    R005,
    R006,
    RULES,
    DEFAULT_OPTIONAL_COLUMNS,
    hits,
    score,
    decisions,
    explain,
)
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
    """9 & 10. Under Policy A strict timestamp maturity, eligibility is determined strictly by transaction age (<= as_of - 30d)."""
    ts = pd.Series(pd.to_datetime(["2026-03-01", "2026-03-20", "2026-03-20"], utc=True))
    reported = pd.Series(pd.to_datetime([None, "2026-03-22", None], utc=True))
    known = label_delay_mask(ts, reported, as_of="2026-04-02", horizon_days=30)
    # as_of = 2026-04-02, maturity cutoff = 2026-03-03
    # Row 0: 32 days old (2026-03-01 <= 2026-03-03) -> mature (True)
    # Row 1: 13 days old (2026-03-20 > 2026-03-03) -> immature (False), early chargeback does not grant admission
    # Row 2: 13 days old (2026-03-20 > 2026-03-03) -> immature (False)
    assert list(known) == [True, False, False]


def test_label_delay_excludes_unlanded_fraud_past_horizon() -> None:
    """10. Transactions older than 30 days are mature under Policy A."""
    ts = pd.Series(pd.to_datetime(["2026-03-01"], utc=True))
    reported = pd.Series(pd.to_datetime(["2026-04-15"], utc=True))
    as_of = "2026-04-05"
    # 2026-03-01 is 35 days before as_of (<= 2026-03-06 cutoff) -> mature (True)
    known = label_delay_mask(ts, reported, as_of=as_of, horizon_days=30)
    assert list(known) == [True]


def test_label_delay_excludes_immature_fraud_even_when_chargeback_lands() -> None:
    """9. Immature transactions are excluded regardless of whether chargeback has landed early."""
    # Transaction 5 days ago, chargeback reported today (on as_of)
    ts = pd.Series(pd.to_datetime(["2026-04-01"], utc=True))
    reported = pd.Series(pd.to_datetime(["2026-04-06"], utc=True))
    as_of = "2026-04-06"
    # Age is only 5 days (< 30 days) -> excluded under Policy A
    known = label_delay_mask(ts, reported, as_of=as_of, horizon_days=30)
    assert list(known) == [False]


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


# ============================================================================
# Section 2: Deterministic Six-Rule Fraud Baseline Unit Tests (Step 3.5)
# ============================================================================


def test_rules_inventory_ids_order_descriptions_and_weights() -> None:
    """13. Exact six-rule inventory, IDs, ordering, descriptions, weights, list container, and frozen immutability."""
    assert len(RULES) == 6
    assert isinstance(RULES, list)

    expected = [
        ("R001", "amount above 50,000", 1.0),
        ("R002", "new device for this account", 1.0),
        ("R003", "more than 5 transactions in the last hour", 1.0),
        ("R004", "amount more than 5x the account's 30-day average", 1.0),
        ("R005", "impossible travel: two countries within an hour", 2.0),
        ("R006", "device shared by more than 3 accounts", 1.0),
    ]

    for rule, (exp_id, exp_desc, exp_weight) in zip(RULES, expected):
        assert rule.id == exp_id
        assert rule.description == exp_desc
        assert rule.weight == exp_weight
        assert callable(rule.predicate)

        # Frozen dataclass check
        with pytest.raises(dataclasses.FrozenInstanceError):
            rule.weight = 99.0  # type: ignore[misc]

    # Verify exported individual constants match tuple elements
    assert RULES[0] is R001
    assert RULES[1] is R002
    assert RULES[2] is R003
    assert RULES[3] is R004
    assert RULES[4] is R005
    assert RULES[5] is R006


def test_rule_r001_amount_boundaries_and_predicates() -> None:
    """14. R001 triggers strictly when amount > 50,000."""
    df = pd.DataFrame(
        {
            "amount": [50000.0, 50000.01, 100000.0, 49999.99, 0.0, -50.0, np.nan],
        }
    )
    res = R001.predicate(df)
    assert isinstance(res, np.ndarray)
    assert res.dtype == bool
    # 50000.0 is strictly not > 50000 -> False
    assert list(res) == [False, True, True, False, False, False, False]


def test_rule_r002_new_device_boundaries_and_default() -> None:
    """15. R002 triggers strictly when device_age_days < 1, with default 99."""
    # Explicit column present
    df = pd.DataFrame(
        {
            "amount": [100.0] * 6,
            "device_age_days": [0.0, 0.5, 0.99, 1.0, 10.0, np.nan],
        }
    )
    res = R002.predicate(df)
    # 0.0, 0.5, 0.99 are < 1 -> True; 1.0 is not < 1 -> False; nan -> default 99 -> False
    assert list(res) == [True, True, True, False, False, False]

    # Column missing entirely -> default 99 for all rows -> False
    df_missing = pd.DataFrame({"amount": [100.0, 200.0]})
    res_missing = R002.predicate(df_missing)
    assert list(res_missing) == [False, False]


def test_rule_r003_txn_count_1h_boundaries_and_default() -> None:
    """16. R003 triggers strictly when txn_count_1h > 5, with default 0."""
    # Explicit column present
    df = pd.DataFrame(
        {
            "amount": [100.0] * 6,
            "txn_count_1h": [0, 4, 5, 6, 10, np.nan],
        }
    )
    res = R003.predicate(df)
    # 6, 10 are > 5 -> True; 5 is not > 5 -> False; nan -> default 0 -> False
    assert list(res) == [False, False, False, True, True, False]

    # Column missing entirely -> default 0 -> False
    df_missing = pd.DataFrame({"amount": [100.0, 200.0]})
    res_missing = R003.predicate(df_missing)
    assert list(res_missing) == [False, False]


def test_rule_r004_amount_ratio_boundaries_and_default() -> None:
    """17. R004 triggers strictly when amount > 5 * max(amount_mean_30d, 1.0), with default 1.0."""
    # Case A: Normal average (100.0) -> threshold = 5 * 100 = 500.0
    df_a = pd.DataFrame(
        {
            "amount": [500.0, 500.01, 1000.0, 499.0],
            "amount_mean_30d": [100.0, 100.0, 100.0, 100.0],
        }
    )
    res_a = R004.predicate(df_a)
    assert list(res_a) == [False, True, True, False]

    # Case B: Low average (< 1.0) or zero/negative -> clamped to 1.0 -> threshold = 5.0
    df_b = pd.DataFrame(
        {
            "amount": [5.0, 5.01, 6.0, 4.0],
            "amount_mean_30d": [0.5, 0.0, -10.0, 0.1],
        }
    )
    res_b = R004.predicate(df_b)
    assert list(res_b) == [False, True, True, False]

    # Case C: Missing column entirely -> default 1.0 -> threshold = 5.0
    df_c = pd.DataFrame(
        {
            "amount": [5.0, 5.01, 6.0, 4.0],
        }
    )
    res_c = R004.predicate(df_c)
    assert list(res_c) == [False, True, True, False]

    # Case D: NaN in amount_mean_30d -> defaults to 1.0 -> threshold = 5.0
    df_d = pd.DataFrame(
        {
            "amount": [5.0, 6.0],
            "amount_mean_30d": [np.nan, np.nan],
        }
    )
    res_d = R004.predicate(df_d)
    assert list(res_d) == [False, True]


def test_rule_r005_impossible_travel_boundaries_and_default() -> None:
    """18. R005 triggers strictly when distinct_countries_1h > 1, with weight 2.0 and default 1."""
    assert R005.weight == 2.0

    # Explicit column present
    df = pd.DataFrame(
        {
            "amount": [100.0] * 5,
            "distinct_countries_1h": [0, 1, 2, 5, np.nan],
        }
    )
    res = R005.predicate(df)
    # 2, 5 are > 1 -> True; 1 is not > 1 -> False; nan -> default 1 -> False
    assert list(res) == [False, False, True, True, False]

    # Column missing entirely -> default 1 -> False
    df_missing = pd.DataFrame({"amount": [100.0, 200.0]})
    res_missing = R005.predicate(df_missing)
    assert list(res_missing) == [False, False]


def test_rule_r006_device_account_degree_boundaries_and_default() -> None:
    """19. R006 triggers strictly when device_account_degree > 3, with default 1."""
    # Explicit column present
    df = pd.DataFrame(
        {
            "amount": [100.0] * 6,
            "device_account_degree": [1, 2, 3, 4, 10, np.nan],
        }
    )
    res = R006.predicate(df)
    # 4, 10 are > 3 -> True; 3 is not > 3 -> False; nan -> default 1 -> False
    assert list(res) == [False, False, False, True, True, False]

    # Column missing entirely -> default 1 -> False
    df_missing = pd.DataFrame({"amount": [100.0, 200.0]})
    res_missing = R006.predicate(df_missing)
    assert list(res_missing) == [False, False]


def test_missing_required_amount_column_fails_closed() -> None:
    """20. Missing required 'amount' column raises KeyError across all functions and predicates requiring it."""
    df_no_amount = pd.DataFrame({"device_age_days": [0.5], "txn_count_1h": [10]})

    with pytest.raises(KeyError, match="amount"):
        hits(df_no_amount)

    with pytest.raises(KeyError, match="amount"):
        score(df_no_amount)

    with pytest.raises(KeyError, match="amount"):
        decisions(df_no_amount)

    with pytest.raises(KeyError, match="amount"):
        explain(df_no_amount)

    with pytest.raises(KeyError, match="amount"):
        R001.predicate(df_no_amount)

    with pytest.raises(KeyError, match="amount"):
        R004.predicate(df_no_amount)

    # Empty DataFrame with no columns
    with pytest.raises(KeyError, match="amount"):
        hits(pd.DataFrame())


def test_weighted_scoring_exact_math() -> None:
    """21. Exact weighted sum of fired rules, verifying weights (1.0 for R001-R004, R006; 2.0 for R005)."""
    df = pd.DataFrame(
        [
            # Row 0: Clean transaction (no rules fire) -> 0.0
            {
                "amount": 100.0,
                "device_age_days": 10.0,
                "txn_count_1h": 1,
                "amount_mean_30d": 100.0,
                "distinct_countries_1h": 1,
                "device_account_degree": 1,
            },
            # Row 1: R001 only (amount > 50_000) -> 1.0
            {
                "amount": 60000.0,
                "device_age_days": 10.0,
                "txn_count_1h": 1,
                "amount_mean_30d": 50000.0,
                "distinct_countries_1h": 1,
                "device_account_degree": 1,
            },
            # Row 2: R005 only (impossible travel) -> 2.0
            {
                "amount": 100.0,
                "device_age_days": 10.0,
                "txn_count_1h": 1,
                "amount_mean_30d": 100.0,
                "distinct_countries_1h": 2,
                "device_account_degree": 1,
            },
            # Row 3: R001 (1.0) + R005 (2.0) -> 3.0
            {
                "amount": 60000.0,
                "device_age_days": 10.0,
                "txn_count_1h": 1,
                "amount_mean_30d": 50000.0,
                "distinct_countries_1h": 3,
                "device_account_degree": 1,
            },
            # Row 4: R001 (1.0) + R002 (1.0) + R003 (1.0) + R004 (1.0) + R006 (1.0) (all except R005) -> 5.0
            {
                "amount": 60000.0,
                "device_age_days": 0.5,
                "txn_count_1h": 8,
                "amount_mean_30d": 1000.0,
                "distinct_countries_1h": 1,
                "device_account_degree": 4,
            },
            # Row 5: ALL 6 rules fire -> 1.0 + 1.0 + 1.0 + 1.0 + 2.0 + 1.0 = 7.0
            {
                "amount": 60000.0,
                "device_age_days": 0.2,
                "txn_count_1h": 9,
                "amount_mean_30d": 100.0,
                "distinct_countries_1h": 2,
                "device_account_degree": 5,
            },
        ]
    )

    s = score(df)
    assert isinstance(s, np.ndarray)
    assert s.dtype == np.float64
    np.testing.assert_array_almost_equal(s, [0.0, 1.0, 2.0, 3.0, 5.0, 7.0])


def test_decision_thresholds_at_varying_min_weights() -> None:
    """22. Decision function evaluates score(df) >= min_weight across multiple thresholds."""
    df = pd.DataFrame(
        [
            # Row 0: score 0.0
            {"amount": 100.0, "amount_mean_30d": 100.0},
            # Row 1: score 1.0 (R001 only)
            {"amount": 60000.0, "amount_mean_30d": 50000.0},
            # Row 2: score 2.0 (R005 only, weight 2.0)
            {"amount": 100.0, "amount_mean_30d": 100.0, "distinct_countries_1h": 2},
            # Row 3: score 3.0 (R001 + R005)
            {"amount": 60000.0, "amount_mean_30d": 50000.0, "distinct_countries_1h": 2},
            # Row 4: score 7.0 (all 6 rules)
            {
                "amount": 60000.0,
                "device_age_days": 0.1,
                "txn_count_1h": 10,
                "amount_mean_30d": 100.0,
                "distinct_countries_1h": 2,
                "device_account_degree": 4,
            },
        ]
    )

    # min_weight = 1.0 (default)
    d_1 = decisions(df, min_weight=1.0)
    assert d_1.dtype == bool
    assert list(d_1) == [False, True, True, True, True]

    # min_weight = 2.0 (row 1 with score 1.0 drops out; row 2 with score 2.0 stays)
    d_2 = decisions(df, min_weight=2.0)
    assert list(d_2) == [False, False, True, True, True]

    # min_weight = 3.0
    d_3 = decisions(df, min_weight=3.0)
    assert list(d_3) == [False, False, False, True, True]

    # min_weight = 7.0
    d_7 = decisions(df, min_weight=7.0)
    assert list(d_7) == [False, False, False, False, True]

    # min_weight = 7.5 (exceeds max possible score)
    d_8 = decisions(df, min_weight=7.5)
    assert list(d_8) == [False, False, False, False, False]


def test_explain_ordering_and_multi_rule_hits() -> None:
    """23. Explanations return ordered IDs of fired rules preserving R001-R006 sequence."""
    df = pd.DataFrame(
        [
            # Row 0: No rules
            {"amount": 100.0, "amount_mean_30d": 100.0},
            # Row 1: R001 only
            {"amount": 60000.0, "amount_mean_30d": 50000.0},
            # Row 2: R005 only
            {"amount": 100.0, "amount_mean_30d": 100.0, "distinct_countries_1h": 2},
            # Row 3: R001 and R005
            {"amount": 60000.0, "amount_mean_30d": 50000.0, "distinct_countries_1h": 2},
            # Row 4: R002 and R006 (skip R001, R003, R004, R005)
            {
                "amount": 100.0,
                "device_age_days": 0.5,
                "amount_mean_30d": 100.0,
                "device_account_degree": 5,
            },
            # Row 5: All 6 rules
            {
                "amount": 60000.0,
                "device_age_days": 0.1,
                "txn_count_1h": 10,
                "amount_mean_30d": 100.0,
                "distinct_countries_1h": 2,
                "device_account_degree": 4,
            },
        ]
    )

    exp = explain(df)
    assert isinstance(exp, list)
    assert len(exp) == 6
    assert exp[0] == []
    assert exp[1] == ["R001"]
    assert exp[2] == ["R005"]
    assert exp[3] == ["R001", "R005"]
    assert exp[4] == ["R002", "R006"]
    assert exp[5] == ["R001", "R002", "R003", "R004", "R005", "R006"]


def test_empty_dataframe_all_functions() -> None:
    """24. Empty DataFrame with required amount column returns valid empty structures."""
    df_empty = pd.DataFrame(columns=["amount", "device_age_days"])

    h = hits(df_empty)
    assert len(h) == 6
    for k, v in h.items():
        assert isinstance(v, np.ndarray)
        assert v.dtype == bool
        assert len(v) == 0

    s = score(df_empty)
    assert isinstance(s, np.ndarray)
    assert s.dtype == np.float64
    assert len(s) == 0

    d = decisions(df_empty)
    assert isinstance(d, np.ndarray)
    assert d.dtype == bool
    assert len(d) == 0

    e = explain(df_empty)
    assert e == []


def test_input_dataframe_immutability() -> None:
    """25. Input DataFrames are strictly not mutated by any baseline function."""
    df = pd.DataFrame(
        {
            "amount": [100.0, 60000.0],
            "device_age_days": [0.5, np.nan],
            "amount_mean_30d": [50.0, 100.0],
            "distinct_countries_1h": [1, 2],
        },
        index=[10, 20],
    )
    df_before = df.copy(deep=True)

    _ = hits(df)
    _ = score(df)
    _ = decisions(df, min_weight=1.5)
    _ = explain(df)

    pd.testing.assert_frame_equal(df, df_before)


def test_non_monotonic_and_duplicate_index_preservation() -> None:
    """26. Arbitrary non-monotonic and duplicate index labels are handled positionally without alignment errors."""
    df = pd.DataFrame(
        {
            "amount": [100.0, 60000.0, 100.0, 70000.0, 100.0, 80000.0],
            "amount_mean_30d": [100.0] * 6,
        },
        index=[100, 20, 100, 5, 50, 20],
    )

    h = hits(df)
    assert len(h["R001"]) == 6
    assert list(h["R001"]) == [False, True, False, True, False, True]
    assert list(h["R004"]) == [False, True, False, True, False, True]

    s = score(df)
    assert len(s) == 6
    np.testing.assert_array_equal(s, [0.0, 2.0, 0.0, 2.0, 0.0, 2.0])

    d = decisions(df, min_weight=2.0)
    assert len(d) == 6
    assert list(d) == [False, True, False, True, False, True]

    e = explain(df)
    assert len(e) == 6
    assert e[0] == []
    assert e[1] == ["R001", "R004"]
    assert e[2] == []
    assert e[3] == ["R001", "R004"]
    assert e[4] == []
    assert e[5] == ["R001", "R004"]


def test_type_and_argument_validation() -> None:
    """27. Non-DataFrame inputs raise TypeError across public baseline functions."""
    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        hits("not_a_df")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        score([{"amount": 100.0}])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        decisions(None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        explain({"amount": 100.0})  # type: ignore[arg-type]
