"""Unit tests for imbalance-aware metrics, cost curves, and cohort slices.

Milestone 3 - Step 3.6.
Conforms to AegisFlow_M3_Implementation_Guide.pdf pages 10-11, 14-15.
"""
from __future__ import annotations

import numpy as np
import pytest

from aegis_ml.evaluation.metrics import (
    amount_bands,
    best_threshold,
    cohort_slices,
    cost_curve,
    lift_over,
    pr_auc,
    recall_at_fpr,
)

RNG = np.random.default_rng(7)


def sample(n: int = 4000, rate: float = 0.03, sep: float = 2.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic evaluation dataset matching guide fixture (page 14)."""
    y = (RNG.random(n) < rate).astype(int)
    s = RNG.normal(0, 1, n) + sep * y
    amt = np.exp(RNG.normal(4, 1, n))
    return y, s, amt


# ============================================================================
# 1. recall_at_fpr tests
# ============================================================================


def test_recall_at_fpr_respects_the_budget() -> None:
    """Guide test (page 14): operating point respects max_fpr and expands with budget."""
    y, s, _ = sample()
    r = recall_at_fpr(y, s, 0.01)
    assert r["fpr"] <= 0.01 + 1e-9
    assert 0.0 < r["recall"] <= 1.0
    assert r["max_fpr"] == 0.01
    assert np.isfinite(r["threshold"])

    loose = recall_at_fpr(y, s, 0.10)
    assert loose["fpr"] <= 0.10 + 1e-9
    assert loose["recall"] >= r["recall"]  # a bigger budget never catches less


def test_recall_at_fpr_handles_a_degenerate_label_vector() -> None:
    """Guide test (page 14): single-class label vector returns recall=0, threshold=inf."""
    # All zeros
    r0 = recall_at_fpr([0, 0, 0], [0.1, 0.2, 0.3], 0.01)
    assert r0["recall"] == 0.0
    assert r0["fpr"] == 0.0
    assert r0["threshold"] == float("inf")
    assert r0["max_fpr"] == 0.01

    # All ones
    r1 = recall_at_fpr([1, 1, 1], [0.1, 0.2, 0.3], 0.01)
    assert r1["recall"] == 0.0
    assert r1["fpr"] == 0.0
    assert r1["threshold"] == float("inf")

    # Empty inputs
    r_empty = recall_at_fpr([], [], 0.01)
    assert r_empty["recall"] == 0.0
    assert r_empty["fpr"] == 0.0
    assert r_empty["threshold"] == float("inf")


def test_recall_at_fpr_hand_crafted_expected_values() -> None:
    """Verifies recall@FPR on hand-computed small dataset with known threshold steps."""
    # 10 negatives, 10 positives
    # Negatives scores: 0.10 .. 0.95
    # Positives scores: 0.55 .. 0.98
    y_true = np.array([0] * 10 + [1] * 10)
    scores = np.array(
        [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
        + [0.55, 0.65, 0.75, 0.85, 0.91, 0.92, 0.93, 0.96, 0.97, 0.98]
    )

    # Budget max_fpr = 0.0: No false positives allowed (fpr <= 0.0)
    # Highest negative is 0.95.
    # Scores > 0.95 are: 0.96, 0.97, 0.98 (3 positives, 0 negatives).
    # Expected recall: 3 / 10 = 0.30.
    r_zero = recall_at_fpr(y_true, scores, max_fpr=0.0)
    assert r_zero["fpr"] == 0.0
    assert r_zero["recall"] == pytest.approx(0.30)
    assert r_zero["threshold"] > 0.95

    # Budget max_fpr = 0.10: 1 false positive allowed out of 10 negatives (FPR <= 0.10)
    # 2nd highest negative is 0.90. Negative 0.95 can be flagged.
    # Positives with score >= 0.91: 6 positives (0.91, 0.92, 0.93, 0.96, 0.97, 0.98).
    # Recall at threshold 0.91: 6 / 10 = 0.60, FPR = 1/10 = 0.10.
    r_10 = recall_at_fpr(y_true, scores, max_fpr=0.10)
    assert r_10["fpr"] <= 0.10 + 1e-9
    assert r_10["recall"] == pytest.approx(0.60)


def test_recall_at_fpr_severe_class_imbalance() -> None:
    """Verifies operating point on severe class imbalance (0.1% fraud rate: 10 in 10,000)."""
    n = 10000
    y = np.zeros(n, dtype=int)
    y[:10] = 1  # 10 fraud transactions
    s = RNG.normal(0, 1, n)
    s[:10] += 3.0  # frauds pushed higher

    r = recall_at_fpr(y, s, max_fpr=0.01)
    assert r["fpr"] <= 0.01 + 1e-9
    assert 0.0 <= r["recall"] <= 1.0
    assert np.isfinite(r["threshold"])


def test_recall_at_fpr_invalid_inputs_rejected() -> None:
    """Rejects invalid max_fpr, length mismatches, non-finite max_fpr, and non-finite scores."""
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        recall_at_fpr([0, 1], [0.1, 0.9], max_fpr=-0.05)

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        recall_at_fpr([0, 1], [0.1, 0.9], max_fpr=1.05)

    # Non-finite max_fpr
    with pytest.raises(ValueError, match="finite float"):
        recall_at_fpr([0, 1], [0.1, 0.9], max_fpr=float("nan"))

    with pytest.raises(ValueError, match="finite float"):
        recall_at_fpr([0, 1], [0.1, 0.9], max_fpr=float("inf"))

    with pytest.raises(ValueError, match="Length mismatch"):
        recall_at_fpr([0, 1, 0], [0.1, 0.9])

    with pytest.raises(ValueError, match="finite numeric values"):
        recall_at_fpr([0, 1], [0.1, np.nan])

    with pytest.raises(ValueError, match="finite numeric values"):
        recall_at_fpr([0, 1], [0.1, np.inf])

    with pytest.raises(ValueError, match="1D array"):
        recall_at_fpr(np.array([[0, 1], [1, 0]]), [0.1, 0.2, 0.3, 0.4])

    with pytest.raises(ValueError, match="1D array"):
        recall_at_fpr([0, 1], np.array([[0.1], [0.9]]))


# ============================================================================
# 2. pr_auc & lift_over tests
# ============================================================================


def test_pr_auc_and_lift() -> None:
    """Guide test (page 15): PR-AUC beats random ranking; lift calculation."""
    y, s, _ = sample()
    assert pr_auc(y, s) > pr_auc(y, RNG.normal(0, 1, len(y)))
    assert lift_over(0.40, 0.52) == pytest.approx(0.30)
    assert lift_over(0.0, 0.5) == 0.0


def test_pr_auc_exact_values_and_degenerate_cases() -> None:
    """PR-AUC returns 1.0 on perfect separation and 0.0 on degenerate/empty inputs."""
    # Perfect model
    y_perf = [0, 0, 1, 1]
    s_perf = [0.1, 0.2, 0.8, 0.9]
    assert pr_auc(y_perf, s_perf) == pytest.approx(1.0)

    # Degenerate single-class
    assert pr_auc([0, 0, 0], [0.1, 0.2, 0.3]) == 0.0
    assert pr_auc([1, 1, 1], [0.1, 0.2, 0.3]) == 0.0
    assert pr_auc([], []) == 0.0

    # Validation errors
    with pytest.raises(ValueError, match="Length mismatch"):
        pr_auc([0, 1], [0.1])
    with pytest.raises(ValueError, match="finite"):
        pr_auc([0, 1], [0.1, np.nan])


def test_lift_over_variations() -> None:
    """Lift calculation handles negative lift, negative baselines, and NaNs."""
    # 20% drop
    assert lift_over(0.50, 0.40) == pytest.approx(-0.20)
    # Zero change
    assert lift_over(0.50, 0.50) == 0.0
    # Negative baseline
    assert lift_over(-0.1, 0.5) == 0.0
    # NaNs
    assert lift_over(np.nan, 0.5) == 0.0
    assert lift_over(0.4, np.nan) == 0.0


# ============================================================================
# 3. cost_curve & best_threshold tests
# ============================================================================


def test_cost_curve_optimum_is_not_the_midpoint() -> None:
    """Guide test (page 15): cost curve produces optimum; cheaper review shifts threshold."""
    y, s, amt = sample()
    curve = cost_curve(y, s, amt, review_cost=3.0, friction_cost=12.0)
    best = best_threshold(curve)
    assert best["total_cost"] == min(r["total_cost"] for r in curve)
    assert 0.0 <= best["flag_rate"] <= 1.0

    # a cheaper review makes it worth flagging more
    cheap = best_threshold(cost_curve(y, s, amt, review_cost=0.1, friction_cost=1.0))
    assert cheap["flag_rate"] >= best["flag_rate"]


def test_cost_curve_exact_hand_calculation() -> None:
    """Hand-calculated cost verification across 3 explicit costs:
    - missed fraud value (chargeback amount)
    - friction cost (declining good customer: friction_cost * FP)
    - review cost (analyst time: review_cost * Flagged)
    """
    y = np.array([0, 0, 1, 1])
    s = np.array([0.2, 0.7, 0.4, 0.9])
    amt = np.array([100.0, 200.0, 500.0, 1000.0])
    review_cost = 10.0
    friction_cost = 50.0

    # Grid of exactly 3 thresholds: [0.2, 0.55, 0.9]
    curve = cost_curve(y, s, amt, review_cost=review_cost, friction_cost=friction_cost, grid=3)
    assert len(curve) == 3

    # Point 0: threshold = 0.2 (All 4 flagged: flag=[True, True, True, True])
    # - missed fraud: 0.0
    # - friction: 2 good customers * 50.0 = 100.0
    # - review: 4 flagged * 10.0 = 40.0
    # - total: 140.0, flag_rate: 1.0
    assert curve[0]["threshold"] == pytest.approx(0.2)
    assert curve[0]["missed_fraud_value"] == pytest.approx(0.0)
    assert curve[0]["friction_cost"] == pytest.approx(100.0)
    assert curve[0]["review_cost"] == pytest.approx(40.0)
    assert curve[0]["total_cost"] == pytest.approx(140.0)
    assert curve[0]["flag_rate"] == pytest.approx(1.0)

    # Point 1: threshold = 0.55 (scores >= 0.55: Tx 1 [0.7] and Tx 3 [0.9] flagged)
    # - flag = [False, True, False, True]
    # - missed fraud: Tx 2 (fraud, amt=500.0) -> missed = 500.0
    # - friction: Tx 1 (legit, flagged) -> 1 * 50.0 = 50.0
    # - review: 2 flagged * 10.0 = 20.0
    # - total: 570.0, flag_rate: 0.5
    assert curve[1]["threshold"] == pytest.approx(0.55)
    assert curve[1]["missed_fraud_value"] == pytest.approx(500.0)
    assert curve[1]["friction_cost"] == pytest.approx(50.0)
    assert curve[1]["review_cost"] == pytest.approx(20.0)
    assert curve[1]["total_cost"] == pytest.approx(570.0)
    assert curve[1]["flag_rate"] == pytest.approx(0.5)

    # Point 2: threshold = 0.9 (score >= 0.9: only Tx 3 [0.9] flagged)
    # - flag = [False, False, False, True]
    # - missed fraud: Tx 2 (fraud, amt=500.0) -> missed = 500.0
    # - friction: 0 legit flagged -> 0.0
    # - review: 1 flagged * 10.0 = 10.0
    # - total: 510.0, flag_rate: 0.25
    assert curve[2]["threshold"] == pytest.approx(0.9)
    assert curve[2]["missed_fraud_value"] == pytest.approx(500.0)
    assert curve[2]["friction_cost"] == pytest.approx(0.0)
    assert curve[2]["review_cost"] == pytest.approx(10.0)
    assert curve[2]["total_cost"] == pytest.approx(510.0)
    assert curve[2]["flag_rate"] == pytest.approx(0.25)

    # best_threshold selects point 0 (total_cost 140.0 < 510.0 < 570.0)
    best = best_threshold(curve)
    assert best["threshold"] == pytest.approx(0.2)
    assert best["total_cost"] == pytest.approx(140.0)


def test_cost_curve_empty_and_invalid_inputs() -> None:
    """Validates cost_curve and best_threshold error handling."""
    assert cost_curve([], [], []) == []

    with pytest.raises(ValueError, match="must not be empty"):
        best_threshold([])

    with pytest.raises(ValueError, match="Length mismatch"):
        cost_curve([0, 1], [0.1, 0.9], [100.0])

    with pytest.raises(ValueError, match="non-negative"):
        cost_curve([0, 1], [0.1, 0.9], [100.0, 200.0], review_cost=-1.0)

    with pytest.raises(ValueError, match="non-negative"):
        cost_curve([0, 1], [0.1, 0.9], [100.0, 200.0], friction_cost=-5.0)

    with pytest.raises(ValueError, match="positive"):
        cost_curve([0, 1], [0.1, 0.9], [100.0, 200.0], grid=0)

    with pytest.raises(ValueError, match="finite"):
        cost_curve([0, 1], [np.nan, 0.9], [100.0, 200.0])

    with pytest.raises(ValueError, match="finite"):
        cost_curve([0, 1], [0.1, 0.9], [100.0, np.inf])


# ============================================================================
# 4. cohort_slices & amount_bands tests
# ============================================================================


def test_cohort_slices_flag_thin_cohorts_instead_of_inventing_numbers() -> None:
    """Guide test (page 15): partitions amounts and flags cohorts with < 50 rows or no fraud."""
    y, s, amt = sample()
    out = cohort_slices(y, s, amount_bands(amt), 0.01)
    assert set(out) == {"amount_0_50", "amount_50_250", "amount_250_1000", "amount_1000_inf"}

    tiny = cohort_slices(
        y, s, {"tiny": np.r_[np.ones(10, bool), np.zeros(len(y) - 10, bool)]}
    )
    assert tiny["tiny"]["recall"] is None and "too few" in tiny["tiny"]["note"]
    assert tiny["tiny"]["n"] == 10


def test_cohort_slices_no_fraud_cohort() -> None:
    """Cohort with >= 50 rows but zero fraud instances flags note and returns recall=None."""
    n = 200
    y = np.zeros(n, dtype=int)  # zero fraud
    s = RNG.normal(0, 1, n)
    cohorts = {"clean_group": np.ones(n, dtype=bool)}

    out = cohort_slices(y, s, cohorts, max_fpr=0.01)
    assert out["clean_group"]["n"] == 200
    assert out["clean_group"]["recall"] is None
    assert "no fraud" in out["clean_group"]["note"]


def test_cohort_slices_sufficient_cohort_metrics() -> None:
    """Cohort with >= 50 rows and >= 1 fraud returns complete recall@FPR dictionary."""
    n = 200
    y = np.zeros(n, dtype=int)
    y[:10] = 1  # 10 frauds
    s = RNG.normal(0, 1, n)
    s[:10] += 3.0
    cohorts = {"valid_cohort": np.ones(n, dtype=bool)}

    out = cohort_slices(y, s, cohorts, max_fpr=0.01)
    entry = out["valid_cohort"]
    assert entry["n"] == 200
    assert 0.0 <= entry["recall"] <= 1.0
    assert entry["fpr"] <= 0.01 + 1e-9
    assert np.isfinite(entry["threshold"])
    assert entry["max_fpr"] == 0.01


def test_amount_bands_partition_coverage() -> None:
    """amount_bands partitions all amounts disjointly and exhaustively."""
    test_amounts = [0.0, 25.0, 50.0, 120.0, 250.0, 600.0, 1000.0, 50000.0]
    bands = amount_bands(test_amounts)

    expected_keys = ["amount_0_50", "amount_50_250", "amount_250_1000", "amount_1000_inf"]
    assert list(bands.keys()) == expected_keys

    # Check partition property: each item belongs to exactly ONE band
    stacked = np.column_stack([bands[k] for k in expected_keys])
    assert np.all(stacked.sum(axis=1) == 1)

    # 50.0 falls in amount_50_250, not amount_0_50 (half-open [lo, hi))
    assert bands["amount_0_50"][2] is np.False_
    assert bands["amount_50_250"][2] is np.True_


def test_input_immutability() -> None:
    """All metrics functions leave input arrays and dicts unchanged."""
    y = np.array([0, 1, 0, 1])
    s = np.array([0.2, 0.8, 0.4, 0.9])
    amt = np.array([50.0, 100.0, 200.0, 500.0])
    cohorts = {"c1": np.array([True, True, False, False])}

    y_before = y.copy()
    s_before = s.copy()
    amt_before = amt.copy()
    cohort_before = cohorts["c1"].copy()

    _ = recall_at_fpr(y, s, 0.01)
    _ = pr_auc(y, s)
    _ = lift_over(0.3, 0.5)
    _ = cost_curve(y, s, amt)
    _ = cohort_slices(y, s, cohorts)
    _ = amount_bands(amt)

    np.testing.assert_array_equal(y, y_before)
    np.testing.assert_array_equal(s, s_before)
    np.testing.assert_array_equal(amt, amt_before)
    np.testing.assert_array_equal(cohorts["c1"], cohort_before)


def test_binary_label_validation_across_all_metrics() -> None:
    """Explicitly validates binary labels across all metrics functions, rejecting non-binary labels."""
    invalid_labels = [
        [0.5, 1.0],  # Non-binary float
        [2, 1],      # Out of bounds integer
        [-1, 1],     # Negative label
        ["0", "1"],  # String labels
        ["fraud", "clean"],  # String labels
        [np.nan, 1.0],  # NaN label
    ]

    for inv in invalid_labels:
        with pytest.raises(ValueError, match="binary labels"):
            recall_at_fpr(inv, [0.1, 0.9])

        with pytest.raises(ValueError, match="binary labels"):
            pr_auc(inv, [0.1, 0.9])

        with pytest.raises(ValueError, match="binary labels"):
            cost_curve(inv, [0.1, 0.9], [100.0, 200.0])

        with pytest.raises(ValueError, match="binary labels"):
            cohort_slices(inv, [0.1, 0.9], {"c": [True, True]})


def test_cohort_slices_mask_and_type_validation() -> None:
    """cohort_slices rejects invalid cohort masks, non-dict inputs, and 2D arrays."""
    with pytest.raises(TypeError, match="cohorts must be a dict"):
        cohort_slices([0, 1], [0.1, 0.9], "not_a_dict")  # type: ignore[arg-type]

    # Non-finite max_fpr
    with pytest.raises(ValueError, match="finite float"):
        cohort_slices([0, 1], [0.1, 0.9], {"c": [True, True]}, max_fpr=float("nan"))

    # Mask length mismatch
    with pytest.raises(ValueError, match="does not match"):
        cohort_slices([0, 1], [0.1, 0.9], {"c": [True, True, False]})

    # 2D mask
    with pytest.raises(ValueError, match="must be 1D"):
        cohort_slices([0, 1], [0.1, 0.9], {"c": np.array([[True], [False]])})

    # Non-boolean/binary mask elements
    with pytest.raises(ValueError, match="boolean or binary"):
        cohort_slices([0, 1], [0.1, 0.9], {"c": [2, 0]})


def test_amount_bands_validation() -> None:
    """amount_bands rejects non-monotonic edges, fewer than 2 edges, and 2D amounts."""
    with pytest.raises(ValueError, match="at least two boundary values"):
        amount_bands([100.0], edges=(0,))

    with pytest.raises(ValueError, match="strictly monotonically increasing"):
        amount_bands([100.0], edges=(100, 50, 250))

    with pytest.raises(ValueError, match="must be a 1D array"):
        amount_bands(np.array([[50.0], [100.0]]))
