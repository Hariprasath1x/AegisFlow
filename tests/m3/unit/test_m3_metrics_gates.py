"""Metrics, calibration, fairness and the champion/challenger rule.

Implements the test suite from pages 14-15 of AegisFlow_M3_Implementation_Guide.pdf
and comprehensive boundary and validation tests for fairness and gates.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from aegis_ml.evaluation.fairness import (
    FOUR_FIFTHS,
    disparate_impact,
    group_rates,
)
from aegis_ml.evaluation.gates import (
    GATES,
    challenger_wins,
    check_gate,
    is_better,
)
from aegis_ml.evaluation.metrics import (
    amount_bands,
    best_threshold,
    cohort_slices,
    cost_curve,
    lift_over,
    pr_auc,
    recall_at_fpr,
)
from aegis_ml.models.calibration import (
    Calibrator,
    expected_calibration_error,
    reliability,
)


RNG = np.random.default_rng(7)


def sample(n=4000, rate=0.03, sep=2.5):
    """Generate synthetic evaluation samples."""
    y = (RNG.random(n) < rate).astype(int)
    s = RNG.normal(0, 1, n) + sep * y
    amt = np.exp(RNG.normal(4, 1, n))
    return y, s, amt


# --- Guide Test Suite (Pages 14-15) ---


def test_recall_at_fpr_respects_the_budget():
    y, s, _ = sample()
    r = recall_at_fpr(y, s, 0.01)
    assert r["fpr"] <= 0.01 + 1e-9
    assert 0.0 < r["recall"] <= 1.0
    loose = recall_at_fpr(y, s, 0.10)
    assert loose["recall"] >= r["recall"]  # a bigger budget never catches less


def test_recall_at_fpr_handles_a_degenerate_label_vector():
    r = recall_at_fpr([0, 0, 0], [0.1, 0.2, 0.3], 0.01)
    assert r["recall"] == 0.0 and r["threshold"] == float("inf")


def test_pr_auc_and_lift():
    y, s, _ = sample()
    assert pr_auc(y, s) > pr_auc(y, RNG.normal(0, 1, len(y)))
    assert lift_over(0.40, 0.52) == pytest.approx(0.30)
    assert lift_over(0.0, 0.5) == 0.0


def test_cost_curve_optimum_is_not_the_midpoint():
    y, s, amt = sample()
    curve = cost_curve(y, s, amt, review_cost=3.0, friction_cost=12.0)
    best = best_threshold(curve)
    assert best["total_cost"] == min(r["total_cost"] for r in curve)
    assert 0.0 <= best["flag_rate"] <= 1.0
    # a cheaper review makes it worth flagging more
    cheap = best_threshold(cost_curve(y, s, amt, review_cost=0.1, friction_cost=1.0))
    assert cheap["flag_rate"] >= best["flag_rate"]


def test_cohort_slices_flag_thin_cohorts_instead_of_inventing_numbers():
    y, s, amt = sample()
    out = cohort_slices(y, s, amount_bands(amt), 0.01)
    assert set(out) == {
        "amount_0_50",
        "amount_50_250",
        "amount_250_1000",
        "amount_1000_inf",
    }
    tiny = cohort_slices(
        y, s, {"tiny": np.r_[np.ones(10, bool), np.zeros(len(y) - 10, bool)]}
    )
    assert tiny["tiny"]["recall"] is None and "too few" in tiny["tiny"]["note"]


def test_isotonic_calibration_fixes_a_squashed_score():
    y, s, _ = sample(n=8000, rate=0.10)
    probs_raw = 1 / (1 + np.exp(-(s * 0.25)))  # deliberately under-confident
    cal = Calibrator().fit(probs_raw[:4000], y[:4000])
    probs_cal = cal.transform(probs_raw[4000:])
    assert expected_calibration_error(y[4000:], probs_cal) < expected_calibration_error(
        y[4000:], probs_raw[4000:]
    )
    rows = reliability(y[4000:], probs_cal)
    assert rows and all(0.0 <= r["observed"] <= 1.0 for r in rows)
    with pytest.raises(RuntimeError):
        Calibrator().transform([0.5])


def test_disparate_impact_uses_the_four_fifths_rule():
    y = np.r_[np.zeros(500, int), np.zeros(500, int)]
    flags = np.r_[
        np.ones(100, bool),
        np.zeros(400, bool),
        np.ones(400, bool),
        np.zeros(100, bool),
    ]
    groups = {
        "a": np.r_[np.ones(500, bool), np.zeros(500, bool)],
        "b": np.r_[np.zeros(500, bool), np.ones(500, bool)],
    }
    rates = group_rates(y, flags, groups)
    di = disparate_impact(rates)
    assert di["ratio"] == pytest.approx(0.25) and di["flag"] is True
    assert di["lowest"] == "a" and di["highest"] == "b"
    assert disparate_impact({"a": {"flag_rate": 0.1}})["ratio"] is None


def test_gates_and_the_noise_band():
    assert check_gate("ece", 0.04)[0] and not check_gate("ece", 0.06)[0]
    assert check_gate("lift_over_rules", 0.30)[0]
    assert not is_better("ece", 0.05, 0.05)
    champ = {"recall_at_1pct_fpr": 0.52}
    ok, why = challenger_wins(
        {
            "recall_at_1pct_fpr": 0.60,
            "lift_over_rules": 0.35,
            "pr_auc": 0.71,
            "ece": 0.03,
        },
        champ,
    )
    assert ok and "promote" in why
    tiny, why2 = challenger_wins(
        {
            "recall_at_1pct_fpr": 0.525,
            "lift_over_rules": 0.35,
            "pr_auc": 0.71,
            "ece": 0.03,
        },
        champ,
    )
    assert not tiny and "noise band" in why2
    bad, why3 = challenger_wins(
        {
            "recall_at_1pct_fpr": 0.60,
            "lift_over_rules": 0.35,
            "pr_auc": 0.71,
            "ece": 0.20,
        },
        champ,
    )
    assert not bad and "ece" in why3


# --- Fairness: In-Depth Boundary & Edge-Case Tests ---


def test_fairness_cohort_size_thresholds():
    """Cohorts with < 50 items return too small note, >= 50 are screened."""
    y = np.zeros(100, dtype=int)
    flags = np.zeros(100, dtype=bool)
    flags[:10] = True

    # 49 items -> too small
    mask_49 = np.r_[np.ones(49, bool), np.zeros(51, bool)]
    # 50 items -> screenable
    mask_50 = np.r_[np.ones(50, bool), np.zeros(50, bool)]

    rates = group_rates(y, flags, {"c49": mask_49, "c50": mask_50})
    assert rates["c49"]["n"] == 49
    assert rates["c49"]["note"] == "cohort too small to screen"
    assert "flag_rate" not in rates["c49"]

    assert rates["c50"]["n"] == 50
    assert "note" not in rates["c50"]
    assert rates["c50"]["flag_rate"] == 0.20


def test_fairness_no_legitimate_transactions_fpr_none():
    """If a cohort consists 100% of fraud, false_positive_rate is None."""
    y = np.ones(60, dtype=int)  # 100% fraud
    flags = np.ones(60, dtype=bool)
    mask = np.ones(60, dtype=bool)

    rates = group_rates(y, flags, {"all_fraud": mask})
    assert rates["all_fraud"]["false_positive_rate"] is None
    assert rates["all_fraud"]["fraud_rate"] == 1.0
    assert rates["all_fraud"]["flag_rate"] == 1.0


def test_disparate_impact_exact_four_fifths_boundary():
    """Ratio exactly 0.80 does NOT trigger the flag; strictly < 0.80 does."""
    # Ratio = 0.40 / 0.50 = 0.80 (exactly 4/5)
    rates_equal = {
        "c1": {"flag_rate": 0.40},
        "c2": {"flag_rate": 0.50},
    }
    di_equal = disparate_impact(rates_equal)
    assert di_equal["ratio"] == pytest.approx(0.80)
    assert di_equal["flag"] is False

    # Ratio = 0.399 / 0.50 = 0.798 (< 4/5)
    rates_below = {
        "c1": {"flag_rate": 0.399},
        "c2": {"flag_rate": 0.50},
    }
    di_below = disparate_impact(rates_below)
    assert di_below["ratio"] == pytest.approx(0.798)
    assert di_below["flag"] is True


def test_disparate_impact_zero_highest_rate():
    """If highest rate is 0.0, ratio is None and flag is False."""
    rates = {
        "c1": {"flag_rate": 0.0},
        "c2": {"flag_rate": 0.0},
    }
    di = disparate_impact(rates)
    assert di["ratio"] is None
    assert di["flag"] is False
    assert di["lowest"] == "c1" or di["lowest"] == "c2"


def test_disparate_impact_fewer_than_two_screenable_cohorts():
    """Returns insufficient-cohort note when < 2 screenable cohorts."""
    rates_empty = {}
    assert disparate_impact(rates_empty) == {
        "ratio": None,
        "note": "need at least two screenable cohorts",
    }

    rates_one = {
        "c1": {"flag_rate": 0.15},
        "c2": {"note": "cohort too small to screen"},
    }
    assert disparate_impact(rates_one) == {
        "ratio": None,
        "note": "need at least two screenable cohorts",
    }


def test_disparate_impact_alternate_key():
    """Supports alternate keys such as false_positive_rate or fraud_rate."""
    rates = {
        "c1": {"flag_rate": 0.5, "false_positive_rate": 0.10},
        "c2": {"flag_rate": 0.5, "false_positive_rate": 0.20},
    }
    di = disparate_impact(rates, key="false_positive_rate")
    assert di["ratio"] == pytest.approx(0.50)
    assert di["lowest"] == "c1"
    assert di["highest"] == "c2"
    assert di["flag"] is True


def test_fairness_input_validation():
    """group_rates and disparate_impact reject malformed shapes and types."""
    y = np.zeros(60, dtype=int)
    f = np.zeros(60, dtype=bool)

    with pytest.raises(ValueError, match="groups must be a dictionary"):
        group_rates(y, f, groups=[])

    with pytest.raises(ValueError, match="y_true length"):
        group_rates(np.zeros(50, dtype=int), f, groups={"g": np.ones(50, bool)})

    with pytest.raises(ValueError, match="binary labels"):
        group_rates([0.5] * 60, f, groups={"g": np.ones(60, bool)})

    with pytest.raises(ValueError, match="mask for group 'g' length"):
        group_rates(y, f, groups={"g": np.ones(10, bool)})

    with pytest.raises(ValueError, match="rates must be a dictionary"):
        disparate_impact("not_a_dict")

    with pytest.raises(ValueError, match="key must be a string"):
        disparate_impact({"a": {"flag_rate": 0.1}}, key=123)


def test_fairness_input_immutability():
    """Inputs to group_rates and disparate_impact are not modified."""
    y = np.array([0, 1] * 30)
    f = np.array([0, 0] * 30)
    y_copy = y.copy()
    f_copy = f.copy()
    groups = {"g": np.ones(60, bool)}

    _ = group_rates(y, f, groups)
    np.testing.assert_array_equal(y, y_copy)
    np.testing.assert_array_equal(f, f_copy)


# --- Gates: In-Depth Boundary & Edge-Case Tests ---


def test_all_exact_gates_threshold_boundaries():
    """Verify each exact gate boundary in GATES for equality, pass, and fail."""
    # 1. recall_at_1pct_fpr >= 0.50
    assert check_gate("recall_at_1pct_fpr", 0.5000)[0] is True
    assert check_gate("recall_at_1pct_fpr", 0.5001)[0] is True
    assert check_gate("recall_at_1pct_fpr", 0.4999)[0] is False

    # 2. lift_over_rules >= 0.30
    assert check_gate("lift_over_rules", 0.3000)[0] is True
    assert check_gate("lift_over_rules", 0.3001)[0] is True
    assert check_gate("lift_over_rules", 0.2999)[0] is False

    # 3. pr_auc >= 0.60
    assert check_gate("pr_auc", 0.6000)[0] is True
    assert check_gate("pr_auc", 0.6001)[0] is True
    assert check_gate("pr_auc", 0.5999)[0] is False

    # 4. ece <= 0.05 (lower is better)
    assert check_gate("ece", 0.0500)[0] is True
    assert check_gate("ece", 0.0499)[0] is True
    assert check_gate("ece", 0.0501)[0] is False

    # 5. typology_detection >= 0.80
    assert check_gate("typology_detection", 0.8000)[0] is True
    assert check_gate("typology_detection", 0.8001)[0] is True
    assert check_gate("typology_detection", 0.7999)[0] is False


def test_check_gate_unknown_metric():
    """Unknown metrics pass check_gate with '(no gate defined)'."""
    ok, msg = check_gate("custom_unregistered_metric", 0.42)
    assert ok is True
    assert "no gate defined" in msg


def test_is_better_strictness_and_ties():
    """Ties are not an improvement. Lower is better only for ECE."""
    # Higher is better
    assert is_better("pr_auc", 0.75, 0.70) is True
    assert is_better("pr_auc", 0.70, 0.75) is False
    assert is_better("pr_auc", 0.70, 0.70) is False  # tie is churn

    # Lower is better (ece)
    assert is_better("ece", 0.03, 0.04) is True
    assert is_better("ece", 0.04, 0.03) is False
    assert is_better("ece", 0.04, 0.04) is False  # tie is churn

    # Unknown metrics default to higher is better
    assert is_better("unknown", 10.0, 5.0) is True
    assert is_better("unknown", 5.0, 10.0) is False
    assert is_better("unknown", 5.0, 5.0) is False


def test_challenger_wins_exact_relative_lift_boundary():
    """Relative lift exactly equal to min_relative_lift passes; strictly less fails."""
    champ = {"recall_at_1pct_fpr": 0.50}

    # Exactly 2% lift: 0.50 * 1.02 = 0.5100 -> lift = 0.01 / 0.50 = 0.02
    chal_exact = {"recall_at_1pct_fpr": 0.5100}
    ok_exact, why_exact = challenger_wins(chal_exact, champ, min_relative_lift=0.02)
    assert ok_exact is True
    assert "promote" in why_exact

    # Just below 2% lift: 0.5099 -> lift = 0.0099 / 0.50 = 0.0198 < 0.02
    chal_below = {"recall_at_1pct_fpr": 0.5099}
    ok_below, why_below = challenger_wins(chal_below, champ, min_relative_lift=0.02)
    assert ok_below is False
    assert "noise band" in why_below


def test_challenger_wins_missing_primary_metric():
    """Missing primary metric in challenger or champion returns explanatory failure."""
    champ = {"recall_at_1pct_fpr": 0.52}
    chal_no_primary = {"pr_auc": 0.75}
    ok1, msg1 = challenger_wins(chal_no_primary, champ)
    assert ok1 is False
    assert "recall_at_1pct_fpr missing from one of the models" in msg1

    champ_no_primary = {"pr_auc": 0.70}
    chal_with_primary = {"recall_at_1pct_fpr": 0.60}
    ok2, msg2 = challenger_wins(chal_with_primary, champ_no_primary)
    assert ok2 is False
    assert "recall_at_1pct_fpr missing from one of the models" in msg2


def test_gates_validation_on_non_finite_and_invalid_inputs():
    """Gates functions reject NaN, Inf, non-numbers, and invalid dictionaries."""
    with pytest.raises(ValueError, match="finite float"):
        check_gate("ece", float("nan"))
    with pytest.raises(ValueError, match="finite float"):
        check_gate("ece", float("inf"))

    with pytest.raises(ValueError, match="candidate must be a finite float"):
        is_better("ece", float("nan"), 0.05)
    with pytest.raises(ValueError, match="incumbent must be a finite float"):
        is_better("ece", 0.04, float("inf"))

    with pytest.raises(ValueError, match="dictionaries"):
        challenger_wins("not_a_dict", {})
    with pytest.raises(ValueError, match="min_relative_lift"):
        challenger_wins({}, {}, min_relative_lift=-0.05)
    with pytest.raises(ValueError, match="min_relative_lift"):
        challenger_wins({}, {}, min_relative_lift=float("nan"))
