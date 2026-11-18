"""Unit tests for isotonic calibration, reliability tables, and ECE."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aegis_ml.models.calibration import (
    Calibrator,
    expected_calibration_error,
    reliability,
)


# --- 1. Calibrator Unit Tests ---


def test_calibrator_unfitted_state():
    """Calibrator starts unfitted and raises RuntimeError on transform before fit."""
    cal = Calibrator()
    assert cal.fitted is False
    with pytest.raises(RuntimeError, match="calibrator not fitted"):
        cal.transform([0.5])


def test_calibrator_fit_returns_self():
    """fit() returns the same calibrator instance and marks it fitted."""
    cal = Calibrator()
    returned = cal.fit([0.1, 0.4, 0.8], [0, 0, 1])
    assert returned is cal
    assert cal.fitted is True


def test_calibrator_monotonic_and_bounded():
    """Calibrator outputs are monotonic with respect to input scores and bounded in [0, 1]."""
    scores = np.array([0.05, 0.2, 0.35, 0.6, 0.85, 0.95])
    labels = np.array([0, 0, 1, 0, 1, 1])

    cal = Calibrator().fit(scores, labels)
    eval_grid = np.linspace(-0.5, 1.5, 50)
    calibrated = cal.transform(eval_grid)

    # Monotonicity: calibrated[i] <= calibrated[i+1]
    assert np.all(np.diff(calibrated) >= -1e-9)

    # Strictly bounded in [0.0, 1.0]
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)
    assert np.all(np.isfinite(calibrated))


def test_calibrator_out_of_bounds_clipping():
    """Calibrator clips inputs below min training score and above max training score."""
    scores = [0.2, 0.8]
    labels = [0, 1]

    cal = Calibrator().fit(scores, labels)
    # Testing far below and far above training score range
    test_scores = [-10.0, 0.0, 0.2, 0.8, 1.0, 50.0]
    preds = cal.transform(test_scores)

    # Input < 0.2 clips to prediction at 0.2 (0.0)
    assert preds[0] == pytest.approx(0.0)
    assert preds[1] == pytest.approx(0.0)
    assert preds[2] == pytest.approx(0.0)

    # Input > 0.8 clips to prediction at 0.8 (1.0)
    assert preds[3] == pytest.approx(1.0)
    assert preds[4] == pytest.approx(1.0)
    assert preds[5] == pytest.approx(1.0)


def test_calibrator_empty_scores_transform():
    """Calibrator transform accepts empty input and returns empty 1D array."""
    cal = Calibrator().fit([0.1, 0.9], [0, 1])
    empty_preds = cal.transform([])
    assert isinstance(empty_preds, np.ndarray)
    assert len(empty_preds) == 0


def test_calibrator_fixes_squashed_scores_guide_scenario():
    """Verify that isotonic calibration reduces ECE on under-confident scores."""
    rng = np.random.default_rng(7)
    n = 8000
    rate = 0.10
    y = (rng.random(n) < rate).astype(int)
    s = rng.normal(0, 1, n) + 2.5 * y
    probs_raw = 1 / (1 + np.exp(-(s * 0.25)))  # deliberately under-confident

    cal = Calibrator().fit(probs_raw[:4000], y[:4000])
    probs_cal = cal.transform(probs_raw[4000:])

    ece_raw = expected_calibration_error(y[4000:], probs_raw[4000:])
    ece_cal = expected_calibration_error(y[4000:], probs_cal)

    assert ece_cal < ece_raw
    rows = reliability(y[4000:], probs_cal)
    assert len(rows) > 0
    assert all(0.0 <= r["observed"] <= 1.0 for r in rows)


# --- 2. Reliability Function Tests ---


def test_reliability_bin_edges_and_keys():
    """reliability generates exact keys and correct bin string formatting."""
    y = [0, 0, 1, 1]
    p = [0.05, 0.08, 0.82, 0.88]
    rows = reliability(y, p, bins=10)

    # Only two bins should have entries: [0.0,0.1) and [0.8,0.9)
    assert len(rows) == 2
    for r in rows:
        assert set(r.keys()) == {"bin", "n", "predicted", "observed"}
        assert isinstance(r["bin"], str)
        assert isinstance(r["n"], int)
        assert isinstance(r["predicted"], float)
        assert isinstance(r["observed"], float)

    assert rows[0]["bin"] == "[0.0,0.1)"
    assert rows[0]["n"] == 2
    assert rows[0]["predicted"] == pytest.approx(0.065)
    assert rows[0]["observed"] == pytest.approx(0.0)

    assert rows[1]["bin"] == "[0.8,0.9)"
    assert rows[1]["n"] == 2
    assert rows[1]["predicted"] == pytest.approx(0.85)
    assert rows[1]["observed"] == pytest.approx(1.0)


def test_reliability_boundary_values_0_and_1():
    """Exact boundary values 0.0 and 1.0 are placed in the first and last bins respectively."""
    y = [0, 1]
    p = [0.0, 1.0]
    rows = reliability(y, p, bins=10)

    assert len(rows) == 2
    assert rows[0]["bin"] == "[0.0,0.1)"
    assert rows[0]["n"] == 1
    assert rows[0]["predicted"] == 0.0
    assert rows[0]["observed"] == 0.0

    assert rows[1]["bin"] == "[0.9,1.0)"
    assert rows[1]["n"] == 1
    assert rows[1]["predicted"] == 1.0
    assert rows[1]["observed"] == 1.0


def test_reliability_empty_bins_omitted():
    """Empty bins are skipped in output list."""
    y = [0, 1]
    p = [0.25, 0.75]
    rows = reliability(y, p, bins=10)

    # 10 bins total, but only 2 have data: [0.2,0.3) and [0.7,0.8)
    assert len(rows) == 2
    bins_present = [r["bin"] for r in rows]
    assert bins_present == ["[0.2,0.3)", "[0.7,0.8)"]


# --- 3. ECE Hand-Calculated Tests ---


def test_ece_hand_computed_exact():
    """expected_calibration_error matches hand-calculated arithmetic."""
    # 2 samples in [0.0, 0.1): predicted=0.05, observed=0.0 -> gap=0.05
    # 2 samples in [0.9, 1.0): predicted=0.95, observed=1.0 -> gap=0.05
    # Total n = 4, weighted gap = (2 * 0.05 + 2 * 0.05) / 4 = 0.05
    y = [0, 0, 1, 1]
    p = [0.05, 0.05, 0.95, 0.95]
    ece = expected_calibration_error(y, p, bins=10)
    assert ece == pytest.approx(0.05)


def test_ece_perfect_calibration():
    """Perfect calibration gives 0.0 ECE."""
    y = [0, 0, 1, 1]
    p = [0.0, 0.0, 1.0, 1.0]
    ece = expected_calibration_error(y, p, bins=10)
    assert ece == pytest.approx(0.0)


def test_ece_worst_case_miscalibration():
    """Completely miscalibrated probabilities give 1.0 ECE."""
    y = [0, 0, 1, 1]
    p = [1.0, 1.0, 0.0, 0.0]
    ece = expected_calibration_error(y, p, bins=10)
    assert ece == pytest.approx(1.0)


# --- 4. Input Validation and Boundary Tests ---


def test_calibrator_fit_validation():
    """Calibrator.fit rejects invalid, mismatched, or non-finite inputs."""
    cal = Calibrator()

    # Mismatched length
    with pytest.raises(ValueError, match="scores length"):
        cal.fit([0.1, 0.2], [0])

    # Empty inputs
    with pytest.raises(ValueError, match="scores cannot be empty"):
        cal.fit([], [])

    # Non-finite scores
    with pytest.raises(ValueError, match="finite numeric values"):
        cal.fit([0.1, np.nan], [0, 1])
    with pytest.raises(ValueError, match="finite numeric values"):
        cal.fit([0.1, np.inf], [0, 1])

    # Multidimensional scores
    with pytest.raises(ValueError, match="must be 1D"):
        cal.fit([[0.1], [0.2]], [0, 1])

    # Invalid labels
    with pytest.raises(ValueError, match="binary labels"):
        cal.fit([0.1, 0.2], [0, 2])
    with pytest.raises(ValueError, match="binary labels"):
        cal.fit([0.1, 0.2], [0.0, 0.5])


def test_calibrator_transform_validation():
    """Calibrator.transform rejects non-finite and multidimensional scores."""
    cal = Calibrator().fit([0.1, 0.9], [0, 1])

    with pytest.raises(ValueError, match="finite numeric values"):
        cal.transform([0.5, np.nan])

    with pytest.raises(ValueError, match="must be 1D"):
        cal.transform([[0.5], [0.8]])


def test_reliability_and_ece_validation():
    """reliability and ECE validate probability bounds, binary labels, shapes, and bins."""
    y = [0, 1]
    p = [0.2, 0.8]

    # Probabilities out of [0, 1] bounds
    with pytest.raises(ValueError, match=r"bounded in \[0\.0, 1\.0\]"):
        reliability(y, [-0.01, 0.5])
    with pytest.raises(ValueError, match=r"bounded in \[0\.0, 1\.0\]"):
        reliability(y, [0.5, 1.01])
    with pytest.raises(ValueError, match=r"bounded in \[0\.0, 1\.0\]"):
        expected_calibration_error(y, [-0.1, 0.5])

    # Non-finite probabilities
    with pytest.raises(ValueError, match="finite numeric values"):
        reliability(y, [0.5, np.nan])
    with pytest.raises(ValueError, match="finite numeric values"):
        reliability(y, [0.5, np.inf])

    # Mismatched lengths
    with pytest.raises(ValueError, match="y_true length"):
        reliability([0, 1, 0], [0.2, 0.8])
    with pytest.raises(ValueError, match="y_true length"):
        expected_calibration_error([0, 1, 0], [0.2, 0.8])

    # Invalid binary labels
    with pytest.raises(ValueError, match="binary labels"):
        reliability([0, 2], p)
    with pytest.raises(ValueError, match="binary labels"):
        expected_calibration_error([0.5, 1.0], p)

    # Empty inputs
    with pytest.raises(ValueError, match="cannot be empty"):
        reliability([], [])
    with pytest.raises(ValueError, match="cannot be empty"):
        expected_calibration_error([], [])

    # Invalid bin arguments
    for invalid_bins in [0, -5, "10", 0.5, True]:
        with pytest.raises(ValueError, match="positive integer"):
            reliability(y, p, bins=invalid_bins)
        with pytest.raises(ValueError, match="positive integer"):
            expected_calibration_error(y, p, bins=invalid_bins)


# --- 5. Immutability and Determinism Tests ---


def test_inputs_are_not_mutated():
    """Neither Calibrator, reliability, nor ECE modify caller arrays in-place."""
    scores = np.array([0.1, 0.4, 0.7, 0.9])
    labels = np.array([0, 0, 1, 1])

    scores_copy = scores.copy()
    labels_copy = labels.copy()

    cal = Calibrator().fit(scores, labels)
    _ = cal.transform(scores)
    _ = reliability(labels, scores)
    _ = expected_calibration_error(labels, scores)

    np.testing.assert_array_equal(scores, scores_copy)
    np.testing.assert_array_equal(labels, labels_copy)


def test_pandas_series_and_list_support():
    """Functions transparently support Pandas Series and Python lists."""
    y_series = pd.Series([0, 1, 0, 1])
    p_series = pd.Series([0.1, 0.9, 0.2, 0.8])

    cal = Calibrator().fit(p_series, y_series)
    preds = cal.transform(p_series)
    assert len(preds) == 4

    rows = reliability(y_series, p_series)
    assert len(rows) > 0

    ece = expected_calibration_error(y_series, p_series)
    assert isinstance(ece, float)
