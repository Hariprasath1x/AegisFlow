"""Promotion gates, with no MLflow import so they are testable on a laptop.
Every bound traces to a line in the spec."""
from __future__ import annotations

import math
import numpy as np

GATES: dict[str, tuple[bool, float, str]] = {
    "recall_at_1pct_fpr": (False, 0.50, "section 19 - absolute floor before anything ships"),
    "lift_over_rules": (False, 0.30, "objective 5.2 - beat the rules baseline by >= 30%"),
    "pr_auc": (False, 0.60, "section 14 - headline ranking quality"),
    "ece": (True, 0.05, "section 14 - calibrated within 5 points"),
    "typology_detection": (False, 0.80, "objective 5.4 - >= 80% of injected typologies"),
}


def check_gate(metric: str, value: float) -> tuple[bool, str]:
    if not isinstance(metric, str):
        raise ValueError("metric must be a string")
    if (
        not isinstance(value, (int, float, np.floating, np.integer))
        or isinstance(value, bool)
        or math.isnan(value)
        or math.isinf(value)
    ):
        raise ValueError(f"value for {metric} must be a finite float, got {value}")

    val = float(value)
    if metric not in GATES:
        return True, f"{metric}={val:.4f} (no gate defined)"
    lower_better, bound, ref = GATES[metric]
    ok = val <= bound if lower_better else val >= bound
    rel = "<=" if lower_better else ">="
    return ok, f"{metric}={val:.4f} {rel} {bound} [{ref}] -> {'pass' if ok else 'FAIL'}"


def is_better(metric: str, candidate: float, incumbent: float) -> bool:
    """Strictly better, in the metric's own direction. A tie is churn, not a win."""
    if not isinstance(metric, str):
        raise ValueError("metric must be a string")
    if (
        not isinstance(candidate, (int, float, np.floating, np.integer))
        or isinstance(candidate, bool)
        or math.isnan(candidate)
        or math.isinf(candidate)
    ):
        raise ValueError(f"candidate must be a finite float, got {candidate}")
    if (
        not isinstance(incumbent, (int, float, np.floating, np.integer))
        or isinstance(incumbent, bool)
        or math.isnan(incumbent)
        or math.isinf(incumbent)
    ):
        raise ValueError(f"incumbent must be a finite float, got {incumbent}")

    cand, inc = float(candidate), float(incumbent)
    lower_better = GATES.get(metric, (False, 0.0, ""))[0]
    return cand < inc if lower_better else cand > inc


def challenger_wins(
    challenger: dict,
    champion: dict,
    primary: str = "recall_at_1pct_fpr",
    min_relative_lift: float = 0.02,
) -> tuple[bool, str]:
    """Champion/challenger promotion rule (FR-7). A challenger has to clear every
    gate AND beat the champion on the primary metric by a margin, because a 0.2%
    'improvement' inside the noise is not worth a production swap."""
    if not isinstance(challenger, dict) or not isinstance(champion, dict):
        raise ValueError("challenger and champion must be dictionaries")
    if not isinstance(primary, str):
        raise ValueError("primary must be a string")
    if (
        not isinstance(min_relative_lift, (int, float, np.floating, np.integer))
        or isinstance(min_relative_lift, bool)
        or math.isnan(min_relative_lift)
        or math.isinf(min_relative_lift)
        or min_relative_lift < 0
    ):
        raise ValueError(
            f"min_relative_lift must be a non-negative finite float, got {min_relative_lift}"
        )

    failed = [
        check_gate(m, v)[1]
        for m, v in challenger.items()
        if m in GATES and not check_gate(m, v)[0]
    ]
    if failed:
        return False, "challenger failed gates: " + "; ".join(failed)

    c, p = challenger.get(primary), champion.get(primary)
    if c is None or p is None:
        return False, f"{primary} missing from one of the models"

    if (
        not isinstance(c, (int, float, np.floating, np.integer))
        or isinstance(c, bool)
        or math.isnan(c)
        or math.isinf(c)
    ):
        raise ValueError(f"challenger {primary} must be a finite float, got {c}")
    if (
        not isinstance(p, (int, float, np.floating, np.integer))
        or isinstance(p, bool)
        or math.isnan(p)
        or math.isinf(p)
    ):
        raise ValueError(f"champion {primary} must be a finite float, got {p}")

    c_f, p_f = float(c), float(p)
    if not is_better(primary, c_f, p_f):
        return False, f"{primary}: challenger {c_f:.4f} does not beat champion {p_f:.4f}"

    lift = abs(c_f - p_f) / max(abs(p_f), 1e-9)
    if lift < min_relative_lift:
        return False, (
            f"{primary}: {lift:.2%} relative lift is inside the noise "
            f"band ({min_relative_lift:.0%})"
        )
    return True, f"{primary}: {p_f:.4f} -> {c_f:.4f} ({lift:+.2%}) - promote"
