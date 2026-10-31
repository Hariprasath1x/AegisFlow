"""Canonical Feature Definitions for AegisFlow M2 Feature Store.

Single source of truth for all 19 AegisFlow features across offline
computation (batch / point-in-time) and online serving (Redis / Lua).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from typing import Mapping, Sequence

# Canonical Epoch for event time windows
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Feature:
    """Authoritative definition for an AegisFlow feature."""

    name: str
    entity: str  # account, device, or ip
    kind: str  # count, sum, distinct, zscore, recency, calendar, or graph
    window_s: int | None
    include_current: bool
    default: float
    source: str  # lua, calendar, or graph


# Declared window definitions in strict canonical order
WINDOWS: dict[str, int] = {
    "1m": 60,
    "1h": 3_600,
    "24h": 86_400,
    "7d": 604_800,
}

# Window and TTL constants
DISTINCT_WINDOW_S: int = 86_400
SHARE_WINDOW_S: int = 604_800
KEY_TTL_S: int = 8 * 86_400  # 8 days for online keys
GRAPH_TTL_S: int = 2 * 3_600  # 2 hours for hourly graph feature snapshots

# Field mapping for graph feature extraction
GRAPH_FIELDS: dict[str, str] = {
    "graph_component_size": "component_size",
    "graph_dist_to_fraud": "dist_to_fraud",
    "graph_risk_rank": "risk_rank",
}


def _build_features() -> tuple[Feature, ...]:
    """Build the ordered list of all 19 AegisFlow canonical features."""
    feats: list[Feature] = []

    # B. Velocity features (8): count and sum across windows in declared window order
    for window_name, window_sec in WINDOWS.items():
        feats.append(
            Feature(
                name=f"txn_count_{window_name}",
                entity="account",
                kind="count",
                window_s=window_sec,
                include_current=True,
                default=0.0,
                source="lua",
            )
        )
        feats.append(
            Feature(
                name=f"amount_sum_minor_{window_name}",
                entity="account",
                kind="sum",
                window_s=window_sec,
                include_current=True,
                default=0.0,
                source="lua",
            )
        )

    # C. Distinct account features (2): 24h distinct devices and IPs
    feats.append(
        Feature(
            name="distinct_devices_24h",
            entity="account",
            kind="distinct",
            window_s=DISTINCT_WINDOW_S,
            include_current=True,
            default=0.0,
            source="lua",
        )
    )
    feats.append(
        Feature(
            name="distinct_ips_24h",
            entity="account",
            kind="distinct",
            window_s=DISTINCT_WINDOW_S,
            include_current=True,
            default=0.0,
            source="lua",
        )
    )

    # D. Device/IP sharing features (2): 7d distinct accounts linked to device / IP
    feats.append(
        Feature(
            name="device_accounts_7d",
            entity="device",
            kind="distinct",
            window_s=SHARE_WINDOW_S,
            include_current=True,
            default=0.0,
            source="lua",
        )
    )
    feats.append(
        Feature(
            name="ip_accounts_7d",
            entity="ip",
            kind="distinct",
            window_s=SHARE_WINDOW_S,
            include_current=True,
            default=0.0,
            source="lua",
        )
    )

    # E. Account statistical features (2): amount z-score and seconds since previous
    feats.append(
        Feature(
            name="amount_zscore",
            entity="account",
            kind="zscore",
            window_s=None,
            include_current=False,
            default=0.0,
            source="lua",
        )
    )
    feats.append(
        Feature(
            name="seconds_since_prev",
            entity="account",
            kind="recency",
            window_s=None,
            include_current=False,
            default=-1.0,
            source="lua",
        )
    )

    # F. Calendar features (2): hour of day and day of week (UTC)
    feats.append(
        Feature(
            name="hour_of_day",
            entity="account",
            kind="calendar",
            window_s=None,
            include_current=True,
            default=0.0,
            source="calendar",
        )
    )
    feats.append(
        Feature(
            name="day_of_week",
            entity="account",
            kind="calendar",
            window_s=None,
            include_current=True,
            default=0.0,
            source="calendar",
        )
    )

    # G. Graph features (3): component size, distance to fraud, risk rank
    feats.append(
        Feature(
            name="graph_component_size",
            entity="account",
            kind="graph",
            window_s=None,
            include_current=False,
            default=1.0,
            source="graph",
        )
    )
    feats.append(
        Feature(
            name="graph_dist_to_fraud",
            entity="account",
            kind="graph",
            window_s=None,
            include_current=False,
            default=-1.0,
            source="graph",
        )
    )
    feats.append(
        Feature(
            name="graph_risk_rank",
            entity="account",
            kind="graph",
            window_s=None,
            include_current=False,
            default=0.0,
            source="graph",
        )
    )

    return tuple(feats)


# Canonical feature collections
FEATURES: tuple[Feature, ...] = _build_features()
FEATURE_LIST: list[str] = [f.name for f in FEATURES]
BY_NAME: dict[str, Feature] = {f.name: f for f in FEATURES}
LUA_OUTPUTS: list[str] = [f.name for f in FEATURES if f.source == "lua"]


def compute_feature_set_version(
    features: Sequence[Feature],
    windows: Mapping[str, int],
    distinct_window_s: int,
    share_window_s: int,
    key_ttl_s: int,
    graph_ttl_s: int,
    graph_fields: Mapping[str, str],
) -> str:
    """Compute a deterministic 12-char SHA-256 fingerprint for the feature set.

    Accounts for all feature definitions, window order, windows, TTLs, and graph fields.
    """
    payload = {
        "distinct_window_s": distinct_window_s,
        "features": [asdict(f) for f in features],
        "graph_fields": dict(graph_fields),
        "graph_ttl_s": graph_ttl_s,
        "key_ttl_s": key_ttl_s,
        "share_window_s": share_window_s,
        "windows": list(windows.items()),
    }
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:12]


# Canonical 12-hex character fingerprint of the feature set
FEATURE_SET_VERSION: str = compute_feature_set_version(
    FEATURES,
    WINDOWS,
    DISTINCT_WINDOW_S,
    SHARE_WINDOW_S,
    KEY_TTL_S,
    GRAPH_TTL_S,
    GRAPH_FIELDS,
)


def epoch_seconds(ts: datetime) -> float:
    """Convert an event timestamp to seconds since Unix epoch with microsecond precision.

    Rejects naive timestamps. Preserves timezone equivalence across offsets.
    """
    if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
        raise ValueError(f"Naive datetime {ts!r} is not allowed; must have timezone info.")
    return (ts - _EPOCH).total_seconds()


def calendar(ts: datetime) -> dict[str, float]:
    """Extract UTC calendar features (hour_of_day, day_of_week) from an event timestamp.

    - Hour range: 0.0 - 23.0
    - Day of week range: 0.0 - 6.0 (Monday = 0.0)
    Rejects naive timestamps.
    """
    if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
        raise ValueError(f"Naive datetime {ts!r} is not allowed; must have timezone info.")
    ts_utc = ts.astimezone(timezone.utc)
    return {
        "hour_of_day": float(ts_utc.hour),
        "day_of_week": float(ts_utc.weekday()),
    }


def amount_minor(amount: Decimal | float | int | str) -> int:
    """Convert currency amount into integer minor units (e.g. cents, paise).

    Supports Decimal, int, and two-decimal-place floats.
    Avoids floating-point precision loss when Decimal inputs are supplied.
    Rejects non-finite amounts (NaN, Infinity) and fractional minor units.
    """
    if isinstance(amount, float):
        if not math.isfinite(amount):
            raise ValueError(f"Amount must be finite, got {amount}")
        # Convert float to Decimal via string representation to eliminate binary float artifacts
        d = Decimal(str(amount))
    elif isinstance(amount, Decimal):
        if not amount.is_finite():
            raise ValueError(f"Amount must be finite, got {amount}")
        d = amount
    elif isinstance(amount, int):
        d = Decimal(amount)
    elif isinstance(amount, str):
        try:
            d = Decimal(amount)
        except InvalidOperation:
            raise ValueError(f"Invalid monetary string {amount!r}")
        if not d.is_finite():
            raise ValueError(f"Amount must be finite, got {amount}")
    else:
        raise TypeError(f"Unsupported amount type: {type(amount)!r}")

    scaled = d * 100
    if scaled != scaled.to_integral_value():
        raise ValueError(
            f"Amount {amount} contains fractional minor units (sub-cent precision)"
        )
    return int(scaled.to_integral_value())

