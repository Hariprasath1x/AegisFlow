"""Unit tests for canonical feature definitions and conversions."""

import sys
from pathlib import Path
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import pytest

# Ensure ml/src is on sys.path
_ML_SRC = Path(__file__).resolve().parents[3] / "ml" / "src"
if str(_ML_SRC) not in sys.path:
    sys.path.insert(0, str(_ML_SRC))

from aegis_ml.features.definitions import (
    Feature,
    WINDOWS,
    DISTINCT_WINDOW_S,
    SHARE_WINDOW_S,
    KEY_TTL_S,
    GRAPH_TTL_S,
    GRAPH_FIELDS,
    FEATURES,
    FEATURE_LIST,
    BY_NAME,
    LUA_OUTPUTS,
    FEATURE_SET_VERSION,
    compute_feature_set_version,
    epoch_seconds,
    calendar,
    amount_minor,
)


def test_feature_count_and_uniqueness():
    """Verify exactly 19 features with unique names."""
    assert len(FEATURES) == 19
    assert len(FEATURE_LIST) == 19
    assert len(set(FEATURE_LIST)) == 19, "Feature names must be unique"
    assert len(BY_NAME) == 19


def test_feature_metadata_and_integrity():
    """Verify all features have valid entities, kinds, sources, and immutability."""
    valid_entities = {"account", "device", "ip"}
    valid_kinds = {"count", "sum", "distinct", "zscore", "recency", "calendar", "graph"}
    valid_sources = {"lua", "calendar", "graph"}

    for feat in FEATURES:
        assert isinstance(feat, Feature)
        assert feat.entity in valid_entities
        assert feat.kind in valid_kinds
        assert feat.source in valid_sources
        assert isinstance(feat.default, float)

        # Immutability check
        with pytest.raises(FrozenInstanceError):
            feat.default = 999.0  # type: ignore

    # Specific metadata verification
    # B. Velocity (8)
    for w in ("1m", "1h", "24h", "7d"):
        cnt_feat = BY_NAME[f"txn_count_{w}"]
        assert cnt_feat.entity == "account"
        assert cnt_feat.kind == "count"
        assert cnt_feat.window_s == WINDOWS[w]
        assert cnt_feat.include_current is True
        assert cnt_feat.default == 0.0
        assert cnt_feat.source == "lua"

        sum_feat = BY_NAME[f"amount_sum_minor_{w}"]
        assert sum_feat.entity == "account"
        assert sum_feat.kind == "sum"
        assert sum_feat.window_s == WINDOWS[w]
        assert sum_feat.include_current is True
        assert sum_feat.default == 0.0
        assert sum_feat.source == "lua"

    # C. Distinct account features (2)
    for name in ("distinct_devices_24h", "distinct_ips_24h"):
        f = BY_NAME[name]
        assert f.entity == "account"
        assert f.kind == "distinct"
        assert f.window_s == DISTINCT_WINDOW_S
        assert f.include_current is True
        assert f.default == 0.0
        assert f.source == "lua"

    # D. Sharing features (2)
    f_dev = BY_NAME["device_accounts_7d"]
    assert f_dev.entity == "device"
    assert f_dev.kind == "distinct"
    assert f_dev.window_s == SHARE_WINDOW_S
    assert f_dev.include_current is True
    assert f_dev.default == 0.0
    assert f_dev.source == "lua"

    f_ip = BY_NAME["ip_accounts_7d"]
    assert f_ip.entity == "ip"
    assert f_ip.kind == "distinct"
    assert f_ip.window_s == SHARE_WINDOW_S
    assert f_ip.include_current is True
    assert f_ip.default == 0.0
    assert f_ip.source == "lua"

    # E. Statistical features (2)
    f_z = BY_NAME["amount_zscore"]
    assert f_z.entity == "account"
    assert f_z.kind == "zscore"
    assert f_z.window_s is None
    assert f_z.include_current is False
    assert f_z.default == 0.0
    assert f_z.source == "lua"

    f_rec = BY_NAME["seconds_since_prev"]
    assert f_rec.entity == "account"
    assert f_rec.kind == "recency"
    assert f_rec.window_s is None
    assert f_rec.include_current is False
    assert f_rec.default == -1.0
    assert f_rec.source == "lua"

    # F. Calendar features (2)
    for name in ("hour_of_day", "day_of_week"):
        f_cal = BY_NAME[name]
        assert f_cal.entity == "account"
        assert f_cal.kind == "calendar"
        assert f_cal.window_s is None
        assert f_cal.include_current is True
        assert f_cal.default == 0.0
        assert f_cal.source == "calendar"

    # G. Graph features (3)
    f_g1 = BY_NAME["graph_component_size"]
    assert f_g1.entity == "account"
    assert f_g1.kind == "graph"
    assert f_g1.window_s is None
    assert f_g1.include_current is False
    assert f_g1.default == 1.0
    assert f_g1.source == "graph"

    f_g2 = BY_NAME["graph_dist_to_fraud"]
    assert f_g2.default == -1.0
    assert f_g2.source == "graph"

    f_g3 = BY_NAME["graph_risk_rank"]
    assert f_g3.default == 0.0
    assert f_g3.source == "graph"


def test_ordered_windows_and_constants():
    """Verify exact window ordering, values, and TTL constants."""
    assert list(WINDOWS.keys()) == ["1m", "1h", "24h", "7d"]
    assert WINDOWS["1m"] == 60
    assert WINDOWS["1h"] == 3_600
    assert WINDOWS["24h"] == 86_400
    assert WINDOWS["7d"] == 604_800

    assert DISTINCT_WINDOW_S == 86_400
    assert SHARE_WINDOW_S == 604_800
    assert KEY_TTL_S == 8 * 86_400
    assert GRAPH_TTL_S == 2 * 3_600


def test_canonical_collections():
    """Verify FEATURE_LIST, BY_NAME, LUA_OUTPUTS, and GRAPH_FIELDS consistency."""
    assert FEATURE_LIST == [f.name for f in FEATURES]
    for f in FEATURES:
        assert BY_NAME[f.name] is f

    assert len(LUA_OUTPUTS) == 14
    assert LUA_OUTPUTS == [f.name for f in FEATURES if f.source == "lua"]

    # Graph fields mapping
    graph_feature_names = [f.name for f in FEATURES if f.source == "graph"]
    assert sorted(GRAPH_FIELDS.keys()) == sorted(graph_feature_names)
    assert GRAPH_FIELDS["graph_component_size"] == "component_size"
    assert GRAPH_FIELDS["graph_dist_to_fraud"] == "dist_to_fraud"
    assert GRAPH_FIELDS["graph_risk_rank"] == "risk_rank"


def test_feature_set_version_determinism():
    """Verify deterministic 12-char SHA-256 fingerprint."""
    assert len(FEATURE_SET_VERSION) == 12
    assert all(c in "0123456789abcdef" for c in FEATURE_SET_VERSION)

    # Recomputing with identical inputs yields identical version
    recomputed = compute_feature_set_version(
        FEATURES,
        WINDOWS,
        DISTINCT_WINDOW_S,
        SHARE_WINDOW_S,
        KEY_TTL_S,
        GRAPH_TTL_S,
        GRAPH_FIELDS,
    )
    assert recomputed == FEATURE_SET_VERSION


def test_feature_set_version_sensitivity_to_changes():
    """Verify fingerprint changes when any semantic definition changes."""
    base_v = FEATURE_SET_VERSION

    # 1. Changing feature default
    mutated_features_default = list(FEATURES)
    mutated_features_default[0] = replace(mutated_features_default[0], default=99.0)
    v1 = compute_feature_set_version(
        mutated_features_default, WINDOWS, DISTINCT_WINDOW_S, SHARE_WINDOW_S, KEY_TTL_S, GRAPH_TTL_S, GRAPH_FIELDS
    )
    assert v1 != base_v

    # 2. Changing feature window
    mutated_features_window = list(FEATURES)
    mutated_features_window[0] = replace(mutated_features_window[0], window_s=120)
    v2 = compute_feature_set_version(
        mutated_features_window, WINDOWS, DISTINCT_WINDOW_S, SHARE_WINDOW_S, KEY_TTL_S, GRAPH_TTL_S, GRAPH_FIELDS
    )
    assert v2 != base_v

    # 3. Changing feature ordering (swap first two features)
    mutated_features_order = list(FEATURES)
    mutated_features_order[0], mutated_features_order[1] = mutated_features_order[1], mutated_features_order[0]
    v3 = compute_feature_set_version(
        mutated_features_order, WINDOWS, DISTINCT_WINDOW_S, SHARE_WINDOW_S, KEY_TTL_S, GRAPH_TTL_S, GRAPH_FIELDS
    )
    assert v3 != base_v

    # 4. Changing TTL
    v4 = compute_feature_set_version(
        FEATURES, WINDOWS, DISTINCT_WINDOW_S, SHARE_WINDOW_S, KEY_TTL_S + 1, GRAPH_TTL_S, GRAPH_FIELDS
    )
    assert v4 != base_v

    # 5. Changing graph field mapping
    mutated_graph_fields = dict(GRAPH_FIELDS)
    mutated_graph_fields["graph_risk_rank"] = "rank_score"
    v5 = compute_feature_set_version(
        FEATURES, WINDOWS, DISTINCT_WINDOW_S, SHARE_WINDOW_S, KEY_TTL_S, GRAPH_TTL_S, mutated_graph_fields
    )
    assert v5 != base_v

    # 6. Changing window definitions / order
    mutated_windows = {"1h": 3600, "1m": 60, "24h": 86400, "7d": 604800}
    v6 = compute_feature_set_version(
        FEATURES, mutated_windows, DISTINCT_WINDOW_S, SHARE_WINDOW_S, KEY_TTL_S, GRAPH_TTL_S, GRAPH_FIELDS
    )
    assert v6 != base_v


def test_epoch_seconds_conversions():
    """Verify timezone normalization, microsecond precision, and naive rejection."""
    # 1. Equivalent timestamps across offsets
    utc_ts = datetime(2026, 6, 15, 12, 0, 0, 500000, tzinfo=timezone.utc)
    ist_ts = utc_ts.astimezone(timezone(timedelta(hours=5, minutes=30)))
    est_ts = utc_ts.astimezone(timezone(timedelta(hours=-5)))

    sec_utc = epoch_seconds(utc_ts)
    sec_ist = epoch_seconds(ist_ts)
    sec_est = epoch_seconds(est_ts)

    assert sec_utc == pytest.approx(sec_ist)
    assert sec_utc == pytest.approx(sec_est)

    # 2. Microsecond precision preserved
    sec_micro = epoch_seconds(datetime(1970, 1, 1, 0, 0, 1, 123456, tzinfo=timezone.utc))
    assert sec_micro == pytest.approx(1.123456)

    # 3. Naive timestamp rejected
    naive_ts = datetime(2026, 6, 15, 12, 0, 0)
    with pytest.raises(ValueError, match="Naive datetime.*is not allowed"):
        epoch_seconds(naive_ts)


def test_calendar_features():
    """Verify UTC calendar extraction, weekday mapping (Monday=0), and naive rejection."""
    # Monday 14:30 UTC
    dt_monday_utc = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
    cal_m = calendar(dt_monday_utc)
    assert cal_m["hour_of_day"] == 14.0
    assert cal_m["day_of_week"] == 0.0  # Monday

    # Sunday 23:45 UTC
    dt_sunday_utc = datetime(2026, 1, 11, 23, 45, tzinfo=timezone.utc)
    cal_s = calendar(dt_sunday_utc)
    assert cal_s["hour_of_day"] == 23.0
    assert cal_s["day_of_week"] == 6.0  # Sunday

    # Timezone conversion: Monday 02:00 IST -> Sunday 20:30 UTC
    dt_ist = datetime(2026, 1, 5, 2, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    cal_ist = calendar(dt_ist)
    assert cal_ist["hour_of_day"] == 20.0
    assert cal_ist["day_of_week"] == 6.0  # Converted to Sunday UTC

    # Naive timestamp rejected
    with pytest.raises(ValueError, match="Naive datetime.*is not allowed"):
        calendar(datetime(2026, 1, 5, 14, 30))


def test_amount_minor_conversions():
    """Verify Decimal and float conversions to integer minor units."""
    # Decimal
    assert amount_minor(Decimal("123.45")) == 12345
    assert amount_minor(Decimal("10.50")) == 1050
    assert amount_minor(Decimal("0.00")) == 0
    assert amount_minor(Decimal("0.01")) == 1
    assert amount_minor(Decimal("999999.99")) == 99999999

    # Floats
    assert amount_minor(10.50) == 1050
    assert amount_minor(29.99) == 2999
    assert amount_minor(0.0) == 0

    # Integers
    assert amount_minor(100) == 10000
    assert amount_minor(0) == 0

    # Negative values
    assert amount_minor(Decimal("-50.25")) == -5025
    assert amount_minor(-10.50) == -1050

    # String format
    assert amount_minor("45.67") == 4567


def test_amount_minor_fractional_cent_rejection():
    """Verify fractional minor units (sub-cent precision) are rejected without silent rounding."""
    for bad_amount in (
        Decimal("123.456"),
        Decimal("0.001"),
        Decimal("10.555"),
        10.555,
        "10.555",
        Decimal("-10.123"),
    ):
        with pytest.raises(ValueError, match="fractional minor units"):
            amount_minor(bad_amount)


def test_amount_minor_non_finite_rejection():
    """Verify non-finite amounts (NaN, Infinity) are rejected."""
    for bad_num in (
        float("nan"),
        float("inf"),
        float("-inf"),
        Decimal("NaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
    ):
        with pytest.raises(ValueError, match="must be finite"):
            amount_minor(bad_num)

    with pytest.raises(TypeError, match="Unsupported amount type"):
        amount_minor(None)  # type: ignore
