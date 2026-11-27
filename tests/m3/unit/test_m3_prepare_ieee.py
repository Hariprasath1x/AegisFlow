"""Unit tests for IEEE-CIS canonical dataset preparation (Milestone 3 - Step 3.3).

Verifies:
1. Canonical schema, column order, and dtypes.
2. Timestamp calculation from pinned and custom origin.
3. Left-join with optional identity records and missing identity tolerance.
4. Prevention of row duplication on non-unique identity keys.
5. Fail-closed validation on missing columns, invalid timestamps, bad labels, and malformed IDs.
6. Deterministic fraud reporting delay calculation within [3, 47] days.
7. Missing reporting timestamps (NaT) for legitimate transactions.
8. Parquet read/write roundtrip and JSON metadata fidelity.
9. Deterministic output on repeated execution.
"""
from __future__ import annotations

import json
import pathlib
import numpy as np
import pandas as pd
import pytest

from aegis_ml.features import definitions as D
from aegis_ml.training.prepare_ieee import (
    CANONICAL_COLUMNS,
    ORIGIN,
    REQUIRED_TRANSACTION_COLS,
    canonicalise,
    compute_metadata,
    load,
    main,
)


def sample_transactions() -> pd.DataFrame:
    """Create a minimal valid raw transactions dataframe."""
    return pd.DataFrame(
        {
            "TransactionID": [3000000, 3000001, 3000044, 3000045, 3000099],
            "TransactionDT": [0, 3600, 86400, 172800, 259200],
            "TransactionAmt": [10.50, 20.00, 150.25, 45.00, 1000.00],
            "ProductCD": ["W", "H", "C", "W", "R"],
            "card1": [1001, 1002, 1003, 1004, 1005],
            "card4": ["visa", "mastercard", None, "discover", "visa"],
            "card6": ["debit", "credit", "credit", None, "debit"],
            "addr1": [315, None, 181, 204, None],
            "P_emaildomain": ["gmail.com", None, "yahoo.com", "anonymous.com", "hotmail.com"],
            "isFraud": [0, 1, 0, 1, 1],
        }
    )


def sample_identity() -> pd.DataFrame:
    """Create a minimal valid raw identity dataframe."""
    return pd.DataFrame(
        {
            "TransactionID": [3000001, 3000045],
            "DeviceType": ["mobile", "desktop"],
            "DeviceInfo": ["iOS Device", "Windows"],
        }
    )


def test_canonical_schema_and_column_ordering() -> None:
    """1. Canonical output produces exactly the expected columns in the exact order."""
    df = sample_transactions()
    res = canonicalise(df)
    assert list(res.columns) == CANONICAL_COLUMNS
    assert len(res) == len(df)


def test_pinned_origin_timestamp_conversion() -> None:
    """2. Correct timestamp conversion from the pinned default origin (2017-12-01 UTC)."""
    df = sample_transactions()
    res = canonicalise(df)

    expected_t0 = pd.Timestamp("2017-12-01 00:00:00", tz="UTC")
    expected_t1 = pd.Timestamp("2017-12-01 01:00:00", tz="UTC")  # 3600 seconds
    expected_t2 = pd.Timestamp("2017-12-02 00:00:00", tz="UTC")  # 86400 seconds

    assert res["ts"].iloc[0] == expected_t0
    assert res["ts"].iloc[1] == expected_t1
    assert res["ts"].iloc[2] == expected_t2
    assert res["ts"].dt.tz is not None


def test_custom_origin_reflected_in_output_and_metadata() -> None:
    """3. Custom origin works and is reflected in both timestamps and metadata."""
    df = sample_transactions()
    custom_origin = "2020-06-15"
    res = canonicalise(df, origin=custom_origin)

    assert res["ts"].iloc[0] == pd.Timestamp("2020-06-15 00:00:00", tz="UTC")
    meta = compute_metadata(res, origin=custom_origin)
    assert meta["origin"] == custom_origin
    assert meta["first_ts"].startswith("2020-06-15")


def test_transaction_id_and_amount_conversion() -> None:
    """4. Correct transaction_id (string), amount (float), and amount_minor (int64) conversion."""
    df = sample_transactions()
    res = canonicalise(df)

    assert pd.api.types.is_string_dtype(res["transaction_id"]) or res["transaction_id"].dtype == object
    assert res["transaction_id"].tolist() == ["3000000", "3000001", "3000044", "3000045", "3000099"]
    assert res["amount"].dtype == float
    assert res["amount"].tolist() == [10.50, 20.00, 150.25, 45.00, 1000.00]
    assert res["amount_minor"].dtype in (np.int64, int)
    assert res["amount_minor"].tolist() == [1050, 2000, 15025, 4500, 100000]


def test_optional_identity_join_and_unknown_fallbacks(tmp_path: pathlib.Path) -> None:
    """5, 6, 7. Identity join works, missing identity is tolerated, row count is preserved."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    tx = sample_transactions()
    tx.to_csv(raw_dir / "train_transaction.csv", index=False)

    # Without identity file
    loaded_no_id = load(raw_dir)
    assert len(loaded_no_id) == len(tx)
    canon_no_id = canonicalise(loaded_no_id)
    assert (canon_no_id["device_type"] == "unknown").all()
    assert (canon_no_id["device_id"] == "unknown").all()

    # With identity file
    idf = sample_identity()
    idf.to_csv(raw_dir / "train_identity.csv", index=False)
    loaded_with_id = load(raw_dir)
    assert len(loaded_with_id) == len(tx)

    canon_with_id = canonicalise(loaded_with_id)
    assert len(canon_with_id) == len(tx)

    # 3000001 had identity
    row_id1 = canon_with_id[canon_with_id["transaction_id"] == "3000001"].iloc[0]
    assert row_id1["device_type"] == "mobile"
    assert row_id1["device_id"] == "iOS Device"

    # 3000000 lacked identity -> 'unknown'
    row_id0 = canon_with_id[canon_with_id["transaction_id"] == "3000000"].iloc[0]
    assert row_id0["device_type"] == "unknown"
    assert row_id0["device_id"] == "unknown"


def test_duplicate_identity_keys_fail_closed(tmp_path: pathlib.Path) -> None:
    """8. Duplicate identity keys raise ValueError to prevent row multiplication."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    tx = sample_transactions()
    tx.to_csv(raw_dir / "train_transaction.csv", index=False)

    dup_idf = pd.DataFrame(
        {
            "TransactionID": [3000001, 3000001],
            "DeviceType": ["mobile", "desktop"],
            "DeviceInfo": ["iOS", "Android"],
        }
    )
    dup_idf.to_csv(raw_dir / "train_identity.csv", index=False)

    with pytest.raises(ValueError, match="Duplicate TransactionID"):
        load(raw_dir)


def test_missing_required_columns_rejected() -> None:
    """9. Missing required transaction columns raise descriptive ValueError."""
    for col in REQUIRED_TRANSACTION_COLS:
        df = sample_transactions().drop(columns=[col])
        with pytest.raises(ValueError, match="Required transaction columns missing"):
            canonicalise(df)


def test_invalid_and_missing_timestamps_fail_closed() -> None:
    """10. Missing, non-numeric, or negative timestamps raise ValueError."""
    df_nan = sample_transactions()
    df_nan["TransactionDT"] = df_nan["TransactionDT"].astype(object)
    df_nan.loc[0, "TransactionDT"] = None
    with pytest.raises(ValueError, match="TransactionDT contains missing/null timestamps"):
        canonicalise(df_nan)

    df_neg = sample_transactions()
    df_neg.loc[0, "TransactionDT"] = -100
    with pytest.raises(ValueError, match="TransactionDT contains negative timestamp values"):
        canonicalise(df_neg)

    df_str = sample_transactions()
    df_str["TransactionDT"] = df_str["TransactionDT"].astype(object)
    df_str.loc[0, "TransactionDT"] = "not_a_number"
    with pytest.raises(ValueError, match="TransactionDT contains non-numeric values"):
        canonicalise(df_str)


def test_invalid_transaction_ids_rejected() -> None:
    """11. Short or non-digit transaction ID suffixes fail closed."""
    df_short = sample_transactions()
    df_short["TransactionID"] = df_short["TransactionID"].astype(object)
    df_short.loc[0, "TransactionID"] = "1"
    with pytest.raises(ValueError, match="last 2 characters must be numeric digits"):
        canonicalise(df_short)

    df_alpha = sample_transactions()
    df_alpha["TransactionID"] = df_alpha["TransactionID"].astype(object)
    df_alpha.loc[0, "TransactionID"] = "TX_A"
    with pytest.raises(ValueError, match="last 2 characters must be numeric digits"):
        canonicalise(df_alpha)


def test_fraud_reporting_delay_logic_and_boundaries() -> None:
    """12, 13, 14. Fraud rows receive deterministic delay in [3, 30] days; legitimate rows get NaT."""
    df = sample_transactions()
    # Add fraud transactions to test boundaries under K=28:
    # TransactionID 3000028: suffix 28 -> (28 % 28) + 3 = 3 days (minimum boundary)
    # TransactionID 3000027: suffix 27 -> (27 % 28) + 3 = 30 days (maximum boundary)
    fraud_boundaries = pd.DataFrame(
        {
            "TransactionID": [3000028, 3000027],
            "TransactionDT": [360000, 360000],
            "TransactionAmt": [250.00, 300.00],
            "ProductCD": ["W", "W"],
            "card1": [1006, 1007],
            "card4": ["visa", "mastercard"],
            "card6": ["debit", "credit"],
            "addr1": [315, 315],
            "P_emaildomain": ["gmail.com", "yahoo.com"],
            "isFraud": [1, 1],
        }
    )
    df = pd.concat([df, fraud_boundaries], ignore_index=True)
    res = canonicalise(df)

    # Legitimate rows: is_fraud == 0 -> label_reported_at must be NaT
    legit = res[res["is_fraud"] == 0]
    assert len(legit) > 0
    assert legit["label_reported_at"].isna().all()

    # Fraud rows: is_fraud == 1 -> label_reported_at > ts
    fraud = res[res["is_fraud"] == 1]
    assert fraud["label_reported_at"].notna().all()
    assert (fraud["label_reported_at"] > fraud["ts"]).all()

    delays = (fraud["label_reported_at"] - fraud["ts"]).dt.total_seconds() / 86400
    assert (delays >= 3).all()
    assert (delays <= 30).all()
    assert delays.min() == 3
    assert delays.max() == 30

    # Check exact values:
    # ID 3000001: suffix 1 -> (1 % 28) + 3 = 4 days
    row1 = res[res["transaction_id"] == "3000001"].iloc[0]
    assert (row1["label_reported_at"] - row1["ts"]) == pd.Timedelta(days=4)

    # ID 3000028: suffix 28 -> (28 % 28) + 3 = 3 days (minimum boundary)
    row28 = res[res["transaction_id"] == "3000028"].iloc[0]
    assert (row28["label_reported_at"] - row28["ts"]) == pd.Timedelta(days=3)

    # ID 3000027: suffix 27 -> (27 % 28) + 3 = 30 days (maximum boundary)
    row27 = res[res["transaction_id"] == "3000027"].iloc[0]
    assert (row27["label_reported_at"] - row27["ts"]) == pd.Timedelta(days=30)

    # ID 3000045: suffix 45 -> (45 % 28) + 3 = 17 + 3 = 20 days
    row45 = res[res["transaction_id"] == "3000045"].iloc[0]
    assert (row45["label_reported_at"] - row45["ts"]) == pd.Timedelta(days=20)

    # ID 3000099: suffix 99 -> (99 % 28) + 3 = 15 + 3 = 18 days
    row99 = res[res["transaction_id"] == "3000099"].iloc[0]
    assert (row99["label_reported_at"] - row99["ts"]) == pd.Timedelta(days=18)


def test_invalid_non_binary_fraud_labels_rejected() -> None:
    """15. Non-binary or null fraud labels are rejected."""
    df_missing = sample_transactions()
    df_missing.loc[0, "isFraud"] = None
    with pytest.raises(ValueError, match="isFraud contains missing/null labels"):
        canonicalise(df_missing)

    df_arbitrary = sample_transactions()
    df_arbitrary.loc[0, "isFraud"] = 2
    with pytest.raises(ValueError, match="isFraud contains non-binary values"):
        canonicalise(df_arbitrary)


def test_parquet_roundtrip_and_schema_fidelity(tmp_path: pathlib.Path) -> None:
    """16. Canonical dataframe survives Parquet save and reload without schema distortion."""
    df = sample_transactions()
    res = canonicalise(df)

    out_file = tmp_path / "test.parquet"
    res.to_parquet(out_file, index=False)

    loaded = pd.read_parquet(out_file)
    assert list(loaded.columns) == CANONICAL_COLUMNS
    assert len(loaded) == len(res)
    assert pd.api.types.is_string_dtype(loaded["transaction_id"]) or loaded["transaction_id"].dtype == object
    assert loaded["amount"].dtype == float
    assert loaded["amount_minor"].dtype in (np.int64, int)
    assert loaded["amount_minor"].tolist() == res["amount_minor"].tolist()
    assert loaded["is_fraud"].dtype in (np.int32, np.int64, int)
    assert loaded["ts"].dt.tz is not None


def test_amount_minor_two_decimal_matches_definitions_amount_minor() -> None:
    """17. Standard 2-decimal amounts produce amount_minor matching D.amount_minor() exactly."""
    df = sample_transactions()
    res = canonicalise(df)

    for amt_float, amt_minor_val in zip(res["amount"], res["amount_minor"]):
        expected = D.amount_minor(amt_float)
        assert amt_minor_val == expected


def test_amount_minor_sub_cent_quantization() -> None:
    """18. Sub-cent amounts are quantized via nearest-cent rule while preserving exact source amounts."""
    df = sample_transactions()
    # Replace TransactionAmt with representative IEEE-CIS values
    df["TransactionAmt"] = [75.887, 16.495, 0.251, 100.50, 32.325]
    res = canonicalise(df)

    # Original amounts are preserved without rounding or truncation
    np.testing.assert_array_equal(res["amount"].to_numpy(), np.array([75.887, 16.495, 0.251, 100.50, 32.325]))

    # Quantized integer minor units (cents)
    # 75.887 * 100 = 7588.7 -> 7589
    # 16.495 * 100 = 1649.5 -> 1650
    # 0.251 * 100 = 25.1 -> 25
    # 100.50 * 100 = 10050.0 -> 10050
    # 32.325 * 100 = 3232.5000000000005 (float64 binary) -> 3233
    expected_minor = [7589, 1650, 25, 10050, 3233]
    assert res["amount_minor"].tolist() == expected_minor
    assert res["amount_minor"].dtype == np.int64


def test_invalid_and_non_finite_amounts_fail_closed() -> None:
    """19. Non-finite, null, NaN, zero, negative, and overflow amounts fail closed with ValueError."""
    # 1. Null / None
    df_null = sample_transactions()
    df_null.loc[0, "TransactionAmt"] = None
    with pytest.raises(ValueError, match="TransactionAmt contains missing/null"):
        canonicalise(df_null)

    # 2. NaN
    df_nan = sample_transactions()
    df_nan.loc[0, "TransactionAmt"] = np.nan
    with pytest.raises(ValueError, match="TransactionAmt contains missing/null"):
        canonicalise(df_nan)

    # 3. Non-numeric string
    df_str = sample_transactions()
    df_str["TransactionAmt"] = df_str["TransactionAmt"].astype(object)
    df_str.loc[0, "TransactionAmt"] = "invalid_money"
    with pytest.raises(ValueError, match="TransactionAmt contains non-numeric values"):
        canonicalise(df_str)

    # 4. Zero amount
    df_zero = sample_transactions()
    df_zero.loc[0, "TransactionAmt"] = 0.0
    with pytest.raises(ValueError, match="TransactionAmt contains non-positive values"):
        canonicalise(df_zero)

    # 5. Negative amount
    df_neg = sample_transactions()
    df_neg.loc[0, "TransactionAmt"] = -15.50
    with pytest.raises(ValueError, match="TransactionAmt contains non-positive values"):
        canonicalise(df_neg)

    # 6. Positive infinity
    df_inf = sample_transactions()
    df_inf.loc[0, "TransactionAmt"] = float("inf")
    with pytest.raises(ValueError, match="TransactionAmt contains non-finite/infinite values"):
        canonicalise(df_inf)

    # 7. Negative infinity
    df_ninf = sample_transactions()
    df_ninf.loc[0, "TransactionAmt"] = float("-inf")
    with pytest.raises(ValueError, match="TransactionAmt contains non-finite/infinite values"):
        canonicalise(df_ninf)

    # 8. Overflow beyond 64-bit signed integer
    df_overflow = sample_transactions()
    df_overflow.loc[0, "TransactionAmt"] = 1e19
    with pytest.raises(ValueError, match="exceed 64-bit integer minor unit"):
        canonicalise(df_overflow)

    # 9. Value scaling to exactly 2**63
    df_exact_2_63 = sample_transactions()
    df_exact_2_63.loc[0, "TransactionAmt"] = float(2**63) / 100.0
    with pytest.raises(ValueError, match="exceed 64-bit integer minor unit"):
        canonicalise(df_exact_2_63)

    # 10. Value rounding up to 2**63
    df_round_to_2_63 = sample_transactions()
    df_round_to_2_63.loc[0, "TransactionAmt"] = (float(2**63) - 0.49) / 100.0
    with pytest.raises(ValueError, match="exceed 64-bit integer minor unit"):
        canonicalise(df_round_to_2_63)


def test_metadata_values_match_generated_output(tmp_path: pathlib.Path) -> None:
    """17. Metadata JSON correctly derives counts, fraud rate, time boundaries and weeks."""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    raw_dir.mkdir()

    tx = sample_transactions()
    tx.to_csv(raw_dir / "train_transaction.csv", index=False)

    ret = main(["--raw", str(raw_dir), "--out", str(out_dir)])
    assert ret == 0

    meta_file = out_dir / "ieee_meta.json"
    assert meta_file.is_file()
    meta = json.loads(meta_file.read_text())

    assert meta["origin"] == ORIGIN
    assert meta["rows"] == 5
    assert meta["fraud_rate"] == 3 / 5  # 3 fraud out of 5
    assert meta["first_ts"] == "2017-12-01 00:00:00+00:00"
    assert meta["last_ts"] == "2017-12-04 00:00:00+00:00"
    assert meta["weeks"] == 1


def test_deterministic_reproducibility() -> None:
    """18. Repeated canonicalisation of the same input produces identical output."""
    df = sample_transactions()
    res1 = canonicalise(df)
    res2 = canonicalise(df)

    pd.testing.assert_frame_equal(res1, res2)


def test_cli_missing_dataset_error(tmp_path: pathlib.Path) -> None:
    """Verify CLI raises SystemExit with actionable error when train_transaction.csv is missing."""
    empty_raw = tmp_path / "empty_raw"
    empty_raw.mkdir()

    with pytest.raises(SystemExit) as exc_info:
        main(["--raw", str(empty_raw), "--out", str(tmp_path / "out")])

    assert "missing - download the IEEE-CIS competition data (Kaggle credentials required)" in str(
        exc_info.value
    )
