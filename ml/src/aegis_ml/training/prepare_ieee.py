"""DVC stage: IEEE-CIS -> the canonical training frame (FR-1 data side).

IEEE-CIS ships TransactionDT as seconds from an unstated origin. The competition
convention is 2017-12-01; the absolute date does not matter, the ORDER does - every
split in this project is a time split, so a wrong origin would silently reorder the
weeks. The origin is written into the output metadata so it can never drift.

python -m aegis_ml.training.prepare_ieee --raw data/raw/ieee --out data/processed
"""
from __future__ import annotations

import argparse
import json
import pathlib
import numpy as np
import pandas as pd

ORIGIN = "2017-12-01"

ID_COLS = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card4",
    "card6",
    "addr1",
    "P_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "isFraud",
]

REQUIRED_TRANSACTION_COLS = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "isFraud",
]

CANONICAL_COLUMNS = [
    "transaction_id",
    "ts",
    "amount",
    "amount_minor",
    "channel",
    "account_id",
    "card_network",
    "card_type",
    "region",
    "email_domain",
    "device_type",
    "device_id",
    "is_fraud",
    "label_reported_at",
]


def load(raw: pathlib.Path | str) -> pd.DataFrame:
    """Load IEEE-CIS raw data and optional identity records."""
    raw_path = pathlib.Path(raw)
    tx_file = raw_path / "train_transaction.csv"
    if not tx_file.exists():
        raise FileNotFoundError(
            f"{tx_file} missing - download the IEEE-CIS competition data (Kaggle credentials required)"
        )

    tx = pd.read_csv(tx_file)
    idf = raw_path / "train_identity.csv"
    if idf.exists():
        id_df = pd.read_csv(idf)
        if "TransactionID" in id_df.columns and id_df["TransactionID"].duplicated().any():
            raise ValueError(
                "Duplicate TransactionID found in identity file: identity records must be unique"
            )
        initial_len = len(tx)
        tx = tx.merge(id_df, on="TransactionID", how="left")
        if len(tx) != initial_len:
            raise ValueError(
                f"Identity join altered transaction row count from {initial_len} to {len(tx)}"
            )
    return tx


def canonicalise(df: pd.DataFrame, origin: str = ORIGIN) -> pd.DataFrame:
    """Transform IEEE-CIS raw dataframe into canonical schema."""
    missing = [col for col in REQUIRED_TRANSACTION_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Required transaction columns missing from input: {missing}")

    try:
        orig_ts = pd.Timestamp(origin)
        if pd.isna(orig_ts):
            raise ValueError(f"Invalid origin timestamp: {origin}")
    except Exception as e:
        raise ValueError(f"Invalid origin timestamp '{origin}': {e}") from e

    if orig_ts.tzinfo is None:
        orig_ts = orig_ts.tz_localize("UTC")
    else:
        orig_ts = orig_ts.tz_convert("UTC")

    # Validate TransactionDT
    if df["TransactionDT"].isna().any():
        raise ValueError("TransactionDT contains missing/null timestamps (NaT/NaN)")
    try:
        dt_sec = pd.to_numeric(df["TransactionDT"], errors="raise")
    except Exception as e:
        raise ValueError(f"TransactionDT contains non-numeric values: {e}") from e
    if (dt_sec < 0).any():
        raise ValueError("TransactionDT contains negative timestamp values")

    # Validate isFraud binary labels
    if df["isFraud"].isna().any():
        raise ValueError("isFraud contains missing/null labels")
    unique_labels = set(df["isFraud"].unique())
    valid_binary = {0, 1, 0.0, 1.0, "0", "1", True, False}
    if not unique_labels.issubset(valid_binary):
        raise ValueError(f"isFraud contains non-binary values: {unique_labels - valid_binary}")

    # Validate TransactionID and format for delay calculation
    if df["TransactionID"].isna().any():
        raise ValueError("TransactionID contains missing/null values")
    tx_id_str = df["TransactionID"].astype(str)
    suffixes = tx_id_str.str.slice(-2)
    if (suffixes.str.len() < 2).any() or not suffixes.str.isdigit().all():
        invalid_examples = tx_id_str[(suffixes.str.len() < 2) | (~suffixes.str.isdigit())].head(3).tolist()
        raise ValueError(
            f"Invalid TransactionID format: last 2 characters must be numeric digits for delay calculation (examples: {invalid_examples})"
        )

    # Validate TransactionAmt
    if df["TransactionAmt"].isna().any():
        raise ValueError("TransactionAmt contains missing/null values")
    try:
        amt = pd.to_numeric(df["TransactionAmt"], errors="raise").astype(float)
    except Exception as e:
        raise ValueError(f"TransactionAmt contains non-numeric values: {e}") from e

    if amt.isna().any():
        raise ValueError("TransactionAmt contains missing/null values")
    if np.isinf(amt).any():
        raise ValueError("TransactionAmt contains non-finite/infinite values")
    if (amt <= 0.0).any():
        raise ValueError("TransactionAmt contains non-positive values (must be strictly positive)")

    # Scale to minor currency units (cents)
    scaled = amt * 100.0
    if np.isinf(scaled).any():
        raise ValueError("TransactionAmt values cause overflow when scaled to minor units")

    # M3 Nearest-cent quantization (USD cents as integer minor units)
    rounded_minor = scaled.round()
    if np.isinf(rounded_minor).any() or (rounded_minor >= 2**63).any():
        raise ValueError("TransactionAmt values exceed 64-bit integer minor unit representation")
    if (rounded_minor <= 0.0).any():
        raise ValueError("Positive transaction amount quantized to non-positive minor units")

    amount_minor = rounded_minor.astype("int64")

    ts = orig_ts + pd.to_timedelta(dt_sec, unit="s")
    is_fraud = df["isFraud"].astype(int)

    out = pd.DataFrame(
        {
            "transaction_id": tx_id_str,
            "ts": ts,
            "amount": amt,
            "amount_minor": amount_minor,
            "channel": df["ProductCD"].fillna("unknown").astype(str),
            "account_id": df["card1"].fillna("unknown").astype(str),
            "card_network": df.get("card4", pd.Series(index=df.index, dtype=object))
            .fillna("unknown")
            .astype(str),
            "card_type": df.get("card6", pd.Series(index=df.index, dtype=object))
            .fillna("unknown")
            .astype(str),
            "region": df.get("addr1", pd.Series(index=df.index, dtype=object))
            .fillna("unknown")
            .astype(str),
            "email_domain": df.get("P_emaildomain", pd.Series(index=df.index, dtype=object))
            .fillna("unknown")
            .astype(str),
            "device_type": df.get("DeviceType", pd.Series(index=df.index, dtype=object))
            .fillna("unknown")
            .astype(str),
            "device_id": df.get("DeviceInfo", pd.Series(index=df.index, dtype=object))
            .fillna("unknown")
            .astype(str),
            "is_fraud": is_fraud,
        }
    )

    # Chargebacks are not instant: give every labelled fraud a realistic reporting
    # delay so label maturity filtering has something honest to work with.
    # Synthetic delay policy: (suffix % 28) + 3 days (bounds 3-30 days inclusive).
    # This is a benchmark simulation policy aligned to the 30-day maturity horizon,
    # ensuring all mature fraud chargebacks land within 30 days without truncation.
    delay = pd.to_timedelta((suffixes.astype(int) % 28) + 3, unit="D")
    out["label_reported_at"] = out["ts"] + delay.where(out["is_fraud"] == 1)

    return out[CANONICAL_COLUMNS].sort_values(["ts", "transaction_id"]).reset_index(drop=True)


def compute_metadata(df: pd.DataFrame, origin: str = ORIGIN) -> dict:
    """Compute dataset summary metadata dictionary."""
    if len(df) == 0:
        return {
            "origin": origin,
            "rows": 0,
            "fraud_rate": 0.0,
            "first_ts": None,
            "last_ts": None,
            "weeks": 0,
        }

    first_ts = df["ts"].min()
    last_ts = df["ts"].max()
    weeks = int((last_ts - first_ts).days // 7) + 1

    return {
        "origin": origin,
        "rows": int(len(df)),
        "fraud_rate": float(df["is_fraud"].mean()),
        "first_ts": str(first_ts),
        "last_ts": str(last_ts),
        "weeks": weeks,
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point for CLI execution."""
    ap = argparse.ArgumentParser(description="Canonicalise IEEE-CIS dataset for AegisFlow M3.")
    ap.add_argument("--raw", default="data/raw/ieee", help="Path to raw IEEE-CIS dataset directory")
    ap.add_argument("--out", default="data/processed", help="Path to processed output directory")
    ap.add_argument("--origin", default=ORIGIN, help="Timestamp origin (YYYY-MM-DD)")
    a = ap.parse_args(argv)

    raw, out = pathlib.Path(a.raw), pathlib.Path(a.out)
    tx_file = raw / "train_transaction.csv"
    if not tx_file.exists():
        raise SystemExit(
            f"{tx_file} missing - download the IEEE-CIS competition data (Kaggle credentials required)"
        )

    df_raw = load(raw)
    df = canonicalise(df_raw, origin=a.origin)
    out.mkdir(parents=True, exist_ok=True)

    parquet_path = out / "ieee_transactions.parquet"
    meta_path = out / "ieee_meta.json"

    df.to_parquet(parquet_path, index=False)
    meta = compute_metadata(df, origin=a.origin)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(
        f"{len(df)} rows, fraud rate {meta['fraud_rate']:.3%}, "
        f"{df['ts'].min().date()} to {df['ts'].max().date()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
