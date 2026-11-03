"""Unit tests for transaction ingestion request and response schemas."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
import sys
from uuid import UUID, uuid4
import pytest
from pydantic import ValidationError

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ML_SRC = _REPO_ROOT / "ml" / "src"
_INGESTION_SRC = _REPO_ROOT / "services" / "ingestion" / "src"
for p in (_ML_SRC, _INGESTION_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ingestion.schemas import TransactionIn, TransactionOut, compare_canonical_fields


def test_valid_transaction_request():
    """Verify valid transaction payload parses successfully."""
    txn_id = uuid4()
    acc_id = uuid4()
    cp_id = uuid4()
    dev_id = uuid4()
    ip_id = uuid4()
    dt = datetime(2024, 5, 1, 12, 0, 0, tzinfo=timezone.utc)

    t = TransactionIn(
        id=txn_id,
        ts=dt,
        account_id=acc_id,
        counterparty_id=cp_id,
        device_id=dev_id,
        ip_id=ip_id,
        amount=Decimal("150.25"),
        currency="USD",
        channel="card",
    )
    assert t.id == txn_id
    assert t.amount == Decimal("150.25")
    assert t.currency == "USD"
    assert t.channel == "card"


def test_naive_timestamp_rejected():
    """Verify timestamps without timezone info are rejected with ValidationError."""
    naive_dt = datetime(2024, 5, 1, 12, 0, 0)
    with pytest.raises(ValidationError, match="timezone-aware"):
        TransactionIn(
            id=uuid4(),
            ts=naive_dt,
            account_id=uuid4(),
            amount=Decimal("100.00"),
            currency="USD",
            channel="card",
        )


def test_timezone_aware_timestamp_accepted():
    """Verify timezone-aware timestamps with non-UTC offsets are accepted and preserved."""
    offset = timezone(timedelta(hours=5, minutes=30))
    aware_dt = datetime(2024, 5, 1, 17, 30, 0, tzinfo=offset)
    t = TransactionIn(
        id=uuid4(),
        ts=aware_dt,
        account_id=uuid4(),
        counterparty_id=uuid4(),
        amount=Decimal("100.00"),
        currency="INR",
        channel="transfer",
    )
    assert t.ts.tzinfo is not None
    assert t.ts == aware_dt


def test_amount_supplied_as_float_rejected():
    """Verify float amounts on the wire are rejected to prevent floating-point inaccuracies."""
    with pytest.raises(ValidationError, match="Float amounts are not accepted"):
        TransactionIn.model_validate({
            "id": str(uuid4()),
            "ts": "2024-05-01T12:00:00Z",
            "account_id": str(uuid4()),
            "amount": 150.25,  # JSON float!
            "currency": "USD",
            "channel": "card",
        })


def test_zero_and_negative_amounts_rejected():
    """Verify zero and negative amounts are strictly rejected."""
    with pytest.raises(ValidationError):
        TransactionIn(
            id=uuid4(),
            ts=datetime.now(timezone.utc),
            account_id=uuid4(),
            amount=Decimal("0.00"),
            currency="USD",
            channel="card",
        )

    with pytest.raises(ValidationError):
        TransactionIn(
            id=uuid4(),
            ts=datetime.now(timezone.utc),
            account_id=uuid4(),
            amount=Decimal("-50.00"),
            currency="USD",
            channel="card",
        )


def test_more_than_two_decimal_places_rejected():
    """Verify currency amounts with >2 decimal places are rejected."""
    with pytest.raises(ValidationError):
        TransactionIn(
            id=uuid4(),
            ts=datetime.now(timezone.utc),
            account_id=uuid4(),
            amount=Decimal("100.555"),
            currency="USD",
            channel="card",
        )


def test_invalid_currency_rejected():
    """Verify currency must be exactly 3 uppercase letters."""
    for invalid in ("us", "usdt", "123", "usd", "US1"):
        with pytest.raises(ValidationError):
            TransactionIn(
                id=uuid4(),
                ts=datetime.now(timezone.utc),
                account_id=uuid4(),
                amount=Decimal("100.00"),
                currency=invalid,
                channel="card",
            )


def test_invalid_channel_rejected():
    """Verify channel must be one of literal ('card', 'transfer', 'wallet')."""
    for invalid in ("wire", "crypto", "cash", "CARD"):
        with pytest.raises(ValidationError):
            TransactionIn(
                id=uuid4(),
                ts=datetime.now(timezone.utc),
                account_id=uuid4(),
                amount=Decimal("100.00"),
                currency="USD",
                channel=invalid,
            )


def test_invalid_uuid_rejected():
    """Verify malformed UUID strings are rejected."""
    with pytest.raises(ValidationError):
        TransactionIn(
            id="not-a-valid-uuid",
            ts=datetime.now(timezone.utc),
            account_id=uuid4(),
            amount=Decimal("100.00"),
            currency="USD",
            channel="card",
        )


def test_feature_input_contract():
    """Verify feature_input produces canonical string UUIDs and integer minor units."""
    txn_id = uuid4()
    acc_id = uuid4()
    dev_id = uuid4()
    ip_id = uuid4()
    dt = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)

    t = TransactionIn(
        id=txn_id,
        ts=dt,
        account_id=acc_id,
        device_id=dev_id,
        ip_id=ip_id,
        amount=Decimal("250.75"),
        currency="USD",
        channel="wallet",
    )
    fi = t.feature_input()

    assert fi["txn_id"] == str(txn_id)
    assert fi["account_id"] == str(acc_id)
    assert fi["device_id"] == str(dev_id)
    assert fi["ip_id"] == str(ip_id)
    assert fi["amount_minor"] == 25075
    assert fi["ts"] == dt


def test_transaction_out_contract():
    """Verify TransactionOut schema initializes decision as None and duplicate as False by default."""
    txn_id = uuid4()
    out = TransactionOut(id=txn_id)
    assert out.id == txn_id
    assert out.decision is None
    assert out.duplicate is False


def test_compare_canonical_fields_exact_match():
    """Verify compare_canonical_fields returns True when all 10 fields match."""
    txn_id = uuid4()
    acc_id = uuid4()
    cp_id = uuid4()
    merch_id = uuid4()
    dev_id = uuid4()
    ip_id = uuid4()
    dt = datetime(2026, 10, 31, 12, 0, 0, tzinfo=timezone.utc)

    t1 = TransactionIn(
        id=txn_id,
        ts=dt,
        account_id=acc_id,
        counterparty_id=cp_id,
        merchant_id=merch_id,
        device_id=dev_id,
        ip_id=ip_id,
        amount=Decimal("199.99"),
        currency="USD",
        channel="card",
    )
    t2 = TransactionIn(
        id=txn_id,
        ts=dt,
        account_id=acc_id,
        counterparty_id=cp_id,
        merchant_id=merch_id,
        device_id=dev_id,
        ip_id=ip_id,
        amount=Decimal("199.99"),
        currency="USD",
        channel="card",
    )
    assert compare_canonical_fields(t1, t2) is True


def test_compare_canonical_fields_timezone_normalization():
    """Verify compare_canonical_fields returns True for same moment with different offsets."""
    txn_id = uuid4()
    acc_id = uuid4()
    dt_utc = datetime(2026, 10, 31, 10, 0, 0, tzinfo=timezone.utc)
    dt_ist = datetime(2026, 10, 31, 15, 30, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))

    t1 = TransactionIn(
        id=txn_id,
        ts=dt_utc,
        account_id=acc_id,
        amount=Decimal("50.00"),
        currency="USD",
        channel="card",
    )
    t2 = TransactionIn(
        id=txn_id,
        ts=dt_ist,
        account_id=acc_id,
        amount=Decimal("50.00"),
        currency="USD",
        channel="card",
    )
    assert compare_canonical_fields(t1, t2) is True


def test_compare_canonical_fields_detects_all_conflicts():
    """Verify compare_canonical_fields returns False when any canonical field differs."""
    txn_id = uuid4()
    acc_id = uuid4()
    dt = datetime(2026, 10, 31, 12, 0, 0, tzinfo=timezone.utc)

    base = TransactionIn(
        id=txn_id,
        ts=dt,
        account_id=acc_id,
        counterparty_id=uuid4(),
        merchant_id=uuid4(),
        device_id=uuid4(),
        ip_id=uuid4(),
        amount=Decimal("100.00"),
        currency="USD",
        channel="card",
    )

    # 1. ID mismatch
    assert compare_canonical_fields(base, base.model_copy(update={"id": uuid4()})) is False
    # 2. Timestamp mismatch
    assert compare_canonical_fields(base, base.model_copy(update={"ts": dt + timedelta(seconds=1)})) is False
    # 3. Account mismatch
    assert compare_canonical_fields(base, base.model_copy(update={"account_id": uuid4()})) is False
    # 4. Counterparty mismatch
    assert compare_canonical_fields(base, base.model_copy(update={"counterparty_id": uuid4()})) is False
    # 5. Merchant mismatch
    assert compare_canonical_fields(base, base.model_copy(update={"merchant_id": uuid4()})) is False
    # 6. Device mismatch
    assert compare_canonical_fields(base, base.model_copy(update={"device_id": uuid4()})) is False
    # 7. IP mismatch
    assert compare_canonical_fields(base, base.model_copy(update={"ip_id": uuid4()})) is False
    # 8. Amount mismatch
    assert compare_canonical_fields(base, base.model_copy(update={"amount": Decimal("100.01")})) is False
    # 9. Currency mismatch
    assert compare_canonical_fields(base, base.model_copy(update={"currency": "EUR"})) is False
    # 10. Channel mismatch
    assert compare_canonical_fields(base, base.model_copy(update={"channel": "wallet"})) is False
