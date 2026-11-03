"""Pydantic request and response schemas for AegisFlow transaction ingestion."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Literal
from typing_extensions import Self
from uuid import UUID
from pydantic import BaseModel, Field, field_validator, model_validator

from aegis_ml.features import definitions as D


def compare_canonical_fields(a: TransactionIn, b: TransactionIn) -> bool:
    """Compare all 10 canonical business fields between two transactions.

    Returns True if all fields match; False if any field conflicts.
    Timestamps are compared normalized to UTC.
    Amounts are compared using exact Decimal/minor-unit semantics.
    """
    if a.id != b.id:
        return False

    a_ts_utc = a.ts.astimezone(timezone.utc) if a.ts.tzinfo is not None else a.ts.replace(tzinfo=timezone.utc)
    b_ts_utc = b.ts.astimezone(timezone.utc) if b.ts.tzinfo is not None else b.ts.replace(tzinfo=timezone.utc)
    if a_ts_utc != b_ts_utc:
        return False

    if a.account_id != b.account_id:
        return False

    if a.counterparty_id != b.counterparty_id:
        return False

    if a.merchant_id != b.merchant_id:
        return False

    if a.device_id != b.device_id:
        return False

    if a.ip_id != b.ip_id:
        return False

    if a.amount != b.amount:
        return False

    if a.currency != b.currency:
        return False

    if a.channel != b.channel:
        return False

    return True


class TransactionIn(BaseModel):
    """Incoming transaction schema validated prior to persistence and feature updates."""

    id: UUID
    ts: datetime
    account_id: UUID
    counterparty_id: UUID | None = None
    merchant_id: UUID | None = None
    device_id: UUID | None = None
    ip_id: UUID | None = None
    amount: Decimal = Field(gt=0, max_digits=14, decimal_places=2)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    channel: Literal["card", "transfer", "wallet"]

    @field_validator("ts")
    @classmethod
    def validate_timezone_aware(cls, v: datetime) -> datetime:
        """Reject timezone-naive timestamps to guarantee event-time correctness."""
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("Timestamp must be timezone-aware (e.g. UTC ISO 8601 string)")
        return v

    @field_validator("amount", mode="before")
    @classmethod
    def reject_float_amount(cls, v: Any) -> Any:
        """Reject JSON float representations of money on the wire to prevent precision loss."""
        if isinstance(v, float):
            raise ValueError(
                "Float amounts are not accepted on the wire; supply amount as a string, integer, or Decimal"
            )
        return v

    @model_validator(mode="after")
    def validate_transfer_destination(self) -> Self:
        """Require counterparty_id for transfer channel to identify destination account."""
        if self.channel == "transfer" and self.counterparty_id is None:
            raise ValueError(
                "counterparty_id is required when channel='transfer' to designate the destination account"
            )
        return self

    def feature_input(self) -> dict[str, Any]:
        """Convert transaction model into canonical feature store input dictionary.

        Converts UUIDs to strings and computes integer minor currency units.
        """
        return {
            "txn_id": str(self.id),
            "id": str(self.id),
            "account_id": str(self.account_id),
            "ts": self.ts,
            "timestamp": self.ts,
            "amount_minor": D.amount_minor(self.amount),
            "amount": self.amount,
            "device_id": str(self.device_id) if self.device_id else None,
            "ip_id": str(self.ip_id) if self.ip_id else None,
        }


class TransactionOut(BaseModel):
    """Outgoing response schema returned to caller upon ingestion."""

    id: UUID
    decision: str | None = None
    duplicate: bool = False
