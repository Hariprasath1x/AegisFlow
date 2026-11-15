"""Pydantic domain models and integrity validators for AegisFlow M2 scenarios."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import re
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from aegis_ml.scenarios.constants import (
    MATURITY_BOUNDARY,
    OBSERVATION_CUTOFF,
    RETAINED_TYPOLOGIES,
    TYPOLOGY_REGISTRY,
)


def _check_utc_datetime(v: Any, field_name: str) -> datetime:
    """Validate that a value is a timezone-aware UTC datetime with offset 0."""
    if v is None:
        return v
    if not isinstance(v, datetime):
        raise ValueError(f"Field '{field_name}' must be a datetime, got {type(v)}")
    if v.tzinfo is None:
        raise ValueError(f"Field '{field_name}' must be timezone-aware (UTC), got naive: {v!r}")
    offset = v.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError(f"Field '{field_name}' must have UTC offset 0, got offset {offset} for {v!r}")
    return v


class EvidenceProvenance(BaseModel):
    """Provenance and traceability metadata for ground-truth evidence."""

    synthetic: Literal[True] = True
    scenario_id: str
    typology: str | None
    investigation_ref: str
    alert_refs: list[str] = Field(default_factory=list)
    dispute_ref: str | None = None
    analyst_id: str
    confirmation_event: str
    manifest_reference_at: datetime

    @field_validator("manifest_reference_at", mode="after")
    @classmethod
    def validate_utc_manifest_ref(cls, v: datetime) -> datetime:
        return _check_utc_datetime(v, "manifest_reference_at")


class SyntheticAccount(BaseModel):
    """Synthetic account representation in scenario manifest."""

    account_id: UUID
    scenario_id: str
    account_number: str
    account_type: Literal["current", "savings", "card", "wallet"]
    currency: str = "INR"
    opened_on: str
    status: Literal["active", "dormant", "frozen", "closed"] = "active"
    is_mule: bool = False

    @field_validator("opened_on", mode="after")
    @classmethod
    def validate_opened_on_format(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"opened_on must be YYYY-MM-DD string, got {v!r}")
        return v


class SyntheticTransaction(BaseModel):
    """Synthetic transaction representation in scenario manifest."""

    txn_id: UUID
    external_ref: str
    account_id: UUID
    scenario_tag: str
    channel: Literal["card", "wire", "upi", "ach", "atm", "transfer", "wallet"]
    direction: Literal["debit", "credit"] | None = None
    amount: Decimal
    currency: str = "INR"
    occurred_at: datetime
    destination_account_id: UUID | None = None
    merchant_id: UUID | None = None
    ip_id: UUID | None = None
    device_id: str | None = None
    ip_address: str | None = None
    evidence: EvidenceProvenance | None = None

    @field_validator("occurred_at", mode="after")
    @classmethod
    def validate_utc_occurred_at(cls, v: datetime) -> datetime:
        return _check_utc_datetime(v, "occurred_at")

    @model_validator(mode="after")
    def validate_transaction_invariants(self) -> SyntheticTransaction:
        if len(self.external_ref) > 64:
            raise ValueError(f"external_ref exceeds 64 chars: {self.external_ref!r}")
        if self.amount <= Decimal("0"):
            raise ValueError(f"amount must be strictly positive: {self.amount}")
        if self.amount.as_tuple().exponent != -2:
            raise ValueError(f"amount must have exactly 2 decimal places: {self.amount}")
        if self.currency != "INR":
            raise ValueError(f"currency must be 'INR', got: {self.currency!r}")
        if self.merchant_id is not None:
            raise ValueError("merchant_id must be None in Phase 2")
        if self.ip_id is not None:
            raise ValueError("ip_id must be None in Phase 2")
        return self


class SyntheticLabel(BaseModel):
    """Ground truth label representation matching post-m2_0003 schema."""

    label_id: UUID
    txn_id: UUID | None = None
    account_id: UUID | None = None
    label_target: Literal["transaction", "account"]
    is_fraud: bool
    fraud_type: str | None = None
    confidence: float
    source: str = "synthetic_seed"
    labeled_at: datetime
    incident_start_at: datetime | None = None
    incident_end_at: datetime | None = None
    evidence: EvidenceProvenance

    @field_validator("labeled_at", "incident_start_at", "incident_end_at", mode="after")
    @classmethod
    def validate_utc_label_datetimes(cls, v: datetime | None, info: Any) -> datetime | None:
        if v is not None:
            return _check_utc_datetime(v, info.field_name)
        return None

    @model_validator(mode="after")
    def validate_label_constraints(self) -> SyntheticLabel:
        # 1. Target presence constraint (ck_ground_truth_target_presence)
        if self.label_target == "transaction":
            if self.txn_id is None or self.account_id is not None:
                raise ValueError("transaction label requires txn_id set and account_id NULL")
        elif self.label_target == "account":
            if self.account_id is None or self.txn_id is not None:
                raise ValueError("account label requires account_id set and txn_id NULL")
        else:
            raise ValueError(f"Invalid label_target: {self.label_target}")

        # 2. Paired incident boundaries (ck_ground_truth_incident_paired_boundaries)
        if (self.incident_start_at is None) != (self.incident_end_at is None):
            raise ValueError("incident boundaries must both be NULL or both non-NULL")

        # 3. Incident chronology & discovery (ck_ground_truth_incident_chronology/discovery)
        if self.incident_start_at and self.incident_end_at:
            if self.incident_start_at > self.incident_end_at:
                raise ValueError(
                    f"incident_start_at ({self.incident_start_at}) must be <= incident_end_at ({self.incident_end_at})"
                )
            if self.incident_start_at > self.labeled_at:
                raise ValueError(
                    f"incident_start_at ({self.incident_start_at}) must be <= labeled_at ({self.labeled_at})"
                )
            if self.incident_end_at > self.labeled_at:
                raise ValueError(
                    f"incident_end_at ({self.incident_end_at}) must be <= labeled_at ({self.labeled_at})"
                )

        if self.labeled_at > OBSERVATION_CUTOFF:
            raise ValueError(f"labeled_at ({self.labeled_at}) cannot exceed OBSERVATION_CUTOFF ({OBSERVATION_CUTOFF})")

        # 4. Confidence & Typology
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")

        valid_typologies = set(TYPOLOGY_REGISTRY.values()) | set(RETAINED_TYPOLOGIES)
        valid_typologies.discard(None)

        if self.is_fraud:
            if self.confidence < 0.85:
                raise ValueError(f"Fraud label confidence must be >= 0.85, got {self.confidence}")
            if self.fraud_type is None:
                raise ValueError("Fraud labels must specify a non-null fraud_type")
            if self.fraud_type not in valid_typologies:
                raise ValueError(f"fraud_type '{self.fraud_type}' not in recognized typologies {valid_typologies}")
        else:
            if self.confidence != 1.0:
                raise ValueError(f"Clean label confidence must be exactly 1.0, got {self.confidence}")
            if self.fraud_type is not None:
                raise ValueError(f"Clean labels must have fraud_type=None, got {self.fraud_type!r}")

        return self


class ScenarioDefinition(BaseModel):
    """Container for accounts, transactions, and labels in a single scenario."""

    scenario_id: str
    typology: str | None
    accounts: list[SyntheticAccount]
    transactions: list[SyntheticTransaction]
    labels: list[SyntheticLabel]


class ScenarioManifest(BaseModel):
    """Root container representing the full reproducible scenario manifest."""

    manifest_version: str = "2.0"
    seed: int = 20261010
    observation_cutoff: datetime
    manifest_reference_at: datetime
    manifest_hash: str = ""
    scenarios: list[ScenarioDefinition]

    @field_validator("observation_cutoff", "manifest_reference_at", mode="after")
    @classmethod
    def validate_utc_manifest_times(cls, v: datetime, info: Any) -> datetime:
        return _check_utc_datetime(v, info.field_name)

    @model_validator(mode="after")
    def validate_manifest_integrity(self) -> ScenarioManifest:
        """Cross-scenario validation of referential integrity and label invariants."""
        all_accounts: dict[UUID, SyntheticAccount] = {}
        all_transactions: dict[UUID, SyntheticTransaction] = {}
        all_labels: list[SyntheticLabel] = []
        all_external_refs: set[str] = set()
        all_label_ids: set[UUID] = set()

        scenario_by_txn: dict[UUID, str] = {}
        scenario_by_account: dict[UUID, str] = {}

        for sc in self.scenarios:
            for acc in sc.accounts:
                if acc.account_id in all_accounts:
                    raise ValueError(f"Duplicate account_id across scenarios: {acc.account_id}")
                all_accounts[acc.account_id] = acc
                scenario_by_account[acc.account_id] = sc.scenario_id

            for txn in sc.transactions:
                if txn.txn_id in all_transactions:
                    raise ValueError(f"Duplicate txn_id across scenarios: {txn.txn_id}")
                if txn.external_ref in all_external_refs:
                    raise ValueError(f"Duplicate external_ref across scenarios: {txn.external_ref}")
                all_transactions[txn.txn_id] = txn
                all_external_refs.add(txn.external_ref)
                scenario_by_txn[txn.txn_id] = sc.scenario_id

            for lbl in sc.labels:
                if lbl.label_id in all_label_ids:
                    raise ValueError(f"Duplicate label_id across scenarios: {lbl.label_id}")
                all_label_ids.add(lbl.label_id)
                all_labels.append(lbl)

        # 1. Referential integrity: transactions -> accounts
        for txn in all_transactions.values():
            if txn.account_id not in all_accounts:
                raise ValueError(f"Transaction {txn.txn_id} references non-existent account {txn.account_id}")
            if txn.destination_account_id and txn.destination_account_id not in all_accounts:
                raise ValueError(
                    f"Transaction {txn.txn_id} destination {txn.destination_account_id} not in manifest accounts"
                )

        # 2. Referential integrity: labels -> accounts / transactions
        for sc in self.scenarios:
            for lbl in sc.labels:
                if lbl.label_target == "transaction":
                    if lbl.txn_id not in all_transactions:
                        raise ValueError(f"Label {lbl.label_id} references non-existent txn {lbl.txn_id}")
                    # Ensure transaction label belongs to the same scenario
                    if scenario_by_txn[lbl.txn_id] != sc.scenario_id:
                        raise ValueError(
                            f"Label {lbl.label_id} belongs to scenario {sc.scenario_id} "
                            f"but references txn {lbl.txn_id} belonging to scenario {scenario_by_txn[lbl.txn_id]}"
                        )
                    target_txn = all_transactions[lbl.txn_id]
                    # Verify incident window contains transaction occurred_at
                    if lbl.incident_start_at and lbl.incident_end_at:
                        if not (lbl.incident_start_at <= target_txn.occurred_at <= lbl.incident_end_at):
                            raise ValueError(
                                f"Transaction {target_txn.txn_id} occurred at {target_txn.occurred_at}, "
                                f"which is outside label incident window [{lbl.incident_start_at}, {lbl.incident_end_at}]"
                            )
                    # Verify maturity boundary for directly labeled fraud transactions
                    if lbl.is_fraud and lbl.confidence >= 0.85:
                        if target_txn.occurred_at > MATURITY_BOUNDARY:
                            raise ValueError(
                                f"Directly labeled fraud transaction {target_txn.txn_id} occurred at "
                                f"{target_txn.occurred_at}, which is after MATURITY_BOUNDARY {MATURITY_BOUNDARY}"
                            )

                elif lbl.label_target == "account":
                    if lbl.account_id not in all_accounts:
                        raise ValueError(f"Label {lbl.label_id} references non-existent account {lbl.account_id}")
                    if scenario_by_account[lbl.account_id] != sc.scenario_id:
                        raise ValueError(
                            f"Label {lbl.label_id} belongs to scenario {sc.scenario_id} "
                            f"but references account {lbl.account_id} belonging to {scenario_by_account[lbl.account_id]}"
                        )

        # 3. Conflict detection
        # Direct transaction label conflicts
        txn_labels: dict[UUID, list[SyntheticLabel]] = {}
        for lbl in all_labels:
            if lbl.label_target == "transaction" and lbl.txn_id:
                txn_labels.setdefault(lbl.txn_id, []).append(lbl)

        for tid, lbls in txn_labels.items():
            has_fraud = any(l.is_fraud for l in lbls)
            has_clean = any(not l.is_fraud for l in lbls)
            if has_fraud and has_clean:
                raise ValueError(f"Transaction {tid} has conflicting direct fraud and clean labels")

        # Account-level label conflicts and incident window overlaps
        acct_labels: dict[UUID, list[SyntheticLabel]] = {}
        for lbl in all_labels:
            if lbl.label_target == "account" and lbl.account_id:
                acct_labels.setdefault(lbl.account_id, []).append(lbl)

        for aid, lbls in acct_labels.items():
            has_fraud = any(l.is_fraud for l in lbls)
            has_clean = any(not l.is_fraud for l in lbls)
            if has_fraud and has_clean:
                raise ValueError(f"Account {aid} has conflicting fraud and clean labels")

            # Check for overlapping windows of same typology
            fraud_by_typology: dict[str, list[SyntheticLabel]] = {}
            for l in lbls:
                if l.is_fraud and l.fraud_type:
                    fraud_by_typology.setdefault(l.fraud_type, []).append(l)

            for typ, flbls in fraud_by_typology.items():
                bounded = [l for l in flbls if l.incident_start_at and l.incident_end_at]
                bounded.sort(key=lambda x: x.incident_start_at)  # type: ignore[arg-type]
                for i in range(len(bounded) - 1):
                    if bounded[i].incident_end_at >= bounded[i + 1].incident_start_at:  # type: ignore[operator]
                        raise ValueError(
                            f"Account {aid} has overlapping incident windows for typology {typ}: "
                            f"[{bounded[i].incident_start_at}, {bounded[i].incident_end_at}] overlaps "
                            f"[{bounded[i+1].incident_start_at}, {bounded[i+1].incident_end_at}]"
                        )

        # 4. Evidence provenance consistency
        real_reg_patterns = [
            re.compile(r"^DISP-\d{4}"),
            re.compile(r"^CB-\d{4}"),
            re.compile(r"^SAR-\d{4}"),
        ]

        for sc in self.scenarios:
            for lbl in sc.labels:
                ev = lbl.evidence
                if not ev.synthetic:
                    raise ValueError(f"Label {lbl.label_id} evidence must have synthetic=True")
                if not ev.investigation_ref.startswith("SYN-INV-"):
                    raise ValueError(f"investigation_ref must start with SYN-INV-, got {ev.investigation_ref}")
                if not ev.analyst_id.startswith("SYN-ANALYST-"):
                    raise ValueError(f"analyst_id must start with SYN-ANALYST-, got {ev.analyst_id}")
                if ev.dispute_ref and not ev.dispute_ref.startswith("SYN-DISP-"):
                    raise ValueError(f"dispute_ref must start with SYN-DISP-, got {ev.dispute_ref}")
                for aref in ev.alert_refs:
                    if not aref.startswith("SYN-ALERT-"):
                        raise ValueError(f"alert_ref must start with SYN-ALERT-, got {aref}")

                for pat in real_reg_patterns:
                    if pat.search(ev.investigation_ref) or (ev.dispute_ref and pat.search(ev.dispute_ref)):
                        raise ValueError("Evidence reference must not resemble real regulatory/dispute formats")

                if ev.scenario_id != sc.scenario_id:
                    raise ValueError(
                        f"Label {lbl.label_id} evidence scenario_id '{ev.scenario_id}' "
                        f"mismatches scenario '{sc.scenario_id}'"
                    )
                if lbl.is_fraud and ev.typology != lbl.fraud_type:
                    raise ValueError(
                        f"Label {lbl.label_id} evidence typology '{ev.typology}' "
                        f"mismatches label fraud_type '{lbl.fraud_type}'"
                    )

        return self

