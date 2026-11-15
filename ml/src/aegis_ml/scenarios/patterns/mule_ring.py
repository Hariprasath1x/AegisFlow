"""Pattern generator for mule_ring (Money Mule Network) scenario."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import numpy as np

from aegis_ml.scenarios.constants import (
    MANIFEST_REFERENCE_AT,
    OBSERVATION_CUTOFF,
    TYPOLOGY_REGISTRY,
    canonical_amount,
    synthetic_uuid,
)
from aegis_ml.scenarios.models import (
    EvidenceProvenance,
    ScenarioDefinition,
    SyntheticAccount,
    SyntheticLabel,
    SyntheticTransaction,
)


def generate_mule_ring(rng: np.random.Generator) -> ScenarioDefinition:
    """Generate the mule_ring scenario with 1 controller and 5 mule accounts.

    All 6 accounts receive account-level fraud labels. Zero transaction-level
    labels are generated, testing the invariant that account-level fraud does
    not automatically propagate to transaction targets.
    """
    scenario_id = "mule_ring"
    typology = TYPOLOGY_REGISTRY[scenario_id]

    # 1. Accounts
    accounts: list[SyntheticAccount] = []
    # Controller account
    controller_id = synthetic_uuid("account", scenario_id, 0)
    accounts.append(
        SyntheticAccount(
            account_id=controller_id,
            scenario_id=scenario_id,
            account_number="AF-M2-SYN-MM-ACC-0000",
            account_type="current",
            currency="INR",
            opened_on="2024-01-15",
            status="active",
            is_mule=False,
        )
    )

    # 5 Mule accounts
    mule_accounts: list[SyntheticAccount] = []
    for m_idx in range(1, 6):
        m_id = synthetic_uuid("account", scenario_id, m_idx)
        acc = SyntheticAccount(
            account_id=m_id,
            scenario_id=scenario_id,
            account_number=f"AF-M2-SYN-MM-ACC-{m_idx:04d}",
            account_type="savings",
            currency="INR",
            opened_on="2024-03-01",
            status="active",
            is_mule=True,
        )
        accounts.append(acc)
        mule_accounts.append(acc)

    # 2. Transactions
    transactions: list[SyntheticTransaction] = []
    txn_global_idx = 0

    base_time = datetime(2025, 6, 1, 8, 0, 0, tzinfo=timezone.utc)
    inbound_times: list[datetime] = []
    outbound_times: list[datetime] = []

    for m_idx, mule in enumerate(mule_accounts):
        # 10 inbound transfers from controller to this mule
        for i_idx in range(10):
            t_offset_hours = m_idx * 24 + i_idx * 3 + int(rng.integers(0, 2))
            txn_time = base_time + timedelta(hours=t_offset_hours)
            inbound_times.append(txn_time)

            # Inbound amount between ₹50,000 and ₹400,000
            amt_val = float(rng.uniform(50000, 400000))
            amt = canonical_amount(amt_val)

            txn_id = synthetic_uuid("txn", scenario_id, txn_global_idx)
            transactions.append(
                SyntheticTransaction(
                    txn_id=txn_id,
                    external_ref=f"AF-M2-SYN-MM-{txn_global_idx:06d}",
                    account_id=controller_id,
                    scenario_tag="MM",
                    channel="transfer",
                    direction="debit",
                    amount=amt,
                    currency="INR",
                    occurred_at=txn_time,
                    destination_account_id=mule.account_id,
                    device_id="DEV-CONTROLLER-001",
                )
            )
            txn_global_idx += 1

            # Rapid outbound wire transfer from mule to external counterparty
            out_offset_hours = t_offset_hours + int(rng.integers(1, 6))
            out_time = base_time + timedelta(hours=out_offset_hours)
            outbound_times.append(out_time)

            out_amt_val = float(amt) * float(rng.uniform(0.85, 0.98))
            out_amt = canonical_amount(out_amt_val)

            out_txn_id = synthetic_uuid("txn", scenario_id, txn_global_idx)
            transactions.append(
                SyntheticTransaction(
                    txn_id=out_txn_id,
                    external_ref=f"AF-M2-SYN-MM-{txn_global_idx:06d}",
                    account_id=mule.account_id,
                    scenario_tag="MM",
                    channel="wire",
                    direction="debit",
                    amount=out_amt,
                    currency="INR",
                    occurred_at=out_time,
                    destination_account_id=None,
                    device_id=f"DEV-MULE-{m_idx + 1:03d}",
                )
            )
            txn_global_idx += 1

    # 3. Ground Truth Labels
    labels: list[SyntheticLabel] = []
    lbl_idx = 0

    first_inbound = min(inbound_times)
    last_inbound = max(inbound_times)
    first_outbound = min(outbound_times)
    last_outbound = max(outbound_times)

    # Controller account label (bounded by dispersal activity)
    controller_start = datetime(first_inbound.year, first_inbound.month, first_inbound.day, 0, 0, 0, tzinfo=timezone.utc)
    controller_end = datetime(last_inbound.year, last_inbound.month, last_inbound.day, 23, 59, 59, tzinfo=timezone.utc)
    controller_labeled = controller_end + timedelta(days=7)

    controller_evidence = EvidenceProvenance(
        synthetic=True,
        scenario_id=scenario_id,
        typology=typology,
        investigation_ref="SYN-INV-MM-0001",
        alert_refs=["SYN-ALERT-MM-0001"],
        dispute_ref=None,
        analyst_id="SYN-ANALYST-001",
        confirmation_event="Controller account identified as mastermind orchestrator in network takedown SYN-INV-MM-0001; fund dispersal patterns confirmed.",
        manifest_reference_at=MANIFEST_REFERENCE_AT,
    )

    labels.append(
        SyntheticLabel(
            label_id=synthetic_uuid("label", scenario_id, lbl_idx),
            account_id=controller_id,
            label_target="account",
            is_fraud=True,
            fraud_type=typology,
            confidence=1.0,
            source="synthetic_seed",
            incident_start_at=controller_start,
            incident_end_at=controller_end,
            labeled_at=controller_labeled,
            evidence=controller_evidence,
        )
    )
    lbl_idx += 1

    # 5 Mule account labels (bounded by recipient activity)
    mule_start = datetime(first_inbound.year, first_inbound.month, first_inbound.day, 0, 0, 0, tzinfo=timezone.utc)
    mule_end = datetime(last_outbound.year, last_outbound.month, last_outbound.day, 23, 59, 59, tzinfo=timezone.utc)
    mule_labeled = mule_end + timedelta(days=7)

    mule_evidence = EvidenceProvenance(
        synthetic=True,
        scenario_id=scenario_id,
        typology=typology,
        investigation_ref="SYN-INV-MM-0001",
        alert_refs=["SYN-ALERT-MM-0002"],
        dispute_ref=None,
        analyst_id="SYN-ANALYST-001",
        confirmation_event="Account identified as mule in network takedown SYN-INV-MM-0001; fund flow analysis confirmed.",
        manifest_reference_at=MANIFEST_REFERENCE_AT,
    )

    for mule in mule_accounts:
        labels.append(
            SyntheticLabel(
                label_id=synthetic_uuid("label", scenario_id, lbl_idx),
                account_id=mule.account_id,
                label_target="account",
                is_fraud=True,
                fraud_type=typology,
                confidence=1.0,
                source="synthetic_seed",
                incident_start_at=mule_start,
                incident_end_at=mule_end,
                labeled_at=mule_labeled,
                evidence=mule_evidence,
            )
        )
        lbl_idx += 1

    return ScenarioDefinition(
        scenario_id=scenario_id,
        typology=typology,
        accounts=accounts,
        transactions=transactions,
        labels=labels,
    )

