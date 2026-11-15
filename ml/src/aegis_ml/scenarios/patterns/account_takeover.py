"""Pattern generator for account_takeover scenario."""

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


def generate_account_takeover(rng: np.random.Generator) -> ScenarioDefinition:
    """Generate the account_takeover scenario for 3 victim accounts.

    Generates pre-takeover normal transactions (unlabeled) and post-takeover
    compromised transactions (labeled with direct transaction fraud labels
    and bounded account-level fraud labels).
    """
    scenario_id = "account_takeover"
    typology = TYPOLOGY_REGISTRY[scenario_id]

    accounts: list[SyntheticAccount] = []
    transactions: list[SyntheticTransaction] = []
    labels: list[SyntheticLabel] = []

    txn_global_idx = 0
    lbl_global_idx = 0

    base_time = datetime(2025, 8, 1, 9, 0, 0, tzinfo=timezone.utc)

    for acc_idx in range(3):
        acc_id = synthetic_uuid("account", scenario_id, acc_idx)
        account = SyntheticAccount(
            account_id=acc_id,
            scenario_id=scenario_id,
            account_number=f"AF-M2-SYN-ATO-ACC-{acc_idx:04d}",
            account_type="savings" if acc_idx % 2 == 0 else "current",
            currency="INR",
            opened_on="2024-05-10",
            status="active",
            is_mule=False,
        )
        accounts.append(account)

        # 1. Pre-takeover legitimate transactions (5 to 8 transactions)
        num_pre = int(rng.integers(5, 9))
        for p_idx in range(num_pre):
            p_time = base_time + timedelta(days=p_idx * 2 + acc_idx * 3, hours=int(rng.integers(1, 8)))
            amt_val = float(rng.uniform(500, 20000))
            amt = canonical_amount(amt_val)
            channel = str(rng.choice(["card", "upi"]))

            t_id = synthetic_uuid("txn", scenario_id, txn_global_idx)
            transactions.append(
                SyntheticTransaction(
                    txn_id=t_id,
                    external_ref=f"AF-M2-SYN-ATO-{txn_global_idx:06d}",
                    account_id=acc_id,
                    scenario_tag="ATO",
                    channel=channel,  # type: ignore[arg-type]
                    direction="debit",
                    amount=amt,
                    currency="INR",
                    occurred_at=p_time,
                    destination_account_id=None,
                    device_id=f"DEV-VICTIM-{acc_idx:03d}",
                )
            )
            txn_global_idx += 1

        # 2. Takeover event & Post-takeover fraudulent transactions (3 to 6 transactions)
        takeover_start_time = base_time + timedelta(days=20 + acc_idx * 4, hours=2)
        num_post = int(rng.integers(3, 7))
        post_txns: list[SyntheticTransaction] = []

        for po_idx in range(num_post):
            po_time = takeover_start_time + timedelta(hours=po_idx * 6 + int(rng.integers(1, 3)))
            amt_val = float(rng.uniform(80000, 450000))
            amt = canonical_amount(amt_val)
            channel = str(rng.choice(["wire", "transfer"]))

            t_id = synthetic_uuid("txn", scenario_id, txn_global_idx)
            txn = SyntheticTransaction(
                txn_id=t_id,
                external_ref=f"AF-M2-SYN-ATO-{txn_global_idx:06d}",
                account_id=acc_id,
                scenario_tag="ATO",
                channel=channel,  # type: ignore[arg-type]
                direction="debit",
                amount=amt,
                currency="INR",
                occurred_at=po_time,
                destination_account_id=None,
                device_id=f"DEV-ATTACKER-{acc_idx:03d}",
            )
            transactions.append(txn)
            post_txns.append(txn)
            txn_global_idx += 1

        # Incident window covers post-takeover activity
        first_post_time = post_txns[0].occurred_at
        last_post_time = post_txns[-1].occurred_at

        incident_start = datetime(
            first_post_time.year, first_post_time.month, first_post_time.day, 0, 0, 0, tzinfo=timezone.utc
        )
        incident_end = datetime(
            last_post_time.year, last_post_time.month, last_post_time.day, 23, 59, 59, tzinfo=timezone.utc
        )
        labeled_at = incident_end + timedelta(days=3)

        evidence = EvidenceProvenance(
            synthetic=True,
            scenario_id=scenario_id,
            typology=typology,
            investigation_ref=f"SYN-INV-ATO-{acc_idx + 1:04d}",
            alert_refs=[f"SYN-ALERT-ATO-{acc_idx + 1:04d}"],
            dispute_ref=f"SYN-DISP-ATO-{acc_idx + 1:04d}",
            analyst_id="SYN-ANALYST-002",
            confirmation_event=f"Victim reported unauthorized access; device fingerprint change confirmed at SYN-ALERT-ATO-{acc_idx + 1:04d}",
            manifest_reference_at=MANIFEST_REFERENCE_AT,
        )

        # Account-level label (bounded by takeover window)
        labels.append(
            SyntheticLabel(
                label_id=synthetic_uuid("label", scenario_id, lbl_global_idx),
                account_id=acc_id,
                label_target="account",
                is_fraud=True,
                fraud_type=typology,
                confidence=0.95,
                source="synthetic_seed",
                incident_start_at=incident_start,
                incident_end_at=incident_end,
                labeled_at=labeled_at,
                evidence=evidence,
            )
        )
        lbl_global_idx += 1

        # Direct transaction labels for post-takeover transactions
        for p_txn in post_txns:
            labels.append(
                SyntheticLabel(
                    label_id=synthetic_uuid("label", scenario_id, lbl_global_idx),
                    txn_id=p_txn.txn_id,
                    label_target="transaction",
                    is_fraud=True,
                    fraud_type=typology,
                    confidence=0.95,
                    source="synthetic_seed",
                    incident_start_at=incident_start,
                    incident_end_at=incident_end,
                    labeled_at=labeled_at,
                    evidence=evidence,
                )
            )
            lbl_global_idx += 1

    return ScenarioDefinition(
        scenario_id=scenario_id,
        typology=typology,
        accounts=accounts,
        transactions=transactions,
        labels=labels,
    )

