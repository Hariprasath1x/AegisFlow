"""Pattern generator for card_fraud (Card-Not-Present) scenario."""

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


def generate_card_fraud(rng: np.random.Generator) -> ScenarioDefinition:
    """Generate the card_fraud scenario for 6 victim card accounts.

    Generates legitimate card activity (unlabeled) alongside fraudulent CNP
    transactions (labeled with direct transaction fraud labels).
    Uses label_target='transaction' exclusively — zero account-level fraud labels.
    """
    scenario_id = "card_fraud"
    typology = TYPOLOGY_REGISTRY[scenario_id]

    accounts: list[SyntheticAccount] = []
    transactions: list[SyntheticTransaction] = []
    labels: list[SyntheticLabel] = []

    txn_global_idx = 0
    lbl_global_idx = 0

    base_time = datetime(2025, 11, 1, 10, 0, 0, tzinfo=timezone.utc)

    for acc_idx in range(6):
        acc_id = synthetic_uuid("account", scenario_id, acc_idx)
        account = SyntheticAccount(
            account_id=acc_id,
            scenario_id=scenario_id,
            account_number=f"AF-M2-SYN-CNP-ACC-{acc_idx:04d}",
            account_type="card",
            currency="INR",
            opened_on="2024-06-20",
            status="active",
            is_mule=False,
        )
        accounts.append(account)

        # 1. Legitimate card transactions (5 to 8 transactions)
        num_legit = int(rng.integers(5, 9))
        for l_idx in range(num_legit):
            l_time = base_time + timedelta(days=l_idx * 2 + acc_idx * 2, hours=int(rng.integers(1, 10)))
            amt_val = float(rng.uniform(500, 15000))
            amt = canonical_amount(amt_val)

            t_id = synthetic_uuid("txn", scenario_id, txn_global_idx)
            transactions.append(
                SyntheticTransaction(
                    txn_id=t_id,
                    external_ref=f"AF-M2-SYN-CNP-{txn_global_idx:06d}",
                    account_id=acc_id,
                    scenario_tag="CNP",
                    channel="card",
                    direction="debit",
                    amount=amt,
                    currency="INR",
                    occurred_at=l_time,
                    destination_account_id=None,
                    device_id=f"DEV-CARDHOLDER-{acc_idx:03d}",
                )
            )
            txn_global_idx += 1

        # 2. Fraudulent Card-Not-Present transactions (2 to 4 transactions)
        fraud_start_time = base_time + timedelta(days=20 + acc_idx * 3, hours=1)
        num_fraud = int(rng.integers(2, 5))
        fraud_txns: list[SyntheticTransaction] = []

        for f_idx in range(num_fraud):
            f_time = fraud_start_time + timedelta(hours=f_idx * 8 + int(rng.integers(1, 4)))
            amt_val = float(rng.uniform(5000, 80000))
            amt = canonical_amount(amt_val)

            t_id = synthetic_uuid("txn", scenario_id, txn_global_idx)
            txn = SyntheticTransaction(
                txn_id=t_id,
                external_ref=f"AF-M2-SYN-CNP-{txn_global_idx:06d}",
                account_id=acc_id,
                scenario_tag="CNP",
                channel="card",
                direction="debit",
                amount=amt,
                currency="INR",
                occurred_at=f_time,
                destination_account_id=None,
                device_id=f"DEV-CNP-ATTACKER-{acc_idx:03d}",
            )
            transactions.append(txn)
            fraud_txns.append(txn)
            txn_global_idx += 1

        # 3. Direct Transaction Labels
        first_fraud_time = fraud_txns[0].occurred_at
        last_fraud_time = fraud_txns[-1].occurred_at

        incident_start = datetime(
            first_fraud_time.year, first_fraud_time.month, first_fraud_time.day, 0, 0, 0, tzinfo=timezone.utc
        )
        # Chargeback filed 7 days after the last fraudulent transaction
        incident_end = datetime(
            last_fraud_time.year, last_fraud_time.month, last_fraud_time.day, 23, 59, 59, tzinfo=timezone.utc
        ) + timedelta(days=7)
        labeled_at = incident_end + timedelta(days=1)

        evidence = EvidenceProvenance(
            synthetic=True,
            scenario_id=scenario_id,
            typology=typology,
            investigation_ref=f"SYN-INV-CNP-{acc_idx + 1:04d}",
            alert_refs=[f"SYN-ALERT-CNP-{acc_idx + 1:04d}"],
            dispute_ref=f"SYN-DISP-CNP-{acc_idx + 1:04d}",
            analyst_id="SYN-ANALYST-004",
            confirmation_event=f"Chargeback raised by cardholder; CNP transaction confirmed at SYN-DISP-CNP-{acc_idx + 1:04d}",
            manifest_reference_at=MANIFEST_REFERENCE_AT,
        )

        for f_txn in fraud_txns:
            labels.append(
                SyntheticLabel(
                    label_id=synthetic_uuid("label", scenario_id, lbl_global_idx),
                    txn_id=f_txn.txn_id,
                    label_target="transaction",
                    is_fraud=True,
                    fraud_type=typology,
                    confidence=1.0,
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

