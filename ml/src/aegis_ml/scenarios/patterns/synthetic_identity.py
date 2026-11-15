"""Pattern generator for synthetic_identity (First-Party Synthetic) scenario."""

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


def generate_synthetic_identity(rng: np.random.Generator) -> ScenarioDefinition:
    """Generate the synthetic_identity scenario for 4 synthetic accounts.

    Generates seasoning history (unlabeled) followed by bust-out wire transfers
    (directly labeled fraud) with account-level incident windows spanning
    from account opening to bust-out completion.
    """
    scenario_id = "synthetic_identity"
    typology = TYPOLOGY_REGISTRY[scenario_id]

    accounts: list[SyntheticAccount] = []
    transactions: list[SyntheticTransaction] = []
    labels: list[SyntheticLabel] = []

    txn_global_idx = 0
    lbl_global_idx = 0

    for acc_idx in range(4):
        opened_date = datetime(2025, 1, 10, 0, 0, 0, tzinfo=timezone.utc)
        acc_id = synthetic_uuid("account", scenario_id, acc_idx)
        account = SyntheticAccount(
            account_id=acc_id,
            scenario_id=scenario_id,
            account_number=f"AF-M2-SYN-FPS-ACC-{acc_idx:04d}",
            account_type="current" if acc_idx % 2 == 0 else "savings",
            currency="INR",
            opened_on=opened_date.strftime("%Y-%m-%d"),
            status="active",
            is_mule=False,
        )
        accounts.append(account)

        # 1. Seasoning transactions (10 to 14 transactions)
        num_seasoning = int(rng.integers(10, 15))
        for s_idx in range(num_seasoning):
            s_time = opened_date + timedelta(days=10 + s_idx * 18 + acc_idx * 2, hours=int(rng.integers(8, 20)))
            amt_val = float(rng.uniform(1000, 25000))
            amt = canonical_amount(amt_val)
            channel = str(rng.choice(["card", "upi", "ach"]))

            t_id = synthetic_uuid("txn", scenario_id, txn_global_idx)
            transactions.append(
                SyntheticTransaction(
                    txn_id=t_id,
                    external_ref=f"AF-M2-SYN-FPS-{txn_global_idx:06d}",
                    account_id=acc_id,
                    scenario_tag="FPS",
                    channel=channel,  # type: ignore[arg-type]
                    direction="debit",
                    amount=amt,
                    currency="INR",
                    occurred_at=s_time,
                    destination_account_id=None,
                    device_id=f"DEV-SYNTHETIC-{acc_idx:03d}",
                )
            )
            txn_global_idx += 1

        # 2. Large wire-out transactions at end of history (1 to 2 transactions)
        num_bust_out = int(rng.integers(1, 3))
        bust_out_start = datetime(2025, 10, 15, 10, 0, 0, tzinfo=timezone.utc) + timedelta(days=acc_idx * 3)
        bust_out_txns: list[SyntheticTransaction] = []

        for b_idx in range(num_bust_out):
            b_time = bust_out_start + timedelta(hours=b_idx * 12 + int(rng.integers(1, 4)))
            amt_val = float(rng.uniform(100000, 200000))
            amt = canonical_amount(amt_val)

            t_id = synthetic_uuid("txn", scenario_id, txn_global_idx)
            txn = SyntheticTransaction(
                txn_id=t_id,
                external_ref=f"AF-M2-SYN-FPS-{txn_global_idx:06d}",
                account_id=acc_id,
                scenario_tag="FPS",
                channel="wire",
                direction="debit",
                amount=amt,
                currency="INR",
                occurred_at=b_time,
                destination_account_id=None,
                device_id=f"DEV-SYNTHETIC-{acc_idx:03d}",
            )
            transactions.append(txn)
            bust_out_txns.append(txn)
            txn_global_idx += 1

        # Incident window spans from account opening through final bust-out
        incident_start = opened_date
        last_bust_out_time = bust_out_txns[-1].occurred_at
        incident_end = datetime(
            last_bust_out_time.year,
            last_bust_out_time.month,
            last_bust_out_time.day,
            23,
            59,
            59,
            tzinfo=timezone.utc,
        )
        labeled_at = incident_end + timedelta(days=5)

        evidence = EvidenceProvenance(
            synthetic=True,
            scenario_id=scenario_id,
            typology=typology,
            investigation_ref=f"SYN-INV-FPS-{acc_idx + 1:04d}",
            alert_refs=[f"SYN-ALERT-FPS-{acc_idx + 1:04d}"],
            dispute_ref=None,
            analyst_id="SYN-ANALYST-003",
            confirmation_event=f"Identity verification failure at account opening; synthetic document cluster SYN-INV-FPS-{acc_idx + 1:04d}",
            manifest_reference_at=MANIFEST_REFERENCE_AT,
        )

        # Account-level label (bounded by full synthetic lifecycle)
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

        # Direct transaction labels for large wire-out transfers
        for b_txn in bust_out_txns:
            labels.append(
                SyntheticLabel(
                    label_id=synthetic_uuid("label", scenario_id, lbl_global_idx),
                    txn_id=b_txn.txn_id,
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

