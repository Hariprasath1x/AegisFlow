"""Pattern generator for legitimate_clean (Negative Controls) scenario."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import numpy as np

from aegis_ml.scenarios.constants import (
    MANIFEST_REFERENCE_AT,
    MATURITY_BOUNDARY,
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


def generate_legitimate_clean(rng: np.random.Generator) -> ScenarioDefinition:
    """Generate the legitimate_clean negative control scenario for 8 accounts.

    Generates legitimate historical transactions across standard retail channels.
    Produces explicit clean account-level labels and explicit clean transaction-level
    labels with confidence=1.0 and fraud_type=None.
    """
    scenario_id = "legitimate_clean"
    typology = TYPOLOGY_REGISTRY[scenario_id]  # None

    accounts: list[SyntheticAccount] = []
    transactions: list[SyntheticTransaction] = []
    labels: list[SyntheticLabel] = []

    txn_global_idx = 0
    lbl_global_idx = 0

    base_date = datetime(2025, 1, 1, 9, 0, 0, tzinfo=timezone.utc)

    for acc_idx in range(8):
        acc_id = synthetic_uuid("account", scenario_id, acc_idx)
        acc_type = ["current", "savings", "card", "wallet"][acc_idx % 4]
        opened_str = f"2024-{(acc_idx % 12) + 1:02d}-01"
        account = SyntheticAccount(
            account_id=acc_id,
            scenario_id=scenario_id,
            account_number=f"AF-M2-SYN-CLN-ACC-{acc_idx:04d}",
            account_type=acc_type,  # type: ignore[arg-type]
            currency="INR",
            opened_on=opened_str,
            status="active",
            is_mule=False,
        )
        accounts.append(account)

        # 10 to 20 transactions per clean account
        num_txns = int(rng.integers(10, 21))
        acc_txns: list[SyntheticTransaction] = []

        for t_idx in range(num_txns):
            # Spread over 14 months, keeping well before MATURITY_BOUNDARY (2026-03-25)
            day_offset = int(rng.integers(0, 420))
            txn_time = base_date + timedelta(
                days=day_offset,
                hours=int(rng.integers(8, 22)),
                minutes=int(rng.integers(0, 60)),
            )
            # Ensure within maturity boundary
            if txn_time > MATURITY_BOUNDARY:
                txn_time = MATURITY_BOUNDARY - timedelta(days=int(rng.integers(1, 10)), hours=int(rng.integers(1, 12)))

            amt_val = float(rng.uniform(200, 50000))
            amt = canonical_amount(amt_val)
            channel = str(rng.choice(["card", "upi", "ach", "atm"]))

            t_id = synthetic_uuid("txn", scenario_id, txn_global_idx)
            txn = SyntheticTransaction(
                txn_id=t_id,
                external_ref=f"AF-M2-SYN-CLN-{txn_global_idx:06d}",
                account_id=acc_id,
                scenario_tag="CLN",
                channel=channel,  # type: ignore[arg-type]
                direction="debit" if t_idx % 3 != 0 else "credit",
                amount=amt,
                currency="INR",
                occurred_at=txn_time,
                destination_account_id=None,
                device_id=f"DEV-CLEAN-{acc_idx:03d}",
            )
            transactions.append(txn)
            acc_txns.append(txn)
            txn_global_idx += 1

        evidence = EvidenceProvenance(
            synthetic=True,
            scenario_id=scenario_id,
            typology=None,
            investigation_ref=f"SYN-INV-CLN-{acc_idx + 1:04d}",
            alert_refs=[],
            dispute_ref=None,
            analyst_id="SYN-ANALYST-006",
            confirmation_event=f"Full KYC verification SYN-INV-CLN-{acc_idx + 1:04d}; transaction pattern review — no anomaly",
            manifest_reference_at=MANIFEST_REFERENCE_AT,
        )

        # 1. Explicit clean account label (timeless representation of observation horizon validity)
        labels.append(
            SyntheticLabel(
                label_id=synthetic_uuid("label", scenario_id, lbl_global_idx),
                account_id=acc_id,
                label_target="account",
                is_fraud=False,
                fraud_type=None,
                confidence=1.0,
                source="synthetic_seed",
                incident_start_at=None,
                incident_end_at=None,
                labeled_at=OBSERVATION_CUTOFF,
                evidence=evidence,
            )
        )
        lbl_global_idx += 1

        # 2. Explicit clean transaction labels for 3 to 5 transactions (e.g. highest amount)
        acc_txns.sort(key=lambda x: x.amount, reverse=True)
        num_clean_txns = min(len(acc_txns), int(rng.integers(3, 6)))
        for c_txn in acc_txns[:num_clean_txns]:
            labels.append(
                SyntheticLabel(
                    label_id=synthetic_uuid("label", scenario_id, lbl_global_idx),
                    txn_id=c_txn.txn_id,
                    label_target="transaction",
                    is_fraud=False,
                    fraud_type=None,
                    confidence=1.0,
                    source="synthetic_seed",
                    incident_start_at=None,
                    incident_end_at=None,
                    labeled_at=OBSERVATION_CUTOFF,
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

