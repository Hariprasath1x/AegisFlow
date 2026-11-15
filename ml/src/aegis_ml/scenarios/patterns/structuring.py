"""Pattern generator for structuring scenario."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import TYPE_CHECKING
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

if TYPE_CHECKING:
    from libs.aegis_domain.rules import RuleSet


@dataclass(frozen=True)
class StructuringParams:
    """Configurable parameters for simulated structuring scenario generation.

    These values model the simulated detection criteria of rule R003_STRUCTURING
    in AegisFlow's rule engine (libs/aegis_domain/rules.py).

    IMPORTANT DISTINCTION:
    - These values are SIMULATED detection parameters used to generate realistic synthetic
      behavior that triggers system alerts.
    - They do NOT represent a universal legal reporting threshold or statutory definition.
    - Hitting this threshold does not constitute independent proof of fraud; the confirmed
      ground-truth status is established by the simulated investigation reference (SYN-INV-STR-*).
    - Currency is strictly INR (matching transactions.currency).
    """

    structuring_report_limit: Decimal
    single_txn_threshold: Decimal
    window_hours: int
    min_sub_threshold_count: int
    currency: str = "INR"

    @classmethod
    def from_rule_set(cls, rule_set: RuleSet | None = None) -> StructuringParams:
        """Factory obtaining thresholds via explicit dependency injection from domain RuleSet."""
        if rule_set is None:
            from libs.aegis_domain.rules import RuleSet as DomainRuleSet

            rule_set = DomainRuleSet()

        return cls(
            structuring_report_limit=rule_set.structuring_report_limit,  # Decimal('1000000')
            single_txn_threshold=rule_set.high_value_threshold,          # Decimal('500000')
            window_hours=rule_set.structuring_window_hours,              # 24
            min_sub_threshold_count=3,                                   # RuleSet.evaluate: vel.sub_threshold_count_24h >= 3
            currency="INR",
        )


def generate_structuring(
    rng: np.random.Generator,
    params: StructuringParams | None = None,
) -> ScenarioDefinition:
    """Generate the structuring scenario for 3 accounts.

    Generates sub-threshold transactions meeting rule R003_STRUCTURING detection
    criteria (count >= 3 sub-threshold transactions within 24 hours summing to >=
    structuring_report_limit).
    """
    if params is None:
        params = StructuringParams.from_rule_set()

    scenario_id = "structuring"
    typology = TYPOLOGY_REGISTRY[scenario_id]

    accounts: list[SyntheticAccount] = []
    transactions: list[SyntheticTransaction] = []
    labels: list[SyntheticLabel] = []

    txn_global_idx = 0
    lbl_global_idx = 0

    base_date = datetime(2025, 12, 5, 0, 0, 0, tzinfo=timezone.utc)

    for acc_idx in range(3):
        acc_id = synthetic_uuid("account", scenario_id, acc_idx)
        account = SyntheticAccount(
            account_id=acc_id,
            scenario_id=scenario_id,
            account_number=f"AF-M2-SYN-STR-ACC-{acc_idx:04d}",
            account_type="current" if acc_idx % 2 == 0 else "savings",
            currency="INR",
            opened_on="2024-07-15",
            status="active",
            is_mule=False,
        )
        accounts.append(account)

        # 1. Background transactions outside the structuring window (2 to 4 transactions)
        num_bg = int(rng.integers(2, 5))
        for bg_idx in range(num_bg):
            bg_time = base_date - timedelta(days=10 + bg_idx * 3, hours=int(rng.integers(2, 10)))
            amt_val = float(rng.uniform(2000, 30000))
            amt = canonical_amount(amt_val)

            t_id = synthetic_uuid("txn", scenario_id, txn_global_idx)
            transactions.append(
                SyntheticTransaction(
                    txn_id=t_id,
                    external_ref=f"AF-M2-SYN-STR-{txn_global_idx:06d}",
                    account_id=acc_id,
                    scenario_tag="STR",
                    channel="ach",
                    direction="debit",
                    amount=amt,
                    currency="INR",
                    occurred_at=bg_time,
                    destination_account_id=None,
                    device_id=f"DEV-STRUCTURING-{acc_idx:03d}",
                )
            )
            txn_global_idx += 1

        # 2. Structuring transaction cluster within 24 hours
        # Count must be >= min_sub_threshold_count (3)
        num_struct = int(rng.integers(params.min_sub_threshold_count, params.min_sub_threshold_count + 3))

        # Each amount must be < single_txn_threshold (₹500,000)
        # and sum of all must be >= structuring_report_limit (₹1,000,000)
        # To guarantee this: target sum between 1.05 and 1.30 of structuring_report_limit
        target_sum = float(params.structuring_report_limit) * float(rng.uniform(1.05, 1.30))
        # Ensure per-transaction amount does not exceed 95% of single_txn_threshold
        max_allowed_single = float(params.single_txn_threshold) * 0.95
        # Base share per transaction
        base_share = target_sum / num_struct
        if base_share > max_allowed_single:
            # Increase count if needed to keep individual transactions sub-threshold
            num_struct = int(np.ceil(target_sum / max_allowed_single)) + 1
            base_share = target_sum / num_struct

        cluster_start = base_date + timedelta(days=acc_idx * 5, hours=8)
        cluster_txns: list[SyntheticTransaction] = []

        running_sum = Decimal("0.00")
        for s_idx in range(num_struct):
            # Jitter amount slightly while staying strictly sub-threshold
            jitter = float(rng.uniform(-15000, 15000))
            amt_val = min(max_allowed_single, max(100000.0, base_share + jitter))
            amt = canonical_amount(amt_val)

            # Ensure total sum satisfies threshold on last transaction if running low
            if s_idx == num_struct - 1 and (running_sum + amt) < params.structuring_report_limit:
                needed = params.structuring_report_limit - running_sum + Decimal("10000.00")
                amt = canonical_amount(min(Decimal(str(max_allowed_single)), needed))

            running_sum += amt
            # Hour within the 24h window (0 to 12h from cluster_start, ensuring incident boundary compliance)
            hour_offset = int((s_idx * 12) / max(1, num_struct - 1))
            s_time = cluster_start + timedelta(hours=hour_offset, minutes=int(rng.integers(0, 50)))

            t_id = synthetic_uuid("txn", scenario_id, txn_global_idx)
            txn = SyntheticTransaction(
                txn_id=t_id,
                external_ref=f"AF-M2-SYN-STR-{txn_global_idx:06d}",
                account_id=acc_id,
                scenario_tag="STR",
                channel="wire" if s_idx % 2 == 0 else "ach",
                direction="debit",
                amount=amt,
                currency="INR",
                occurred_at=s_time,
                destination_account_id=None,
                device_id=f"DEV-STRUCTURING-{acc_idx:03d}",
            )
            transactions.append(txn)
            cluster_txns.append(txn)
            txn_global_idx += 1

        # 3. Ground Truth Label: Bounded account-level label covering the 24h structuring window
        incident_start = datetime(
            cluster_start.year, cluster_start.month, cluster_start.day, 0, 0, 0, tzinfo=timezone.utc
        )
        incident_end = incident_start + timedelta(hours=params.window_hours)
        labeled_at = incident_end + timedelta(days=3)

        evidence = EvidenceProvenance(
            synthetic=True,
            scenario_id=scenario_id,
            typology=typology,
            investigation_ref=f"SYN-INV-STR-{acc_idx + 1:04d}",
            alert_refs=[f"SYN-ALERT-STR-{acc_idx + 1:04d}"],
            dispute_ref=None,
            analyst_id="SYN-ANALYST-005",
            confirmation_event=f"Simulated STR filing analysis SYN-INV-STR-{acc_idx + 1:04d} confirmed sub-threshold structuring pattern consistent with rule R003_STRUCTURING",
            manifest_reference_at=MANIFEST_REFERENCE_AT,
        )

        labels.append(
            SyntheticLabel(
                label_id=synthetic_uuid("label", scenario_id, lbl_global_idx),
                account_id=acc_id,
                label_target="account",
                is_fraud=True,
                fraud_type=typology,
                confidence=0.90,
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
