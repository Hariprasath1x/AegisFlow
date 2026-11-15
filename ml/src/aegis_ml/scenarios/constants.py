"""Canonical constants and deterministic derivation helpers for AegisFlow M2 scenarios."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Mapping, Sequence
import uuid

import numpy as np


# Global Seed
SEED: int = 20261010

# Fixed RFC 4122 namespace for all synthetic scenario UUIDs
SCENARIO_NAMESPACE: uuid.UUID = uuid.uuid5(
    uuid.NAMESPACE_DNS,
    "aegisflow.scenarios.v1.seed=20261010",
)

# Observation Cutoff and Maturity Boundary
OBSERVATION_CUTOFF: datetime = datetime(2026, 4, 1, 0, 0, 0, tzinfo=timezone.utc)
MATURITY_BUFFER_DAYS: int = 7
MATURITY_BOUNDARY: datetime = OBSERVATION_CUTOFF - timedelta(days=MATURITY_BUFFER_DAYS)

# Pinned Manifest Reference Timestamp (not actual execution wall-clock time)
MANIFEST_REFERENCE_AT: datetime = datetime(2026, 4, 1, 0, 0, 0, tzinfo=timezone.utc)

# Canonical Scenario Order (fixed sequence index mapped to SeedSequence child sequences)
SCENARIO_ORDER: tuple[str, ...] = (
    "mule_ring",
    "account_takeover",
    "synthetic_identity",
    "card_fraud",
    "structuring",
    "legitimate_clean",
)

# Authoritative Typology Registry mapping scenario_id -> fraud_type DB value
TYPOLOGY_REGISTRY: dict[str, str | None] = {
    "mule_ring": "money_mule",
    "account_takeover": "account_takeover",
    "synthetic_identity": "first_party_synthetic",
    "card_fraud": "card_not_present",
    "structuring": "structuring",
    "legitimate_clean": None,
}

# Retained typologies for future scenarios
RETAINED_TYPOLOGIES: frozenset[str] = frozenset({"velocity_spike", "bust_out"})


def synthetic_uuid(entity_type: str, scenario_id: str, index: int) -> uuid.UUID:
    """Generate a deterministic UUIDv5 from entity type, scenario ID, and index."""
    return uuid.uuid5(SCENARIO_NAMESPACE, f"{entity_type}:{scenario_id}:{index}")


def canonical_amount(value: int | float | str | Decimal) -> Decimal:
    """Normalize monetary amount to exactly 2 decimal places (INR)."""
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def utc_z(dt: datetime) -> str:
    """Serialize a timezone-aware UTC datetime as ISO 8601 with Z suffix.

    Rejects naive datetimes and non-UTC timezone offsets.
    """
    if dt.tzinfo is None:
        raise ValueError(f"Expected timezone-aware UTC datetime, got naive: {dt!r}")
    offset = dt.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError(f"Expected UTC timezone with offset 0, got offset {offset} for {dt!r}")
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def make_pattern_rngs(
    seed: int = SEED,
    scenario_order: Sequence[str] = SCENARIO_ORDER,
) -> dict[str, np.random.Generator]:
    """Derive one Generator per scenario from a shared SeedSequence.

    Guarantees:
    1. Stable Scenario-to-Sequence Mapping: Each scenario in scenario_order is
       mapped to child_sequences[i] strictly by its integer index i.
    2. Execution Order Independence: Generators can be consumed in any order
       without altering the output numbers of any scenario.
    3. Append Safety: Adding scenarios to the end of scenario_order preserves
       the child sequences and streams for indices 0..N-1.
    4. Python hash()-free: Child sequences are produced via NumPy's 128-bit
       SeedSequence integer hashing.
    """
    ss = np.random.SeedSequence(seed)
    child_sequences = ss.spawn(len(scenario_order))
    return {
        scenario_id: np.random.default_rng(child_seq)
        for scenario_id, child_seq in zip(scenario_order, child_sequences)
    }

