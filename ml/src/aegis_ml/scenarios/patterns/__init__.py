"""Pattern generators for the six AegisFlow M2 known-fraud scenarios."""

from __future__ import annotations

from aegis_ml.scenarios.patterns.account_takeover import generate_account_takeover
from aegis_ml.scenarios.patterns.card_fraud import generate_card_fraud
from aegis_ml.scenarios.patterns.legitimate_clean import generate_legitimate_clean
from aegis_ml.scenarios.patterns.mule_ring import generate_mule_ring
from aegis_ml.scenarios.patterns.structuring import StructuringParams, generate_structuring
from aegis_ml.scenarios.patterns.synthetic_identity import generate_synthetic_identity

__all__ = [
    "StructuringParams",
    "generate_account_takeover",
    "generate_card_fraud",
    "generate_legitimate_clean",
    "generate_mule_ring",
    "generate_structuring",
    "generate_synthetic_identity",
]

