"""AegisFlow M2 Step 2.10 — Deterministic Synthetic Known-Fraud Scenario Generation."""

from __future__ import annotations

from aegis_ml.scenarios.constants import (
    MANIFEST_REFERENCE_AT,
    MATURITY_BOUNDARY,
    OBSERVATION_CUTOFF,
    RETAINED_TYPOLOGIES,
    SCENARIO_NAMESPACE,
    SCENARIO_ORDER,
    SEED,
    TYPOLOGY_REGISTRY,
    canonical_amount,
    make_pattern_rngs,
    synthetic_uuid,
    utc_z,
)
from aegis_ml.scenarios.generator import (
    ManifestGenerator,
    canonical_json_bytes,
    compute_manifest_hash,
    finalize_manifest,
    verify_manifest_hash,
)
from aegis_ml.scenarios.models import (
    EvidenceProvenance,
    ScenarioDefinition,
    ScenarioManifest,
    SyntheticAccount,
    SyntheticLabel,
    SyntheticTransaction,
)

__all__ = [
    "EvidenceProvenance",
    "MANIFEST_REFERENCE_AT",
    "MATURITY_BOUNDARY",
    "ManifestGenerator",
    "OBSERVATION_CUTOFF",
    "RETAINED_TYPOLOGIES",
    "SCENARIO_NAMESPACE",
    "SCENARIO_ORDER",
    "SEED",
    "ScenarioDefinition",
    "ScenarioManifest",
    "SyntheticAccount",
    "SyntheticLabel",
    "SyntheticTransaction",
    "TYPOLOGY_REGISTRY",
    "canonical_amount",
    "canonical_json_bytes",
    "compute_manifest_hash",
    "finalize_manifest",
    "make_pattern_rngs",
    "synthetic_uuid",
    "utc_z",
    "verify_manifest_hash",
]

