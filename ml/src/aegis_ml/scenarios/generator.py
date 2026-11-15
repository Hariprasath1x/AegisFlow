"""Manifest generator and canonical hashing for AegisFlow M2 scenarios."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from aegis_ml.scenarios.constants import (
    MANIFEST_REFERENCE_AT,
    OBSERVATION_CUTOFF,
    SCENARIO_ORDER,
    SEED,
    make_pattern_rngs,
)
from aegis_ml.scenarios.models import ScenarioDefinition, ScenarioManifest
from aegis_ml.scenarios.patterns.account_takeover import generate_account_takeover
from aegis_ml.scenarios.patterns.card_fraud import generate_card_fraud
from aegis_ml.scenarios.patterns.legitimate_clean import generate_legitimate_clean
from aegis_ml.scenarios.patterns.mule_ring import generate_mule_ring
from aegis_ml.scenarios.patterns.structuring import StructuringParams, generate_structuring
from aegis_ml.scenarios.patterns.synthetic_identity import generate_synthetic_identity

if TYPE_CHECKING:
    from libs.aegis_domain.rules import RuleSet


def _manifest_dict_for_hashing(manifest: ScenarioManifest) -> dict[str, Any]:
    """Return dictionary representation of manifest with manifest_hash excluded."""
    d = manifest.model_dump(mode="json")
    d["manifest_hash"] = ""
    return d


def canonical_json_bytes(data: dict[str, Any]) -> bytes:
    """Encode dictionary into canonical, deterministic UTF-8 JSON bytes."""
    return json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")


def compute_manifest_hash(manifest: ScenarioManifest) -> str:
    """Compute SHA-256 over canonical serialization of manifest (hash excluded)."""
    payload = canonical_json_bytes(_manifest_dict_for_hashing(manifest))
    return hashlib.sha256(payload).hexdigest()


def finalize_manifest(manifest: ScenarioManifest) -> ScenarioManifest:
    """Compute manifest hash and return a new manifest with hash populated."""
    h = compute_manifest_hash(manifest)
    return manifest.model_copy(update={"manifest_hash": h})


def verify_manifest_hash(manifest: ScenarioManifest) -> bool:
    """Verify manifest integrity by recomputing hash over content with hash excluded."""
    expected = compute_manifest_hash(manifest)
    return bool(manifest.manifest_hash and manifest.manifest_hash == expected)


class ManifestGenerator:
    """Orchestrator for deterministic synthetic scenario generation."""

    def __init__(
        self,
        seed: int = SEED,
        rule_set: RuleSet | None = None,
    ) -> None:
        self.seed = seed
        self.rule_set = rule_set

    def generate(
        self,
        seed: int | None = None,
        rule_set: RuleSet | None = None,
    ) -> ScenarioManifest:
        """Generate the complete, validated, hashed ScenarioManifest."""
        effective_seed = seed if seed is not None else self.seed
        effective_rule_set = rule_set if rule_set is not None else self.rule_set

        # 1. Derive sub-RNGs with SeedSequence.spawn()
        rngs = make_pattern_rngs(seed=effective_seed)

        # 2. Generate each scenario in canonical order
        scenarios: list[ScenarioDefinition] = [
            generate_mule_ring(rngs["mule_ring"]),
            generate_account_takeover(rngs["account_takeover"]),
            generate_synthetic_identity(rngs["synthetic_identity"]),
            generate_card_fraud(rngs["card_fraud"]),
            generate_structuring(rngs["structuring"], StructuringParams.from_rule_set(effective_rule_set)),
            generate_legitimate_clean(rngs["legitimate_clean"]),
        ]

        # 3. Assemble and cross-validate manifest
        raw_manifest = ScenarioManifest(
            manifest_version="2.0",
            seed=effective_seed,
            observation_cutoff=OBSERVATION_CUTOFF,
            manifest_reference_at=MANIFEST_REFERENCE_AT,
            manifest_hash="",
            scenarios=scenarios,
        )

        # 4. Finalize with deterministic SHA-256 digest
        return finalize_manifest(raw_manifest)

