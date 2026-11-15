"""Unit tests for AegisFlow M2 Deterministic Synthetic Known-Fraud Scenario Generation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import re
from uuid import UUID
import numpy as np
import pytest

from aegis_ml.scenarios.constants import (
    MANIFEST_REFERENCE_AT,
    MATURITY_BOUNDARY,
    OBSERVATION_CUTOFF,
    RETAINED_TYPOLOGIES,
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
from aegis_ml.scenarios.patterns.account_takeover import generate_account_takeover
from aegis_ml.scenarios.patterns.card_fraud import generate_card_fraud
from aegis_ml.scenarios.patterns.legitimate_clean import generate_legitimate_clean
from aegis_ml.scenarios.patterns.mule_ring import generate_mule_ring
from aegis_ml.scenarios.patterns.structuring import StructuringParams, generate_structuring
from aegis_ml.scenarios.patterns.synthetic_identity import generate_synthetic_identity
from libs.aegis_domain.rules import RuleSet


@pytest.fixture(scope="module")
def manifest() -> ScenarioManifest:
    """Fixture providing a freshly generated canonical manifest."""
    return ManifestGenerator(seed=SEED).generate()


# 1. Determinism Test
def test_manifest_hash_determinism():
    """Two independent generator runs produce identical manifests and hashes."""
    m1 = ManifestGenerator(seed=SEED).generate()
    m2 = ManifestGenerator(seed=SEED).generate()

    assert m1.manifest_hash == m2.manifest_hash
    assert m1.model_dump(mode="json") == m2.model_dump(mode="json")
    assert compute_manifest_hash(m1) == compute_manifest_hash(m2)


# 2. Hash Verification and Tamper Detection
def test_manifest_hash_verification(manifest: ScenarioManifest):
    """Manifest hash verification passes, and tampering causes failure."""
    assert verify_manifest_hash(manifest) is True

    # Tamper with an amount
    tampered_dict = manifest.model_dump(mode="json")
    tampered_dict["scenarios"][0]["transactions"][0]["amount"] = "999999.99"
    tampered_manifest = ScenarioManifest.model_validate(tampered_dict)
    assert verify_manifest_hash(tampered_manifest) is False

    # Tamper with a timestamp
    tampered_dict2 = manifest.model_dump(mode="json")
    tampered_dict2["scenarios"][0]["transactions"][0]["occurred_at"] = "2025-01-01T00:00:00Z"
    tampered_manifest2 = ScenarioManifest.model_validate(tampered_dict2)
    assert verify_manifest_hash(tampered_manifest2) is False


# 3. RNG Stability Guarantees
def test_rng_stability_guarantees():
    """Verify SeedSequence.spawn guarantees: execution order independence and append safety."""
    # (a) Execution order independence: running in reversed order produces identical scenarios
    rngs_forward = make_pattern_rngs(seed=SEED)
    sc_mule_fwd = generate_mule_ring(rngs_forward["mule_ring"])
    sc_clean_fwd = generate_legitimate_clean(rngs_forward["legitimate_clean"])

    rngs_reverse = make_pattern_rngs(seed=SEED)
    sc_clean_rev = generate_legitimate_clean(rngs_reverse["legitimate_clean"])
    sc_mule_rev = generate_mule_ring(rngs_reverse["mule_ring"])

    assert sc_mule_fwd.model_dump(mode="json") == sc_mule_rev.model_dump(mode="json")
    assert sc_clean_fwd.model_dump(mode="json") == sc_clean_rev.model_dump(mode="json")

    # (b) Append safety: adding a scenario to the end preserves existing streams
    extended_order = list(SCENARIO_ORDER) + ["new_future_scenario"]
    rngs_base = make_pattern_rngs(seed=SEED)
    rngs_extended = make_pattern_rngs(seed=SEED, scenario_order=extended_order)
    for sid in SCENARIO_ORDER:
        orig_stream = rngs_base[sid].integers(0, 1000000, size=10)
        ext_stream = rngs_extended[sid].integers(0, 1000000, size=10)
        assert np.array_equal(orig_stream, ext_stream), f"Stream diverged for {sid}"


# 4. Scenario Composition
def test_all_six_scenarios_present(manifest: ScenarioManifest):
    """Manifest contains exactly the 6 required scenarios in canonical order."""
    scenario_ids = [sc.scenario_id for sc in manifest.scenarios]
    assert scenario_ids == list(SCENARIO_ORDER)
    assert len(manifest.scenarios) == 6


# 5. Typology Registry Consistency
def test_typology_registry_consistency(manifest: ScenarioManifest):
    """Every label's fraud_type strictly matches the canonical typology registry."""
    for sc in manifest.scenarios:
        expected_typology = TYPOLOGY_REGISTRY[sc.scenario_id]
        assert sc.typology == expected_typology

        for lbl in sc.labels:
            if lbl.is_fraud:
                assert lbl.fraud_type == expected_typology
                assert lbl.confidence >= 0.85
            else:
                assert lbl.fraud_type is None
                assert lbl.confidence == 1.0


# 6. Incident Constraint Compliance (m2_0003 constraints)
def test_incident_constraint_compliance(manifest: ScenarioManifest):
    """Every label satisfies paired boundaries, chronology, and discovery constraints."""
    for sc in manifest.scenarios:
        for lbl in sc.labels:
            # Paired boundaries
            assert (lbl.incident_start_at is None) == (lbl.incident_end_at is None)

            if lbl.incident_start_at and lbl.incident_end_at:
                # Chronology
                assert lbl.incident_start_at <= lbl.incident_end_at
                # Discovery
                assert lbl.incident_start_at <= lbl.labeled_at
                assert lbl.incident_end_at <= lbl.labeled_at

            # Must not exceed observation cutoff
            assert lbl.labeled_at <= OBSERVATION_CUTOFF


# 7. Bounded Incident Coverage
def test_bounded_incident_coverage(manifest: ScenarioManifest):
    """All 5 fraud scenarios generate bounded incident windows."""
    bounded_scenarios = {"mule_ring", "account_takeover", "synthetic_identity", "card_fraud", "structuring"}

    for sc in manifest.scenarios:
        if sc.scenario_id in bounded_scenarios:
            for lbl in sc.labels:
                assert lbl.incident_start_at is not None, f"Missing start boundary for {sc.scenario_id}"
                assert lbl.incident_end_at is not None, f"Missing end boundary for {sc.scenario_id}"
        elif sc.scenario_id == "legitimate_clean":
            for lbl in sc.labels:
                assert lbl.incident_start_at is None
                assert lbl.incident_end_at is None


# 8. Referential Integrity
def test_referential_integrity(manifest: ScenarioManifest):
    """All foreign key references within manifest resolve to valid accounts and transactions."""
    all_account_ids = {acc.account_id for sc in manifest.scenarios for acc in sc.accounts}
    all_txn_ids = {txn.txn_id for sc in manifest.scenarios for txn in sc.transactions}

    for sc in manifest.scenarios:
        for txn in sc.transactions:
            assert txn.account_id in all_account_ids
            if txn.destination_account_id:
                assert txn.destination_account_id in all_account_ids

        for lbl in sc.labels:
            if lbl.label_target == "transaction":
                assert lbl.txn_id in all_txn_ids
                assert lbl.account_id is None
            else:
                assert lbl.account_id in all_account_ids
                assert lbl.txn_id is None


# 9. No Direct Label Conflicts
def test_no_direct_label_conflicts(manifest: ScenarioManifest):
    """No transaction receives conflicting direct fraud and clean labels."""
    txn_labels: dict[UUID, list[SyntheticLabel]] = {}
    for sc in manifest.scenarios:
        for lbl in sc.labels:
            if lbl.label_target == "transaction" and lbl.txn_id:
                txn_labels.setdefault(lbl.txn_id, []).append(lbl)

    for tid, lbls in txn_labels.items():
        has_fraud = any(l.is_fraud for l in lbls)
        has_clean = any(not l.is_fraud for l in lbls)
        assert not (has_fraud and has_clean), f"Transaction {tid} has conflicting fraud and clean labels"


# 10. Clean Account Validity Horizon and Conflicts
def test_clean_account_validity_horizon(manifest: ScenarioManifest):
    """Clean account labels apply strictly to [opened_on, cutoff]; no clean account has fraud labels."""
    clean_scenario = next(sc for sc in manifest.scenarios if sc.scenario_id == "legitimate_clean")
    clean_account_ids = {acc.account_id for acc in clean_scenario.accounts}

    # Verify no fraud label targets any clean account
    for sc in manifest.scenarios:
        for lbl in sc.labels:
            if lbl.account_id in clean_account_ids and lbl.is_fraud:
                pytest.fail(f"Clean account {lbl.account_id} targeted by fraud label in {sc.scenario_id}")

    # Verify clean transactions are within the observation horizon
    for txn in clean_scenario.transactions:
        assert txn.occurred_at <= OBSERVATION_CUTOFF
        assert txn.occurred_at <= MATURITY_BOUNDARY


# 11. No Same-Typology Overlapping Incidents
def test_no_same_typology_overlapping_incidents(manifest: ScenarioManifest):
    """No account has overlapping incident windows for the same fraud typology."""
    acct_labels: dict[UUID, list[SyntheticLabel]] = {}
    for sc in manifest.scenarios:
        for lbl in sc.labels:
            if lbl.label_target == "account" and lbl.account_id:
                acct_labels.setdefault(lbl.account_id, []).append(lbl)

    for aid, lbls in acct_labels.items():
        by_typology: dict[str, list[SyntheticLabel]] = {}
        for l in lbls:
            if l.is_fraud and l.fraud_type:
                by_typology.setdefault(l.fraud_type, []).append(l)

        for typ, flbls in by_typology.items():
            bounded = [l for l in flbls if l.incident_start_at and l.incident_end_at]
            bounded.sort(key=lambda x: x.incident_start_at)  # type: ignore[arg-type]
            for i in range(len(bounded) - 1):
                assert bounded[i].incident_end_at < bounded[i + 1].incident_start_at  # type: ignore[operator]


# 12. Mule Ring Controller and Account Labels
def test_mule_ring_controller_and_account_labels(manifest: ScenarioManifest):
    """Verify intentional ground truth for all accounts in mule_ring: 6 account labels, 0 txn labels."""
    mule_sc = next(sc for sc in manifest.scenarios if sc.scenario_id == "mule_ring")

    # Exactly 6 accounts: 1 controller + 5 mules
    assert len(mule_sc.accounts) == 6
    controller = next(acc for acc in mule_sc.accounts if not acc.is_mule)
    mules = [acc for acc in mule_sc.accounts if acc.is_mule]
    assert len(mules) == 5

    # Exactly 6 account-level labels, 0 transaction labels
    assert len(mule_sc.labels) == 6
    assert all(l.label_target == "account" for l in mule_sc.labels)

    # Controller has intentional fraud label
    ctrl_lbl = next(l for l in mule_sc.labels if l.account_id == controller.account_id)
    assert ctrl_lbl.is_fraud is True
    assert ctrl_lbl.fraud_type == "money_mule"
    assert ctrl_lbl.confidence == 1.0
    assert "mastermind" in ctrl_lbl.evidence.confirmation_event

    # Mules have intentional fraud labels
    for m in mules:
        m_lbl = next(l for l in mule_sc.labels if l.account_id == m.account_id)
        assert m_lbl.is_fraud is True
        assert m_lbl.confidence == 1.0


# 13. External References Globally Unique and Valid
def test_external_refs_unique_and_valid(manifest: ScenarioManifest):
    """All external_ref values are globally unique and <= 64 characters."""
    all_refs = [txn.external_ref for sc in manifest.scenarios for txn in sc.transactions]
    assert len(all_refs) == len(set(all_refs)), "Duplicate external_ref detected!"
    for ref in all_refs:
        assert len(ref) <= 64
        assert ref.startswith("AF-M2-SYN-")


# 14. Evidence Provenance Markers
def test_evidence_provenance_markers(manifest: ScenarioManifest):
    """All EvidenceProvenance records have synthetic=True, SYN- prefixes, and no real regulatory IDs."""
    real_pat = re.compile(r"^(DISP|CB|SAR)-\d{4}")

    for sc in manifest.scenarios:
        for lbl in sc.labels:
            ev = lbl.evidence
            assert ev.synthetic is True
            assert ev.investigation_ref.startswith("SYN-INV-")
            assert ev.analyst_id.startswith("SYN-ANALYST-")
            assert ev.manifest_reference_at == MANIFEST_REFERENCE_AT
            assert not real_pat.search(ev.investigation_ref)

            if ev.dispute_ref:
                assert ev.dispute_ref.startswith("SYN-DISP-")
                assert not real_pat.search(ev.dispute_ref)

            for aref in ev.alert_refs:
                assert aref.startswith("SYN-ALERT-")
                assert not real_pat.search(aref)


# 15. Amounts Currency and Precision
def test_amounts_currency_and_precision(manifest: ScenarioManifest):
    """All transaction amounts are strictly positive, currency is INR, and have 2 decimal places."""
    for sc in manifest.scenarios:
        for txn in sc.transactions:
            assert txn.currency == "INR"
            assert txn.amount > Decimal("0")
            assert txn.amount.as_tuple().exponent == -2
            assert txn.merchant_id is None
            assert txn.ip_id is None


# 16. Structuring Factory and Detection Condition
def test_structuring_factory_and_detection_condition(manifest: ScenarioManifest):
    """Verify StructuringParams.from_rule_set factory and simulated R003_STRUCTURING condition."""
    rs = RuleSet()
    params = StructuringParams.from_rule_set(rs)

    assert params.structuring_report_limit == rs.structuring_report_limit  # Decimal('1000000')
    assert params.single_txn_threshold == rs.high_value_threshold        # Decimal('500000')
    assert params.window_hours == rs.structuring_window_hours            # 24
    assert params.min_sub_threshold_count == 3

    struct_sc = next(sc for sc in manifest.scenarios if sc.scenario_id == "structuring")
    for acc in struct_sc.accounts:
        acc_txns = [t for t in struct_sc.transactions if t.account_id == acc.account_id]

        # Find 24h structuring cluster
        acc_lbl = next(l for l in struct_sc.labels if l.account_id == acc.account_id)
        cluster = [
            t for t in acc_txns
            if acc_lbl.incident_start_at <= t.occurred_at <= acc_lbl.incident_end_at  # type: ignore[operator]
        ]

        # Verification of R003_STRUCTURING:
        # 1. Count >= 3
        assert len(cluster) >= params.min_sub_threshold_count
        # 2. Each amount < single_txn_threshold
        for t in cluster:
            assert t.amount < params.single_txn_threshold, f"Debit {t.amount} not sub-threshold"
        # 3. Sum >= structuring_report_limit
        total_cluster_sum = sum(t.amount for t in cluster)
        assert total_cluster_sum >= params.structuring_report_limit


# 17. CLI Output Safety & Overwrite Protection Tests
from scripts.generate_synthetic_scenarios import main as cli_main


def test_cli_export_new_destination_succeeds(tmp_path):
    """Test 1: New destination: export succeeds and creates valid manifest."""
    out_file = tmp_path / "exported_manifest.json"
    ret = cli_main(["--output-json", str(out_file), "--verify-hash"])
    assert ret == 0
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["manifest_hash"] != ""
    assert len(data["scenarios"]) == 6


def test_cli_export_existing_without_overwrite_fails_and_preserves(tmp_path, capsys):
    """Test 2: Existing destination without --overwrite fails and preserves contents."""
    out_file = tmp_path / "protected_manifest.json"
    sentinel = "PRE_EXISTING_CRITICAL_CONTENT_DO_NOT_OVERWRITE"
    out_file.write_text(sentinel, encoding="utf-8")

    ret = cli_main(["--output-json", str(out_file)])
    assert ret == 1
    # Verify file was NOT modified
    assert out_file.read_text(encoding="utf-8") == sentinel

    captured = capsys.readouterr()
    assert "Error: destination file already exists" in captured.err
    assert "--overwrite" in captured.err


def test_cli_export_existing_with_overwrite_succeeds(tmp_path):
    """Test 3: Existing destination with --overwrite succeeds and replaces with valid JSON."""
    out_file = tmp_path / "replaceable_manifest.json"
    out_file.write_text("OLD_DATA", encoding="utf-8")

    ret = cli_main(["--output-json", str(out_file), "--overwrite", "--verify-hash"])
    assert ret == 0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert content != "OLD_DATA"
    data = json.loads(content)
    assert data["manifest_hash"] != ""
    assert len(data["scenarios"]) == 6


def test_cli_export_creates_parent_directories(tmp_path):
    """Test 4: Parent directory creation continues working when destination is deeply nested."""
    nested_dir = tmp_path / "level1" / "level2" / "level3"
    out_file = nested_dir / "nested_manifest.json"
    assert not nested_dir.exists()

    ret = cli_main(["--output-json", str(out_file)])
    assert ret == 0
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert len(data["scenarios"]) == 6


def test_cli_export_no_partial_or_truncated_output_on_write_failure(tmp_path, monkeypatch):
    """Test 5: No partial or truncated output remains after a failed export midway."""
    out_file = tmp_path / "aborted_export.json"
    assert not out_file.exists()

    original_open = open

    class FailingWriter:
        def __init__(self, real_file):
            self.real_file = real_file

        def write(self, s):
            self.real_file.write(s[:50])
            raise OSError("Simulated disk full during write")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.real_file.close()

    def mock_open(file, mode="r", *args, **kwargs):
        if str(file) == str(out_file) and "x" in mode:
            real_f = original_open(file, mode, *args, **kwargs)
            return FailingWriter(real_f)
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", mock_open)

    ret = cli_main(["--output-json", str(out_file)])
    assert ret == 1
    # Verify that the partial file was unlinked and does not remain on disk
    assert not out_file.exists(), "Partial output file should have been cleaned up"


def test_cli_export_temp_file_cleaned_up_on_replace_failure(tmp_path, monkeypatch):
    """Test 6: Temporary file is cleaned up and original file preserved when os.replace fails."""
    import glob
    import os

    out_file = tmp_path / "replace_target.json"
    sentinel = "ORIGINAL_MANIFEST_CONTENT"
    out_file.write_text(sentinel, encoding="utf-8")

    def mock_replace(src, dst):
        raise OSError("Simulated atomic replace failure")

    monkeypatch.setattr(os, "replace", mock_replace)

    ret = cli_main(["--output-json", str(out_file), "--overwrite"])
    assert ret == 1
    # Target file preserved
    assert out_file.read_text(encoding="utf-8") == sentinel

    # Zero leftover temp files in destination directory
    leftovers = glob.glob(str(tmp_path / ".manifest_tmp_*"))
    assert len(leftovers) == 0, f"Leftover temp files detected: {leftovers}"


def test_cli_export_permissions_appropriate(tmp_path):
    """Test 7: Exported manifest file permissions respect umask and preserve existing permissions."""
    import os
    import stat

    current_umask = os.umask(0)
    os.umask(current_umask)
    expected_default_mode = 0o666 & ~current_umask

    # (a) New file export respects process umask
    new_out = tmp_path / "umask_manifest.json"
    ret = cli_main(["--output-json", str(new_out)])
    assert ret == 0
    mode_new = stat.S_IMODE(os.stat(new_out).st_mode)
    assert mode_new == expected_default_mode

    # (b) Overwrite export preserves custom existing file permissions
    custom_out = tmp_path / "custom_perm_manifest.json"
    custom_out.write_text("{}", encoding="utf-8")
    os.chmod(custom_out, 0o640)

    ret = cli_main(["--output-json", str(custom_out), "--overwrite"])
    assert ret == 0
    mode_overwritten = stat.S_IMODE(os.stat(custom_out).st_mode)
    assert mode_overwritten == 0o640
