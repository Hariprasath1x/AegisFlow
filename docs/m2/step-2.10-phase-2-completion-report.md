# AegisFlow M2 Step 2.10 — Phase 2 Completion Report: Deterministic Synthetic Known-Fraud Scenario Generation

**Milestone:** M2 — Feature Store & Entity Graph  
**Step:** 2.10 — Deterministic Synthetic Known-Fraud Scenario Generation  
**Phase:** Phase 2 — In-Memory Synthetic Manifest Generation & Safe CLI Export  
**Status:** COMPLETE & ACCEPTED  
**Date:** November 16, 2026  
**Git Baseline Commit:** `c0dc58931ef12545d6b59b2e698774f839a8b42c` (`c0dc589`)  
**Target Branch:** `feature/m2-feature-store`  
**Execution Environment:** Isolated in-memory Python 3.12 (`.venv-m2`)  

---

## 1. Executive Summary

### Purpose of Phase 2
The objective of Phase 2 was to implement a fully deterministic, in-memory synthetic known-fraud scenario generation system for AegisFlow Milestone 2 Step 2.10. This corpus provides verified, multi-typology ground truth for training supervised fraud models and evaluating feature store pipelines without introducing non-deterministic simulation noise or depending on volatile external services.

### What Was Implemented
- **Domain Modeling & Referential Validation:** High-fidelity Pydantic v2 schemas defining synthetic accounts, transactions, incident-bounded labels, evidence provenance, scenarios, and complete cross-scenario referential manifests.
- **Deterministic Multi-Pattern Generators:** Six distinct scenario generators covering money mules, account takeover, synthetic identity, card-not-present fraud, sub-threshold structuring, and legitimate clean controls.
- **Reproducible RNG Architecture:** Hierarchical sub-stream derivation using NumPy `SeedSequence.spawn()` anchored to canonical seed `20261010`, ensuring stream independence and execution order invariance.
- **Cryptographic Serialization & Hashing:** Canonical JSON encoding with sorted keys, compact separators, strict UTC ISO-8601 formatting with `Z` suffixes, and deterministic SHA-256 digest computation (`25490a2b679ae60bcab42662cda3799f23ef79a62ef964c72398200614b50b6c`).
- **Hardened CLI Export Engine:** Safe manifest export utility with exclusive atomic file creation (`open(path, "x")`), opt-in `--overwrite` protection, atomic replacement via temporary files, automatic failure cleanup, and umask-preserving permissions.

### Overall Completion and Acceptance Status
Phase 2 implementation has passed all read-only audits, targeted verifications, and unit testing suites (25/25 unit and CLI tests passing). It is functionally complete, deterministic, and strictly isolated from database persistence layers. Phase 2 is formally **ACCEPTED** and ready for Phase 3 authorization.

---

## 2. Scope and Objectives

Phase 2 strictly adhered to the authorized scope outlined in `phase2_revised_proposal.md` (Revision 4):

1. **Synthetic Scenario Generation:**
   Constructing 6 self-consistent scenarios comprising 30 accounts, 405 transactions, and 102 labels.
2. **Supported Fraud and Legitimate Typologies:**
   - `mule_ring`: Rapid dispersal money mule network (`money_mule`).
   - `account_takeover`: Victim account credential takeover and unauthorized fund drain (`account_takeover`).
   - `synthetic_identity`: First-party bust-out credit establishment and rapid exhaustion (`first_party_synthetic`).
   - `card_fraud`: Compromised card-not-present bursts with confirmed chargebacks (`card_not_present`).
   - `structuring`: Deliberate sub-threshold transaction splitting triggering AML rule `R003_STRUCTURING` (`structuring`).
   - `legitimate_clean`: Explicit negative controls asserting clean account and transaction validity (`is_fraud=False`).
3. **Deterministic Manifest Generation:**
   Reproducing bit-for-bit identical manifest JSON across independent runs and operating systems using canonical seed `20261010`.
4. **Referential Integrity & Schema Compliance:**
   Guaranteeing all transaction `account_id` and `destination_account_id` foreign keys resolve to accounts declared within the manifest; enforcing `ck_ground_truth_*` constraint invariants on all labels.
5. **Safe CLI Manifest Export:**
   Providing standalone CLI execution (`scripts/generate_synthetic_scenarios.py`) supporting inspection, verification, and non-destructive disk export.

---

## 3. Implementation Details

```
ml/src/aegis_ml/scenarios/
├── __init__.py               # Package entrypoint
├── constants.py              # Canonical constants, namespaces, seeds, and RNG derivation
├── models.py                 # Pydantic domain models with strict UTC & constraint validators
├── generator.py              # ManifestGenerator orchestrator and canonical SHA-256 hashing
└── patterns/
    ├── __init__.py           # Pattern generator exports
    ├── mule_ring.py          # 1 controller + 5 mules, fan-in & dispersal
    ├── account_takeover.py   # 3 victims, baseline -> ATO -> drain
    ├── synthetic_identity.py # 4 accounts, credit build -> bust-out drain
    ├── card_fraud.py         # 6 card accounts, CNP bursts + chargebacks
    ├── structuring.py        # 3 accounts, sub-threshold 24h clusters (RuleSet DI)
    └── legitimate_clean.py   # 8 verified clean accounts & transactions
scripts/
└── generate_synthetic_scenarios.py  # CLI export and manifest hash verification
```

### 3.1 Domain Models (`models.py`)
- **`EvidenceProvenance`:** Tracks synthetic evidence origin (`SYN-` references, alert IDs, dispute IDs, analyst attribution, confirmation events, and deterministic `manifest_reference_at`).
- **`SyntheticAccount`:** Validates account types (`current`, `savings`, `card`, `wallet`), statuses (`active`, `dormant`, `frozen`, `closed`), `YYYY-MM-DD` opened dates, and currency (`INR`).
- **`SyntheticTransaction`:** Enforces strictly positive amounts with exactly 2 decimal places, currency `INR`, channel enumerations, and mandates `merchant_id=None` and `ip_id=None` in Phase 2. Enforces PostgreSQL `ck_transactions_transfer_target` constraint invariants (`channel == "transfer"` requires non-null `destination_account_id`; non-transfer channels require null `destination_account_id`).
- **`SyntheticLabel`:** Directly mirrors PostgreSQL `ground_truth_labels` schema post-`m2_0003`. Enforces:
  - Target mutual exclusivity (`label_target="transaction"` requires `txn_id` and `account_id IS NULL`; `label_target="account"` requires `account_id` and `txn_id IS NULL`).
  - Paired incident boundaries (`incident_start_at` and `incident_end_at` must both be set or both be `NULL`).
  - Chronological consistency: $\text{incident\_start\_at} \le \text{incident\_end\_at} \le \text{labeled\_at} \le \text{OBSERVATION\_CUTOFF}$.
  - Confidence and typology pairing: fraud labels require confidence $\ge 0.85$ and valid `fraud_type`; clean labels mandate confidence $= 1.0$ and `fraud_type=None`.
- **`ScenarioManifest`:** Validates global uniqueness of `account_id`, `txn_id`, `label_id`, and `external_ref`; validates all intra-manifest foreign keys; verifies that clean accounts possess zero conflicting fraud labels.

### 3.2 Constants and Typology Registry (`constants.py`)
- **Seed & Namespace:** Fixed `SEED = 20261010` and RFC-4122 DNS-derived namespace `SCENARIO_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "aegisflow.scenarios.v1.seed=20261010")`.
- **Temporal Anchors:** `OBSERVATION_CUTOFF = 2026-04-01T00:00:00Z`, `MATURITY_BOUNDARY = 2026-03-25T00:00:00Z`, `MANIFEST_REFERENCE_AT = 2026-04-01T00:00:00Z`.
- **Typology Registry:** Authoritative mapping between scenario identifiers and PostgreSQL `fraud_type` column values:
  - `mule_ring` $\rightarrow$ `"money_mule"`
  - `account_takeover` $\rightarrow$ `"account_takeover"`
  - `synthetic_identity` $\rightarrow$ `"first_party_synthetic"`
  - `card_fraud` $\rightarrow$ `"card_not_present"`
  - `structuring` $\rightarrow$ `"structuring"`
  - `legitimate_clean` $\rightarrow$ `None`
- **RNG Derivation (`make_pattern_rngs`):** Uses NumPy `SeedSequence(SEED).spawn(len(SCENARIO_ORDER))` to generate isolated, independent child random number generators. Guarantees that scenario execution order does not affect generated data streams and that appending future scenarios preserves existing sequences.

### 3.3 Scenario Pattern Generators (`patterns/`)
- **`mule_ring.py`:** Generates 1 controller and 5 mule accounts. Controller deposits funds into mules who rapidly disperse via external transfers. Generates 6 bounded account-level fraud labels (`money_mule`) with zero transaction labels.
- **`account_takeover.py`:** Models 3 victim accounts. Legitimate baseline behavior is interrupted by device change, credential takeover, and rapid fund drain. Generates 3 bounded account labels and 16 direct transaction fraud labels (`account_takeover`). All post-takeover drain transactions strictly use `channel = "wire"` (preserving RNG draw sequences and PostgreSQL `ck_transactions_transfer_target` compliance).
- **`synthetic_identity.py`:** Models 4 credit-building synthetic accounts aged over months before executing bust-out wire drains. Generates 4 bounded account labels and 5 direct transaction labels (`first_party_synthetic`).
- **`card_fraud.py`:** Models 6 victim card accounts. Normal spending is mixed with bursts of unauthorized card-not-present debits. Generates 20 direct transaction labels (`card_not_present`) with distinct chargeback confirmation (`incident_end_at`) and labeling timestamps.
- **`structuring.py`:** Models 3 accounts executing deliberate sub-threshold transactions. Uses `StructuringParams.from_rule_set(rs)` factory dependency injection querying `libs/aegis_domain/rules.py:RuleSet`. Verifies simulated detection criteria of rule `R003_STRUCTURING` (each debit $< \text{₹}500{,}000$, 24h count $\ge 3$, 24h sum $\ge \text{₹}1{,}000{,}000$). Generates 3 bounded account labels (`structuring`).
- **`legitimate_clean.py`:** Models 8 legitimate accounts across ATM, POS, UPI, and ACH channels. Generates 8 timeless clean account labels and 37 clean transaction labels (`is_fraud=False`, confidence 1.0).

### 3.4 Hardened CLI Export Engine (`scripts/generate_synthetic_scenarios.py`)
- **Exclusive Atomic Creation:** Default export uses `open(path, "x")` (`O_CREAT | O_EXCL`). If the destination path already exists, the OS immediately raises `FileExistsError` without opening or modifying the destination file. Stderr outputs an error and exits with code `1`.
- **Opt-in `--overwrite` Flag:** Requires explicit user intent to replace existing files.
- **Atomic File Replacement:** Overwrite mode writes to a temporary file in the destination directory (`tempfile.mkstemp(dir=out_dir, prefix=".manifest_tmp_")`) and performs an atomic POSIX replacement (`os.replace`).
- **Guaranteed Cleanup:** Employs a `try ... finally` block tracking `replaced: bool`. Any failure during write or replacement immediately closes file descriptors and unlinks temporary files.
- **Permission Normalization:** Overwritten files explicitly adopt the existing destination's file mode or adhere to the process `umask` (`0o666 & ~umask`, typically `0644`), avoiding the restrictive `0600` default of `mkstemp`.

---

## 4. Files Delivered

All 13 files authorized for Phase 2 were created in the designated repository locations:

| # | File Path | Role / Responsibility |
|---|---|---|
| 1 | `ml/src/aegis_ml/scenarios/__init__.py` | Package root; exposes public API for models, constants, and generator. |
| 2 | `ml/src/aegis_ml/scenarios/constants.py` | Canonical seed, temporal cutoff constants, typology registry, UUIDv5 derivation, and `SeedSequence` RNG derivation. |
| 3 | `ml/src/aegis_ml/scenarios/models.py` | Pydantic v2 schemas with UTC validators, constraint checkers, and cross-scenario manifest referential integrity validation. |
| 4 | `ml/src/aegis_ml/scenarios/generator.py` | Manifest orchestrator, canonical JSON serialization, and SHA-256 manifest hashing. |
| 5 | `ml/src/aegis_ml/scenarios/patterns/__init__.py` | Scenario pattern registry and exports. |
| 6 | `ml/src/aegis_ml/scenarios/patterns/mule_ring.py` | Money mule network pattern generator (6 accounts, 100 txns, 6 labels). |
| 7 | `ml/src/aegis_ml/scenarios/patterns/account_takeover.py` | Account takeover pattern generator (3 accounts, 34 txns, 19 labels). |
| 8 | `ml/src/aegis_ml/scenarios/patterns/synthetic_identity.py` | Synthetic identity bust-out generator (4 accounts, 56 txns, 9 labels). |
| 9 | `ml/src/aegis_ml/scenarios/patterns/card_fraud.py` | Card-not-present fraud generator (6 accounts, 61 txns, 20 labels). |
| 10 | `ml/src/aegis_ml/scenarios/patterns/structuring.py` | AML sub-threshold structuring generator with RuleSet DI (3 accounts, 21 txns, 3 labels). |
| 11 | `ml/src/aegis_ml/scenarios/patterns/legitimate_clean.py` | Negative control legitimate generator (8 accounts, 133 txns, 45 labels). |
| 12 | `scripts/generate_synthetic_scenarios.py` | Standalone CLI utility for generation, verification, and non-destructive JSON export. |
| 13 | `tests/m2/unit/test_synthetic_scenarios.py` | Comprehensive unit and CLI safety test suite (25 passing tests). |

---

## 5. Safety and Reliability

The implementation enforces multi-tiered safety guarantees:

### 5.1 Existing Destination Protection
Exporting without `--overwrite` utilizes atomic exclusive creation (`open(path, "x")`). Unlike simple `os.path.exists()` checks, this kernel-level check is immune to Time-of-Check to Time-of-Use (TOCTOU) race conditions. Pre-existing files cannot be overwritten, truncated, or corrupted.

### 5.2 Atomic Replacement & Failure Cleanup
When `--overwrite` is specified, writes target a hidden temporary file in the destination directory. Destination replacement is achieved via `os.replace()`, guaranteeing that external processes never observe a partially written or truncated manifest. In the event of an I/O error, disk full condition, or process interruption, a `finally` block guarantees that unreplaced temporary files are immediately unlinked.

### 5.3 Output File Permission Behavior
Temporary files created via `mkstemp()` default to restrictive `0600` permissions. The generator captures the process `umask` or the pre-existing file's mode and applies `os.fchmod(temp_fd, target_mode)` prior to replacement. This ensures generated manifests are appropriately accessible to multi-user or containerized pipelines.

### 5.4 Referential Integrity & Constraint Enforcement
Manifest construction executes exhaustive validation:
- All transaction source and destination accounts must exist in `manifest.scenarios[*].accounts`.
- Label foreign keys must resolve to valid transactions or accounts.
- Transaction amounts must be strictly positive with exactly two decimal places in INR.
- Clean accounts cannot be associated with any positive fraud label.
- Incident windows cannot overlap for the same account and typology.

### 5.5 Deterministic Generation Guarantees
- Sub-RNGs are derived via `np.random.SeedSequence(20261010).spawn(6)`.
- Scenarios are decoupled from global random state; execution order does not alter output values.
- Datetime objects are strictly validated as timezone-aware UTC with zero offset and serialized with canonical `Z` suffix.
- SHA-256 digest is computed over canonical JSON bytes with sorted keys and compact separators.

---

## 6. Testing and Verification

Targeted testing was conducted in the isolated Python 3.12 environment (`.venv-m2`). All 25 unit and CLI safety tests pass.

### 6.1 Unit Test Execution Command
```bash
PYTHONPATH=services/ingestion/src:ml/src:. .venv-m2/bin/pytest tests/m2/unit/test_synthetic_scenarios.py -v
```

### 6.2 Test Execution Results
```text
============================= test session starts ==============================
platform darwin -- Python 3.12.15, pytest-8.3.4, pluggy-1.7.0 -- /Users/hariprasathc/Projects/aegisflow/.venv-m2/bin/python
cachedir: .pytest_cache
rootdir: /Users/hariprasathc/Projects/aegisflow
plugins: anyio-4.15.1
collected 25 items

tests/m2/unit/test_synthetic_scenarios.py::test_manifest_hash_determinism PASSED [  4%]
tests/m2/unit/test_synthetic_scenarios.py::test_manifest_hash_verification PASSED [  8%]
tests/m2/unit/test_rng_stability_guarantees PASSED [ 12%]
tests/m2/unit/test_all_six_scenarios_present PASSED [ 16%]
tests/m2/unit/test_typology_registry_consistency PASSED [ 20%]
tests/m2/unit/test_incident_constraint_compliance PASSED [ 24%]
tests/m2/unit/test_bounded_incident_coverage PASSED [ 28%]
tests/m2/unit/test_referential_integrity PASSED [ 32%]
tests/m2/unit/test_no_direct_label_conflicts PASSED [ 36%]
tests/m2/unit/test_clean_account_validity_horizon PASSED [ 40%]
tests/m2/unit/test_no_same_typology_overlapping_incidents PASSED [ 44%]
tests/m2/unit/test_mule_ring_controller_and_account_labels PASSED [ 48%]
tests/m2/unit/test_external_refs_unique_and_valid PASSED [ 52%]
tests/m2/unit/test_evidence_provenance_markers PASSED [ 56%]
tests/m2/unit/test_amounts_currency_and_precision PASSED [ 60%]
tests/m2/unit/test_structuring_factory_and_detection_condition PASSED [ 64%]
tests/m2/unit/test_cli_export_new_destination_succeeds PASSED [ 68%]
tests/m2/unit/test_cli_export_existing_without_overwrite_fails_and_preserves PASSED [ 72%]
tests/m2/unit/test_cli_export_existing_with_overwrite_succeeds PASSED [ 76%]
tests/m2/unit/test_cli_export_creates_parent_directories PASSED [ 80%]
tests/m2/unit/test_cli_export_no_partial_or_truncated_output_on_write_failure PASSED [ 84%]
tests/m2/unit/test_cli_export_temp_file_cleaned_up_on_replace_failure PASSED [ 88%]
tests/m2/unit/test_cli_export_permissions_appropriate PASSED [ 92%]
tests/m2/unit/test_transfer_target_constraint_model_validation PASSED [ 96%]
tests/m2/unit/test_account_takeover_channel_alignment_regression PASSED [100%]

============================== 25 passed in 0.53s ==============================
```

### 6.3 Standalone CLI Verification
```bash
$ PYTHONPATH=services/ingestion/src:ml/src:. .venv-m2/bin/python scripts/generate_synthetic_scenarios.py --verify-hash
Generated manifest: 6 scenarios, hash=25490a2b679ae60bcab42662cda3799f23ef79a62ef964c72398200614b50b6c
Manifest hash verification: PASSED
```

---

## 7. Deterministic Manifest Verification

| Metric | Verified Value |
|---|---|
| **Scenario Count** | `6` |
| **Total Accounts** | `30` |
| **Total Transactions** | `405` |
| **Total Labels** | `102` |
| **Random Seed** | `20261010` |
| **Manifest Version** | `"2.0"` |
| **Canonical SHA-256 Digest** | `25490a2b679ae60bcab42662cda3799f23ef79a62ef964c72398200614b50b6c` |

### Scenario Breakdown

| Scenario ID | Typology | Accounts | Transactions | Labels | Target & Incident Type |
|---|---|---|---|---|---|
| `mule_ring` | `"money_mule"` | 6 | 100 | 6 | 6 account labels (bounded) |
| `account_takeover` | `"account_takeover"` | 3 | 34 | 19 | 3 account labels, 16 txn labels (bounded) |
| `synthetic_identity` | `"first_party_synthetic"` | 4 | 56 | 9 | 4 account labels, 5 txn labels (bounded) |
| `card_fraud` | `"card_not_present"` | 6 | 61 | 20 | 20 txn labels (bounded) |
| `structuring` | `"structuring"` | 3 | 21 | 3 | 3 account labels (bounded) |
| `legitimate_clean` | `None` (`is_fraud=False`) | 8 | 133 | 45 | 8 account labels (timeless), 37 txn labels |
| **Total** | — | **30** | **405** | **102** | — |

### 7.1 Phase 3 Risk 1 Alignment: ATO Channel Normalization (Option B)
During pre-Phase 3 schema alignment audits, 7 post-takeover drain transactions in `account_takeover.py` were identified that randomly selected `channel = "transfer"` without defining internal destination accounts (`destination_account_id = None`), which would violate PostgreSQL constraint `ck_transactions_transfer_target`.
- **Resolution:** In accordance with approved Option B, post-takeover external drains were aligned to `channel = "wire"` (representing external interbank wire transfers).
- **RNG Sequence Invariant:** In `account_takeover.py`, `_ = rng.choice(["wire", "transfer"])` was maintained to preserve identical RNG consumption across all 34 transactions, ensuring all transaction IDs, account IDs, amounts, and timestamps remained completely unchanged.
- **Model Invariant:** `SyntheticTransaction` validation was strengthened to enforce `ck_transactions_transfer_target` at model construction time.
- **Full Manifest Constraint Audit:** Audited all 405 transactions across all 6 scenarios against PostgreSQL constraint `ck_transactions_transfer_target`; verified 0 violations.
- **Updated Digest:** The cryptographic digest of the canonical manifest updated from `551ac3b1...` to `25490a2b679ae60bcab42662cda3799f23ef79a62ef964c72398200614b50b6c`.

### Why Preserving the Hash Matters
1. **Model Reproducibility:** Downstream machine learning feature pipelines and supervised classifiers trained in Phase 4 rely on exact, stable label boundaries and transaction histories.
2. **Regression Prevention:** Any unintended modification to generator code, random number consumption, or business logic immediately changes the SHA-256 hash, failing CI/CD verification.
3. **Data Drift Detection:** A fixed cryptographic anchor distinguishes synthetic training baselines from production event streams.
4. **Tamper Evidence:** Guarantees that synthetic ground truth ingested into PostgreSQL in Phase 3 exactly matches the audited specification.

---

## 8. Database and Migration Isolation

Phase 2 was executed under strict database isolation safeguards:
- **Zero Database Operations:** No database connections were established; no SQL statements or queries were executed against any database.
- **Zero Database Writes:** PostgreSQL storage was completely uncontacted.
- **Zero Migrations Executed:** Alembic migrations were not executed. The development database remains at revision `m2_0002` (as established at the conclusion of Step 2.9). Migration `m2_0003` remains committed in Git but unapplied to the development database.
- **Zero Phase 3 Code:** No seeding scripts, database loaders, or DVC pipelines were implemented. Phase 2 remains strictly in-memory and file-export-only.

---

## 9. Git and Delivery Status

### 9.1 Historical Pre-Commit Baseline State (Phase 2 Initial Delivery)
> [!NOTE]
> The terminal output below records the historical working-tree status prior to the creation of the Phase 2 delivery commit (`1c2e80f`), capturing the initial delivery state of the 13 implementation and test files alongside this report.

```text
On branch feature/m2-feature-store
Your branch is up to date with 'origin/feature/m2-feature-store'.

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	docs/m2/step-2.10-phase-2-completion-report.md
	ml/src/aegis_ml/scenarios/
	scripts/generate_synthetic_scenarios.py
	tests/m2/unit/test_synthetic_scenarios.py

nothing added to commit but untracked files present (use "git add" to track)
```

### 9.2 Baseline Commit
- **Commit Hash:** `1c2e80f97e28a99f3a76bf2c7f36b291681c1697` (`1c2e80f`)
- **Commit Subject:** `feat(m2): complete synthetic scenario generation`
- **Phase 2 Status:** Implemented, verified, and Option B alignment applied.
- **Working Tree Integrity:** Clean on all tracked repository files (`git diff --check` passes with zero output). Exactly 13 implementation/test files and 1 documentation file delivered.

---

## 10. Acceptance Decision

| Acceptance Criteria | Verification Assessment | Status |
|---|---|---|
| **Functional Completeness** | All 6 scenarios implemented with required counts, typologies, and relationships. | **PASS** |
| **Test Coverage** | 25 targeted unit tests passing (100% success rate). | **PASS** |
| **Determinism** | Bit-for-bit serialization determinism verified; canonical hash matching. | **PASS** |
| **CLI Output Safety** | Exclusive creation, `--overwrite` flag, atomic replacement, cleanup on error verified. | **PASS** |
| **Scope Compliance** | Exactly 13 authorized code/test files delivered; zero database operations. | **PASS** |
| **Readiness for Phase 3** | Codebase and manifest ready for database seeding upon authorization. | **ACCEPTED** |

**Formal Decision:** Phase 2 is **APPROVED FOR CLOSURE**.

---

## 11. Next Steps

With Phase 2 verified and complete, the next activity in the AegisFlow development sequence is:

- **Phase 3 Authorization & Execution:**
  - Upgrading the development database to Alembic revision `m2_0003`.
  - Implementing idempotent database seeding for the 30 accounts, 405 transactions, and 102 labels from the canonical manifest.
  - Verifying relational integrity, foreign keys, and constraint adherence inside PostgreSQL.
  - Validating database-persisted ground truth against in-memory manifest hash.

