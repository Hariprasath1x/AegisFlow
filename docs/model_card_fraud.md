# Model Card — aegisflow-fraud

## Overview

| Field | Value |
|---|---|
| Model Name | `aegisflow-fraud` |
| Aliases | `champion` (active scoring service), `challenger` (shadow scoring) |
| Fallback Alias | `previous_champion` (one-command rollback target) |
| Architecture | XGBoost (Champion) / LightGBM (Challenger) with Isotonic Calibration |
| Training Data | IEEE-CIS weeks 1–6 (time split, 24h embargo, 30d label maturity horizon) |
| Evaluation Data | IEEE-CIS weeks 7–8 (strictly out-of-time test split) |
| Objective Primary Metric | `recall_at_1pct_fpr` (minimum relative improvement threshold: >= 2.0%) |
| Calibration | Isotonic regression with ECE <= 0.05 on held-out calibration partition |
| Features | 28 canonical features declared in `aegis_ml.features.definitions` |

## Intended Use

Ranking incoming payment transactions for real-time `approve` / `review` / `decline` decisioning inside AegisFlow.
- The deterministic rules layer (`rules_baseline.py`) can force a decline regardless of model score.
- A load-time SHAP TreeExplainer generates the top-5 reason codes for every reviewed or declined transaction.

## Out of Scope

Any decision about a customer beyond the specific transaction. The model score is NOT a creditworthiness signal, NOT a persistent customer risk rating, and NOT evidence in a Suspicious Activity Report (SAR). The SAR narrative is generated from the entity evidence pack, not raw model scores.

## Known Limitations

- Trained primarily on card-present and card-not-present retail payment patterns.
- High-velocity wire and wallet channels have distinct distributions and are monitored with cohort-specific recall reporting.

---

## Registry Architecture & Champion / Challenger Policy (FR-7)

### Aliases
- **`champion`**: The model version actively loaded and evaluated by the production scoring service.
- **`challenger`**: The shadow-scored candidate model version evaluating real traffic in parallel.
- **`previous_champion`**: The predecessor champion version preserved for zero-downtime rollback.

### Candidate Registration
- Training runs register new candidates under `aegisflow-fraud`.
- A newly registered model lands strictly as `challenger`. It **never** overwrites an active `champion`.
- **First Model Registration**: If no `champion` alias exists, the initial registered model version is assigned both `champion` and `challenger`.
- The fitted `IsotonicRegression` calibrator travels **with** the model artifact inside the same run context (`calibrator/`).
- Exact feature column ordering is persisted as `model/columns.json`.
- Frequency encoding mapping and model metadata are logged in `model_meta.json`.

### Promotion Requirements
1. **Gate Compliance**: The challenger must satisfy all five spec-traced gates. All five metrics must be present and finite; omitting any metric, or supplying NaN, Inf, or non-numeric values, immediately rejects the challenger:
   - `recall_at_1pct_fpr >= 0.50` (Section 19 floor)
   - `lift_over_rules >= 0.30` (Objective 5.2)
   - `pr_auc >= 0.60` (Section 14 headline quality)
   - `ece <= 0.05` (Section 14 calibration bound)
   - `typology_detection >= 0.80` (Objective 5.4)
2. **Noise Band Protection**: The challenger must beat the champion on the primary metric (`recall_at_1pct_fpr`) by at least **2.0% relative lift**. A tie or marginal lift within noise (< 2%) is rejected.
3. **Audit Attribution**: Promotion is an explicit human action (`make promote ACTOR=<operator>` or `python scripts/champion.py --promote --actor <operator>`). Operator identity must be explicitly provided; the Makefile does not implicitly default to `$USER`, and omitting `ACTOR`, supplying an empty string, or passing `unknown` is rejected by the CLI with exit code 2. The promoting operator's identity is stamped in the `promoted_by` model-version tag.
4. **Failure-Safe Operation Ordering**:
   - Step 1: Re-validate `compare()` immediately before mutations to prevent stale-session promotions.
   - Step 2: Retrieve and retain the active champion version.
   - Step 3: Preserve the former champion under `previous_champion` **before** touching the `champion` alias. If preservation fails, the active champion remains completely unmutated.
   - Step 4: Reassign `champion` to the challenger. If this alias update fails, the previous champion remains the active serving version.
   - Step 5: Apply the `promoted_by` tag. If tagging fails after the alias mutation, promotion is aborted and the champion alias is restored to the previous version; if restoration also fails, a critical alert is surfaced for immediate operator intervention.
   - Step 6: Return success only when all required operations have cleanly completed.

### Rollback Procedure & Operational Semantics
- If a promoted champion displays anomalies in production, operators execute:
  ```bash
  make rollback ACTOR=<operator>
  # or: python scripts/champion.py --rollback --actor <operator>
  ```
- Operator identity must be explicitly provided; the Makefile does not default to `$USER`, and omitting `ACTOR`, supplying an empty string, or passing `unknown` is rejected by the CLI with exit code 2.
- The `champion` alias is reassigned to the version recorded under `previous_champion`.
- The operator's identity is stamped in the `rolled_back_by` model-version tag.
- If tagging fails during rollback, a partial rollback warning is surfaced noting that the alias was restored but tagging failed.

### MLflow Prerequisites & Limitations
- Requires an active MLflow tracking and model registry server (`MLFLOW_TRACKING_URI`).
- Supported backends: SQLite/PostgreSQL metadata store with artifact storage (e.g., S3/local file store).
- **Non-Transactional Aliases**: MLflow model registry alias and tag updates are independent HTTP/database operations and are **not** ACID transactional. The registry implementation uses failure-safe ordering and compensating restorations to protect production routing.
- **Metric Extraction Error Handling**: Server, network, and permission errors during run retrieval surface immediately. Model version tags serve as an extraction fallback only when run metadata is archived or unavailable (`RESOURCE_DOES_NOT_EXIST` / `NOT_FOUND`). Genuinely missing metric data rejects candidate comparison.
