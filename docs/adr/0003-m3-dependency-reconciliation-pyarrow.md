# ADR-0003: Reconcile PyArrow Pin for MLflow 2.17.2 Compatibility in Milestone 3

**Status:** Accepted
**Date:** 2026-11-17
**Scope:** Milestone 3 (`feature/m3-fraud-models`)

## 1. Context & Incompatible Constraints

The `AegisFlow_M3_Implementation_Guide.pdf` (page 3) specifies the following training stack:
- `mlflow==2.17.2`
- `pyarrow==18.1.0`
- `pandas==2.2.3`
- `numpy==1.26.4`
- `scikit-learn==1.5.2`
- `xgboost==2.1.3`
- `lightgbm==4.5.0`
- `shap==0.46.0`
- `networkx==3.4.2`
- `dvc==3.56.0`
- `pyyaml==6.0.2`
- `sqlalchemy==2.0.36`
- `psycopg[binary]==3.2.3`

During Step 3.1 setup, pip dependency resolution failed closed:
```text
ERROR: Cannot install -r ml/requirements-train.txt (line 10) and pyarrow==18.1.0 because these package versions have conflicting dependencies.
The conflict is caused by:
    The user requested pyarrow==18.1.0
    mlflow 2.17.2 depends on pyarrow<18 and >=4.0.0
```
Package metadata analysis confirms that `mlflow==2.17.2` strictly requires `pyarrow<18,>=4.0.0`. The intersection with `pyarrow==18.1.0` is empty:
$$\{18.1.0\} \cap [4.0.0, 18.0.0) = \emptyset$$
The original specification set is mathematically unresolvable in a single Python environment.

## 2. Decision

Under explicit authorization, adopt the following targeted reconciliation in `ml/requirements-train.txt`:
1. Retain `mlflow==2.17.2`.
2. Reconcile `pyarrow` to `17.0.0` (adjusting from the guide's listed `18.1.0`).
3. Preserve every other guide dependency pin exactly as specified:
   - `xgboost==2.1.3`
   - `lightgbm==4.5.0`
   - `scikit-learn==1.5.2`
   - `shap==0.46.0`
   - `pandas==2.2.3`
   - `numpy==1.26.4`
   - `networkx==3.4.2`
   - `dvc==3.56.0`
   - `pyyaml==6.0.2`
   - `sqlalchemy==2.0.36`
   - `psycopg[binary]==3.2.3`
   - `pytest==8.3.4`

## 3. Rationale

1. **Why Retain MLflow 2.17.2:**
   The guide's retrain CI workflow (`.github/workflows/retrain.yml`, page 24) deploys the official Docker service image `ghcr.io/mlflow/mlflow:v2.17.2`. Retaining `mlflow==2.17.2` for the client ensures strict client/server protocol compatibility and avoids registry/artifact API drift. Upgrading MLflow to 2.18+ or 3.x risks cascading upgrades to core libraries (e.g., NumPy 2.x, Scikit-learn 1.9+).

2. **Why Select PyArrow 17.0.0:**
   - Satisfies `mlflow 2.17.2`'s `pyarrow<18,>=4.0.0` requirement ($17.0.0 \in [4.0.0, 18.0.0)$).
   - Satisfies `pandas 2.2.3`'s parquet requirement (`pyarrow>=10.0.1`).
   - Conforms to Apache Parquet format 2.6/2.9 specification with identical schema, timestamp, and Arrow table serialization behavior to 18.1.0.
   - Pinned resolver test confirms zero dependency collisions and no cascading changes across the entire M3 and M2 test matrix.

## 4. Verification

The dependency graph with `pyarrow==17.0.0` was validated using `pip install --dry-run` and completed with exit code 0. All 13 core packages resolve cleanly without requiring CUDA/GPU libraries.
