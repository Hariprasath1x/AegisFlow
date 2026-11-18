"""Champion/challenger against a real MLflow server. Skipped without
MLFLOW_TRACKING_URI. Uses a throwaway model name so it cannot touch the real one.
"""
from __future__ import annotations

import os
import uuid
import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("MLFLOW_TRACKING_URI"),
    reason="needs the MLflow server",
)

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from aegis_ml.models.calibration import Calibrator  # noqa: E402
from aegis_ml.training import registry  # noqa: E402


@pytest.fixture()
def scratch_model(monkeypatch):
    name = f"test-aegis-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(registry, "MODEL", name)
    yield name
    client = MlflowClient()
    try:
        client.delete_registered_model(name)
    except Exception:  # noqa: BLE001
        pass


def _log(
    recall: float,
    lift: float = 0.35,
    pr_auc: float = 0.71,
    ece: float = 0.03,
    typology: float = 0.85,
):
    model = LogisticRegression().fit([[0.0], [1.0]], [0, 1])
    metrics = {
        "recall_at_1pct_fpr": recall,
        "lift_over_rules": lift,
        "pr_auc": pr_auc,
        "ece": ece,
        "typology_detection": typology,
    }
    with mlflow.start_run() as run:
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(model, artifact_path="model")
    return run.info.run_id, model, metrics


def test_first_model_becomes_both_aliases(scratch_model):
    run_id, model, metrics = _log(0.50)
    cal = Calibrator().fit([0.1, 0.9], [0, 1])
    d = registry.register_candidate(
        run_id,
        model,
        cal,
        ["f"],
        metrics,
        model_name=scratch_model,
    )
    assert d.changed and "champion and challenger" in d.reason
    client = MlflowClient()
    champ_mv = client.get_model_version_by_alias(scratch_model, "champion")
    chal_mv = client.get_model_version_by_alias(scratch_model, "challenger")
    assert str(champ_mv.version) == str(d.version)
    assert str(chal_mv.version) == str(d.version)


def test_a_tiny_lift_does_not_promote(scratch_model):
    cal = Calibrator().fit([0.1, 0.9], [0, 1])
    run_a, model_a, metrics_a = _log(0.50)
    registry.register_candidate(
        run_a,
        model_a,
        cal,
        ["f"],
        metrics_a,
        model_name=scratch_model,
    )
    run_b, model_b, metrics_b = _log(0.505)
    d_b = registry.register_candidate(
        run_b,
        model_b,
        cal,
        ["f"],
        metrics_b,
        model_name=scratch_model,
    )
    assert d_b.changed and "champion stays" in d_b.reason

    client = MlflowClient()
    assert str(client.get_model_version_by_alias(scratch_model, "champion").version) == "1"
    assert str(client.get_model_version_by_alias(scratch_model, "challenger").version) == "2"

    decision = registry.compare(model_name=scratch_model)
    assert not decision.changed and "noise band" in decision.reason

    promote_dec = registry.promote_challenger("tester", model_name=scratch_model)
    assert not promote_dec.changed
    assert "refusing to promote" in promote_dec.reason


def test_a_winning_challenger_is_promoted_and_rollback_restores(scratch_model):
    cal = Calibrator().fit([0.1, 0.9], [0, 1])
    run_a, model_a, metrics_a = _log(0.50)
    registry.register_candidate(
        run_a,
        model_a,
        cal,
        ["f"],
        metrics_a,
        model_name=scratch_model,
    )
    # 0.50 -> 0.55 is 10% relative lift (> 2% threshold)
    run_b, model_b, metrics_b = _log(0.55)
    registry.register_candidate(
        run_b,
        model_b,
        cal,
        ["f"],
        metrics_b,
        model_name=scratch_model,
    )

    decision = registry.compare(model_name=scratch_model)
    assert decision.changed
    assert "promote" in decision.reason

    # Promote challenger
    promoted = registry.promote_challenger("lead_analyst", model_name=scratch_model)
    assert promoted.changed
    assert "lead_analyst" in promoted.reason

    client = MlflowClient()
    champ_mv = client.get_model_version_by_alias(scratch_model, "champion")
    prev_mv = client.get_model_version_by_alias(scratch_model, "previous_champion")
    assert str(champ_mv.version) == "2"
    assert str(prev_mv.version) == "1"
    assert champ_mv.tags.get("promoted_by") == "lead_analyst"

    # Rollback
    rolled = registry.rollback("operator_bob", model_name=scratch_model)
    assert rolled.changed
    assert "operator_bob" in rolled.reason

    restored_champ = client.get_model_version_by_alias(scratch_model, "champion")
    assert str(restored_champ.version) == "1"
    assert restored_champ.tags.get("rolled_back_by") == "operator_bob"
