"""Unit tests for M3 Step 3.11: Champion/Challenger registry and promotion CLI."""
from __future__ import annotations

import pathlib
from typing import Any
import uuid
import pytest
from sklearn.linear_model import LogisticRegression

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from aegis_ml.models.calibration import Calibrator
from aegis_ml.training import registry
from scripts import champion as champion_cli


@pytest.fixture(scope="module")
def shared_mlflow_env(tmp_path_factory):
    """Sets up an isolated SQLite-backed MLflow tracking and registry database once for the module."""
    tmpdir = tmp_path_factory.mktemp("mlflow_shared")
    db_file = tmpdir / "mlflow_unit.db"
    uri = f"sqlite:///{db_file}"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    return uri, client


@pytest.fixture
def local_mlflow(shared_mlflow_env, monkeypatch):
    """Provides a fresh model name and configured environment per test."""
    uri, client = shared_mlflow_env
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    model_name = f"test_fraud_{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(registry, "MODEL", model_name)
    return client


def _log_test_run(
    metrics: dict[str, float] | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[str, Any, Calibrator]:
    """Helper to log a test run and return (run_id, model, calibrator)."""
    model = LogisticRegression().fit([[0.0], [1.0]], [0, 1])
    calibrator = Calibrator().fit([0.2, 0.8], [0, 1])

    with mlflow.start_run() as run:
        if metrics:
            mlflow.log_metrics(metrics)
        if params:
            mlflow.log_params(params)
        run_id = run.info.run_id

    return run_id, model, calibrator


# --- Unit Test Suite Covering All 18 Requirements ---


def test_missing_champion_and_challenger_aliases(local_mlflow):
    """1. Missing champion and challenger aliases: compare() safely handles unconfigured state."""
    decision = registry.compare(client=local_mlflow)
    assert not decision.changed
    assert "both champion and challenger aliases must be set" in decision.reason


def test_first_model_registration_assigns_both_aliases(local_mlflow):
    """2. First model registration assigns both champion and challenger aliases."""
    metrics = {
        "recall_at_1pct_fpr": 0.52,
        "lift_over_rules": 0.35,
        "pr_auc": 0.65,
        "ece": 0.03,
    }
    run_id, model, cal = _log_test_run(metrics=metrics)
    features = ["f1", "f2", "f3"]

    decision = registry.register_candidate(
        run_id=run_id,
        model=model,
        calibrator=cal,
        feature_names=features,
        metrics=metrics,
        client=local_mlflow,
    )

    assert decision.changed
    assert decision.version == "1"
    assert "first model: v1 is champion and challenger" in decision.reason

    champ = local_mlflow.get_model_version_by_alias(registry.MODEL, registry.CHAMPION)
    chal = local_mlflow.get_model_version_by_alias(registry.MODEL, registry.CHALLENGER)
    assert str(champ.version) == "1"
    assert str(chal.version) == "1"


def test_subsequent_registration_changes_challenger_preserves_champion(local_mlflow):
    """3. Subsequent registration updates challenger alias but preserves the existing champion."""
    metrics1 = {"recall_at_1pct_fpr": 0.52, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03}
    run1, m1, c1 = _log_test_run(metrics=metrics1)
    registry.register_candidate(run1, m1, c1, ["f1"], metrics1, client=local_mlflow)

    # Register candidate 2
    metrics2 = {"recall_at_1pct_fpr": 0.53, "lift_over_rules": 0.36, "pr_auc": 0.66, "ece": 0.03}
    run2, m2, c2 = _log_test_run(metrics=metrics2)
    decision2 = registry.register_candidate(run2, m2, c2, ["f1"], metrics2, client=local_mlflow)

    assert decision2.changed
    assert decision2.version == "2"
    assert "v2 registered as challenger (champion stays v1)" in decision2.reason

    champ = local_mlflow.get_model_version_by_alias(registry.MODEL, registry.CHAMPION)
    chal = local_mlflow.get_model_version_by_alias(registry.MODEL, registry.CHALLENGER)
    assert str(champ.version) == "1"
    assert str(chal.version) == "2"


def test_model_and_calibrator_logged_to_correct_run(local_mlflow):
    """4. Model and isotonic calibrator are logged as artifacts to the specified run_id."""
    metrics = {"recall_at_1pct_fpr": 0.52}
    run_id, model, cal = _log_test_run(metrics=metrics)

    registry.register_candidate(
        run_id=run_id,
        model=model,
        calibrator=cal,
        feature_names=["f1", "f2"],
        metrics=metrics,
        client=local_mlflow,
    )

    artifacts = [a.path for a in local_mlflow.list_artifacts(run_id)]
    assert "model" in artifacts
    assert "calibrator" in artifacts

    # Check calibrator contains MLmodel specification
    cal_artifacts = [a.path for a in local_mlflow.list_artifacts(run_id, "calibrator")]
    assert any("MLmodel" in a for a in cal_artifacts)


def test_feature_ordering_persisted_exactly(local_mlflow):
    """5. Feature ordering is persisted exactly in model/columns.json."""
    expected_features = ["user_recency", "device_sharing_7d", "ip_count_24h", "amount_zscore"]
    metrics = {"recall_at_1pct_fpr": 0.52}
    run_id, model, cal = _log_test_run(metrics=metrics)

    registry.register_candidate(
        run_id=run_id,
        model=model,
        calibrator=cal,
        feature_names=expected_features,
        metrics=metrics,
        client=local_mlflow,
    )

    columns_path = local_mlflow.download_artifacts(run_id, "model/columns.json")
    import json
    data = json.loads(pathlib.Path(columns_path).read_text())
    assert data == {"feature_names": expected_features}


def test_frequency_encoding_metadata_accessible(local_mlflow):
    """6. Frequency encoding dictionary remains accessible in model artifacts."""
    mapping = {"channel": {"web": 0.6, "pos": 0.4}, "region": {"US": 0.8, "EU": 0.2}}
    metrics = {"recall_at_1pct_fpr": 0.52}
    run_id, model, cal = _log_test_run(metrics=metrics)

    registry.register_candidate(
        run_id=run_id,
        model=model,
        calibrator=cal,
        feature_names=["f1"],
        metrics=metrics,
        client=local_mlflow,
        frequency_encoding=mapping,
    )

    meta_path = local_mlflow.download_artifacts(run_id, "model/frequency_encoding.json")
    import json
    data = json.loads(pathlib.Path(meta_path).read_text())
    assert data == mapping


def test_evaluation_metrics_correctly_attached(local_mlflow):
    """7. Evaluation metrics are formatted and attached as tags on the model version."""
    metrics = {
        "recall_at_1pct_fpr": 0.571234,
        "lift_over_rules": 0.4289,
        "pr_auc": 0.7234,
        "ece": 0.0215,
        "int_metric": 42,
    }
    run_id, model, cal = _log_test_run(metrics=metrics)

    decision = registry.register_candidate(
        run_id=run_id,
        model=model,
        calibrator=cal,
        feature_names=["f1"],
        metrics=metrics,
        client=local_mlflow,
    )

    mv = local_mlflow.get_model_version(registry.MODEL, decision.version)
    tags = mv.tags
    assert tags["recall_at_1pct_fpr"] == "0.571234"
    assert tags["lift_over_rules"] == "0.428900"
    assert tags["pr_auc"] == "0.723400"
    assert tags["ece"] == "0.021500"
    assert tags["int_metric"] == "42"


def test_comparison_succeeds_when_all_gates_and_relative_lift_pass(local_mlflow):
    """8. Comparison succeeds when all gates pass and relative lift >= 2%."""
    # Champion: recall 0.50
    m_champ = {"recall_at_1pct_fpr": 0.50, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r1, m1, c1 = _log_test_run(metrics=m_champ)
    registry.register_candidate(r1, m1, c1, ["f"], m_champ, client=local_mlflow)

    # Challenger: recall 0.53 (+6% relative lift > 2%)
    m_chal = {"recall_at_1pct_fpr": 0.53, "lift_over_rules": 0.38, "pr_auc": 0.68, "ece": 0.025, "typology_detection": 0.85}
    r2, m2, c2 = _log_test_run(metrics=m_chal)
    registry.register_candidate(r2, m2, c2, ["f"], m_chal, client=local_mlflow)

    decision = registry.compare(client=local_mlflow)
    assert decision.changed
    assert decision.version == "2"
    assert "0.5000 -> 0.5300 (+6.00%) - promote" in decision.reason


def test_comparison_rejects_challenger_failing_gates(local_mlflow):
    """9. Comparison rejects a challenger that fails any evaluation gate."""
    # Champion
    m_champ = {"recall_at_1pct_fpr": 0.50, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r1, m1, c1 = _log_test_run(metrics=m_champ)
    registry.register_candidate(r1, m1, c1, ["f"], m_champ, client=local_mlflow)

    # Challenger has high recall (0.60), but fails ECE gate (> 0.05)
    m_chal = {"recall_at_1pct_fpr": 0.60, "lift_over_rules": 0.40, "pr_auc": 0.70, "ece": 0.075, "typology_detection": 0.85}
    r2, m2, c2 = _log_test_run(metrics=m_chal)
    registry.register_candidate(r2, m2, c2, ["f"], m_chal, client=local_mlflow)

    decision = registry.compare(client=local_mlflow)
    assert not decision.changed
    assert "challenger failed gates" in decision.reason
    assert "ece=0.0750" in decision.reason


def test_comparison_rejects_insufficient_relative_lift(local_mlflow):
    """10. Comparison rejects lift inside the 2% noise band."""
    # Champion: recall 0.50
    m_champ = {"recall_at_1pct_fpr": 0.50, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r1, m1, c1 = _log_test_run(metrics=m_champ)
    registry.register_candidate(r1, m1, c1, ["f"], m_champ, client=local_mlflow)

    # Challenger: recall 0.505 (+1% relative lift < 2%)
    m_chal = {"recall_at_1pct_fpr": 0.505, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r2, m2, c2 = _log_test_run(metrics=m_chal)
    registry.register_candidate(r2, m2, c2, ["f"], m_chal, client=local_mlflow)

    decision = registry.compare(client=local_mlflow)
    assert not decision.changed
    assert "relative lift is inside the noise band (2%)" in decision.reason


def test_comparison_rejects_ties_and_missing_metrics(local_mlflow):
    """11. Comparison rejects ties and missing metrics."""
    # Case A: Same version for both champion and challenger
    m_champ = {"recall_at_1pct_fpr": 0.50, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r1, m1, c1 = _log_test_run(metrics=m_champ)
    registry.register_candidate(r1, m1, c1, ["f"], m_champ, client=local_mlflow)

    # Same version 1
    d_same = registry.compare(client=local_mlflow)
    assert not d_same.changed
    assert "same version" in d_same.reason

    # Case B: Exactly identical performance (tie) on version 2
    r2, m2, c2 = _log_test_run(metrics=m_champ)
    registry.register_candidate(r2, m2, c2, ["f"], m_champ, client=local_mlflow)

    d_tie = registry.compare(client=local_mlflow)
    assert not d_tie.changed
    assert "does not beat champion" in d_tie.reason

    # Case C: Missing primary metric
    r3, m3, c3 = _log_test_run(metrics={"lift_over_rules": 0.40})
    registry.register_candidate(r3, m3, c3, ["f"], {"lift_over_rules": 0.40}, client=local_mlflow)
    d_missing = registry.compare(client=local_mlflow)
    assert not d_missing.changed
    assert "missing" in d_missing.reason


def test_promotion_rechecks_rule_before_changing_alias(local_mlflow):
    """12. Promotion re-checks compare() immediately before mutating the alias."""
    # First model (v1)
    m1 = {"recall_at_1pct_fpr": 0.50, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r1, model1, c1 = _log_test_run(metrics=m1)
    registry.register_candidate(r1, model1, c1, ["f"], m1, client=local_mlflow)

    # Candidate 2 (v2) has tiny lift (+0.5%)
    m2 = {"recall_at_1pct_fpr": 0.5025, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r2, model2, c2 = _log_test_run(metrics=m2)
    registry.register_candidate(r2, model2, c2, ["f"], m2, client=local_mlflow)

    # Attempting to promote v2 directly must refuse because rule is rechecked
    d = registry.promote_challenger(actor="alice", client=local_mlflow)
    assert not d.changed
    assert "refusing to promote" in d.reason

    # Verify champion alias is STILL v1
    champ = local_mlflow.get_model_version_by_alias(registry.MODEL, registry.CHAMPION)
    assert str(champ.version) == "1"


def test_promotion_records_promoted_by(local_mlflow):
    """13. Promotion tags the promoted version with the human operator identity."""
    m1 = {"recall_at_1pct_fpr": 0.50, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r1, model1, c1 = _log_test_run(metrics=m1)
    registry.register_candidate(r1, model1, c1, ["f"], m1, client=local_mlflow)

    m2 = {"recall_at_1pct_fpr": 0.55, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r2, model2, c2 = _log_test_run(metrics=m2)
    registry.register_candidate(r2, model2, c2, ["f"], m2, client=local_mlflow)

    d = registry.promote_challenger(actor="lead_analyst_p.raman", client=local_mlflow)
    assert d.changed
    assert "lead_analyst_p.raman" in d.reason

    mv2 = local_mlflow.get_model_version(registry.MODEL, "2")
    assert mv2.tags.get("promoted_by") == "lead_analyst_p.raman"


def test_previous_champion_is_preserved(local_mlflow):
    """14. When champion is updated, previous champion is saved under previous_champion alias."""
    m1 = {"recall_at_1pct_fpr": 0.50, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r1, model1, c1 = _log_test_run(metrics=m1)
    registry.register_candidate(r1, model1, c1, ["f"], m1, client=local_mlflow)

    m2 = {"recall_at_1pct_fpr": 0.55, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r2, model2, c2 = _log_test_run(metrics=m2)
    registry.register_candidate(r2, model2, c2, ["f"], m2, client=local_mlflow)

    registry.promote_challenger(actor="operator", client=local_mlflow)

    prev = local_mlflow.get_model_version_by_alias(registry.MODEL, registry.PREVIOUS_CHAMPION)
    current = local_mlflow.get_model_version_by_alias(registry.MODEL, registry.CHAMPION)
    assert str(prev.version) == "1"
    assert str(current.version) == "2"


def test_rollback_restores_previous_champion(local_mlflow):
    """15. Rollback restores the previous champion and records rolled_back_by tag."""
    m1 = {"recall_at_1pct_fpr": 0.50, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r1, model1, c1 = _log_test_run(metrics=m1)
    registry.register_candidate(r1, model1, c1, ["f"], m1, client=local_mlflow)

    m2 = {"recall_at_1pct_fpr": 0.55, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r2, model2, c2 = _log_test_run(metrics=m2)
    registry.register_candidate(r2, model2, c2, ["f"], m2, client=local_mlflow)
    registry.promote_challenger(actor="promoter", client=local_mlflow)

    # Perform rollback
    rb_decision = registry.rollback(actor="rollback_agent", client=local_mlflow)
    assert rb_decision.changed
    assert "champion rolled back to v1 by rollback_agent" in rb_decision.reason

    champ = local_mlflow.get_model_version_by_alias(registry.MODEL, registry.CHAMPION)
    assert str(champ.version) == "1"
    mv1 = local_mlflow.get_model_version(registry.MODEL, "1")
    assert mv1.tags.get("rolled_back_by") == "rollback_agent"


def test_rollback_safely_handles_missing_previous_champion(local_mlflow):
    """16. Rollback fails safely when no previous champion exists."""
    # No models registered at all
    d1 = registry.rollback(actor="tester", client=local_mlflow)
    assert not d1.changed
    assert "no previous_champion alias to roll back to" in d1.reason

    # Only one model registered (so it has no predecessor)
    m1 = {"recall_at_1pct_fpr": 0.50, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03}
    r1, model1, c1 = _log_test_run(metrics=m1)
    registry.register_candidate(r1, model1, c1, ["f"], m1, client=local_mlflow)

    d2 = registry.rollback(actor="tester", client=local_mlflow)
    assert not d2.changed
    assert "no previous_champion alias to roll back to" in d2.reason


def test_registry_server_failures_not_misreported_as_missing_aliases(monkeypatch):
    """17. Server, network, or auth failures are raised and not masked as missing aliases."""
    class BrokenClient:
        def get_model_version_by_alias(self, model, alias):
            raise MlflowException("500 Internal Server Error", error_code="INTERNAL_ERROR")

    client = BrokenClient()
    with pytest.raises(MlflowException, match="500 Internal Server Error"):
        registry._alias_version(client, "champion")

    class AuthFailClient:
        def get_model_version_by_alias(self, model, alias):
            raise MlflowException("403 Forbidden: Invalid token", error_code="PERMISSION_DENIED")

    with pytest.raises(MlflowException, match="403 Forbidden"):
        registry._alias_version(AuthFailClient(), "champion")


def test_cli_champion_operations_and_actor_validation(local_mlflow, capsys):
    """18. CLI champion.py handles compare, promote, rollback, actor validation, and exit codes."""
    # 1. Missing actor for promote
    ret_promote_no_actor = champion_cli.main(["--promote"])
    assert ret_promote_no_actor == 2
    err = capsys.readouterr().err
    assert "--actor must be explicitly specified" in err

    # 2. Unknown actor for rollback
    ret_rb_unknown_actor = champion_cli.main(["--rollback", "--actor", "unknown"])
    assert ret_rb_unknown_actor == 2
    err = capsys.readouterr().err
    assert "--actor must be explicitly specified" in err

    # 3. Conflicting flags
    with pytest.raises(SystemExit):
        champion_cli.main(["--compare", "--promote"])

    # 4. Read-only compare via CLI
    # Register v1
    m1 = {"recall_at_1pct_fpr": 0.50, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r1, model1, c1 = _log_test_run(metrics=m1)
    registry.register_candidate(r1, model1, c1, ["f"], m1, client=local_mlflow)

    # Register winning v2
    m2 = {"recall_at_1pct_fpr": 0.54, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r2, model2, c2 = _log_test_run(metrics=m2)
    registry.register_candidate(r2, model2, c2, ["f"], m2, client=local_mlflow)

    ret_compare = champion_cli.main(["--compare"])
    out = capsys.readouterr().out
    assert ret_compare == 0
    assert "0.5000 -> 0.5400 (+8.00%) - promote" in out

    # Ensure compare did NOT mutate champion
    champ = local_mlflow.get_model_version_by_alias(registry.MODEL, registry.CHAMPION)
    assert str(champ.version) == "1"

    # 5. Promote via CLI
    ret_promote = champion_cli.main(["--promote", "--actor", "cli_tester"])
    out = capsys.readouterr().out
    assert ret_promote == 0
    assert "champion v1 -> v2 by cli_tester" in out

    champ_after = local_mlflow.get_model_version_by_alias(registry.MODEL, registry.CHAMPION)
    assert str(champ_after.version) == "2"

    # 6. Rollback via CLI
    ret_rollback = champion_cli.main(["--rollback", "--actor", "cli_rollback_tester"])
    out = capsys.readouterr().out
    assert ret_rollback == 0
    assert "champion rolled back to v1 by cli_rollback_tester" in out

    champ_rolled = local_mlflow.get_model_version_by_alias(registry.MODEL, registry.CHAMPION)
    assert str(champ_rolled.version) == "1"


# --- Targeted Failure-Injection & Error Handling Tests (Corrections 2 & 3) ---


def test_promotion_preserves_previous_champion_before_mutating_champion(local_mlflow):
    """19. Successful promotion sets previous_champion to former champion and updates champion."""
    m1 = {"recall_at_1pct_fpr": 0.50, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r1, model1, c1 = _log_test_run(metrics=m1)
    registry.register_candidate(r1, model1, c1, ["f"], m1, client=local_mlflow)

    m2 = {"recall_at_1pct_fpr": 0.55, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r2, model2, c2 = _log_test_run(metrics=m2)
    registry.register_candidate(r2, model2, c2, ["f"], m2, client=local_mlflow)

    decision = registry.promote_challenger(actor="lead_analyst", client=local_mlflow)
    assert decision.changed
    assert "champion v1 -> v2 by lead_analyst" in decision.reason

    champ = local_mlflow.get_model_version_by_alias(registry.MODEL, registry.CHAMPION)
    prev = local_mlflow.get_model_version_by_alias(registry.MODEL, registry.PREVIOUS_CHAMPION)
    assert str(champ.version) == "2"
    assert str(prev.version) == "1"


def test_promotion_failure_setting_previous_champion_aborts_without_changing_champion(local_mlflow, monkeypatch):
    """20. Failure setting previous_champion aborts promotion before touching champion alias."""
    m1 = {"recall_at_1pct_fpr": 0.50, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r1, model1, c1 = _log_test_run(metrics=m1)
    registry.register_candidate(r1, model1, c1, ["f"], m1, client=local_mlflow)

    m2 = {"recall_at_1pct_fpr": 0.55, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r2, model2, c2 = _log_test_run(metrics=m2)
    registry.register_candidate(r2, model2, c2, ["f"], m2, client=local_mlflow)

    orig_set_alias = local_mlflow.set_registered_model_alias

    def failing_set_alias(name, alias, version):
        if alias == registry.PREVIOUS_CHAMPION:
            raise MlflowException("Simulated error setting previous_champion", error_code="INTERNAL_ERROR")
        return orig_set_alias(name, alias, version)

    monkeypatch.setattr(local_mlflow, "set_registered_model_alias", failing_set_alias)

    with pytest.raises(MlflowException, match="Simulated error setting previous_champion"):
        registry.promote_challenger(actor="alice", client=local_mlflow)

    # Active champion must remain untouched as v1
    champ = local_mlflow.get_model_version_by_alias(registry.MODEL, registry.CHAMPION)
    assert str(champ.version) == "1"


def test_promotion_failure_changing_champion_leaves_previous_champion_active(local_mlflow, monkeypatch):
    """21. Failure setting champion alias surfaces failure and ensures previous champion is active."""
    m1 = {"recall_at_1pct_fpr": 0.50, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r1, model1, c1 = _log_test_run(metrics=m1)
    registry.register_candidate(r1, model1, c1, ["f"], m1, client=local_mlflow)

    m2 = {"recall_at_1pct_fpr": 0.55, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r2, model2, c2 = _log_test_run(metrics=m2)
    registry.register_candidate(r2, model2, c2, ["f"], m2, client=local_mlflow)

    orig_set_alias = local_mlflow.set_registered_model_alias

    def failing_champ_alias(name, alias, version):
        if alias == registry.CHAMPION and version == "2":
            raise MlflowException("Simulated error setting champion to v2", error_code="INTERNAL_ERROR")
        return orig_set_alias(name, alias, version)

    monkeypatch.setattr(local_mlflow, "set_registered_model_alias", failing_champ_alias)

    with pytest.raises(RuntimeError, match="Failed to update champion alias to v2"):
        registry.promote_challenger(actor="alice", client=local_mlflow)

    # Champion must still be v1
    champ = local_mlflow.get_model_version_by_alias(registry.MODEL, registry.CHAMPION)
    assert str(champ.version) == "1"


def test_promotion_failure_tagging_restores_previous_champion(local_mlflow, monkeypatch):
    """22. Failure setting promoted_by tag aborts promotion and safely restores former champion."""
    m1 = {"recall_at_1pct_fpr": 0.50, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r1, model1, c1 = _log_test_run(metrics=m1)
    registry.register_candidate(r1, model1, c1, ["f"], m1, client=local_mlflow)

    m2 = {"recall_at_1pct_fpr": 0.55, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r2, model2, c2 = _log_test_run(metrics=m2)
    registry.register_candidate(r2, model2, c2, ["f"], m2, client=local_mlflow)

    def failing_set_tag(name, version, key, value):
        if key == "promoted_by":
            raise MlflowException("Simulated network failure writing promoted_by tag", error_code="INTERNAL_ERROR")
        return local_mlflow.set_model_version_tag(name, version, key, value)

    monkeypatch.setattr(local_mlflow, "set_model_version_tag", failing_set_tag)

    with pytest.raises(RuntimeError, match="Promotion aborted: failed to set 'promoted_by' tag.*Restored champion to v1"):
        registry.promote_challenger(actor="alice", client=local_mlflow)

    # Champion must be restored to v1
    champ = local_mlflow.get_model_version_by_alias(registry.MODEL, registry.CHAMPION)
    assert str(champ.version) == "1"


def test_promotion_critical_double_failure_reported(local_mlflow, monkeypatch):
    """23. Failure during tagging AND failure during subsequent restoration reports critical partial state."""
    m1 = {"recall_at_1pct_fpr": 0.50, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r1, model1, c1 = _log_test_run(metrics=m1)
    registry.register_candidate(r1, model1, c1, ["f"], m1, client=local_mlflow)

    m2 = {"recall_at_1pct_fpr": 0.55, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r2, model2, c2 = _log_test_run(metrics=m2)
    registry.register_candidate(r2, model2, c2, ["f"], m2, client=local_mlflow)

    orig_set_alias = local_mlflow.set_registered_model_alias
    call_counts = {"set_alias": 0}

    def failing_restore_alias(name, alias, version):
        call_counts["set_alias"] += 1
        # Call 1: previous_champion -> 1
        # Call 2: champion -> 2
        # Call 3: restoration of champion -> 1 fails!
        if call_counts["set_alias"] == 3:
            raise MlflowException("Simulated backend crash during restoration", error_code="INTERNAL_ERROR")
        return orig_set_alias(name, alias, version)

    monkeypatch.setattr(local_mlflow, "set_registered_model_alias", failing_restore_alias)
    monkeypatch.setattr(
        local_mlflow,
        "set_model_version_tag",
        lambda *args, **kwargs: (_ for _ in ()).throw(MlflowException("Tag failure", error_code="INTERNAL_ERROR")),
    )

    with pytest.raises(RuntimeError, match="CRITICAL PARTIAL STATE"):
        registry.promote_challenger(actor="alice", client=local_mlflow)


def test_rollback_failure_tagging_reports_partial_state(local_mlflow, monkeypatch):
    """24. Rollback tagging failure reports partial rollback state."""
    m1 = {"recall_at_1pct_fpr": 0.50, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r1, model1, c1 = _log_test_run(metrics=m1)
    registry.register_candidate(r1, model1, c1, ["f"], m1, client=local_mlflow)

    m2 = {"recall_at_1pct_fpr": 0.55, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r2, model2, c2 = _log_test_run(metrics=m2)
    registry.register_candidate(r2, model2, c2, ["f"], m2, client=local_mlflow)
    registry.promote_challenger(actor="promoter", client=local_mlflow)

    monkeypatch.setattr(
        local_mlflow,
        "set_model_version_tag",
        lambda *args, **kwargs: (_ for _ in ()).throw(MlflowException("Tag write failure", error_code="INTERNAL_ERROR")),
    )

    with pytest.raises(RuntimeError, match="PARTIAL ROLLBACK: champion alias was restored to v1"):
        registry.rollback(actor="bob", client=local_mlflow)

    champ = local_mlflow.get_model_version_by_alias(registry.MODEL, registry.CHAMPION)
    assert str(champ.version) == "1"


def test_extract_metrics_surfaces_operational_mlflow_exception(local_mlflow, monkeypatch):
    """25. Operational MLflowException (e.g. 500 error) during get_run is raised, not swallowed."""
    m1 = {"recall_at_1pct_fpr": 0.50, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r1, model1, c1 = _log_test_run(metrics=m1)
    registry.register_candidate(r1, model1, c1, ["f"], m1, client=local_mlflow)

    m2 = {"recall_at_1pct_fpr": 0.55, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r2, model2, c2 = _log_test_run(metrics=m2)
    registry.register_candidate(r2, model2, c2, ["f"], m2, client=local_mlflow)

    monkeypatch.setattr(
        local_mlflow,
        "get_run",
        lambda run_id: (_ for _ in ()).throw(MlflowException("500 Internal Server Error", error_code="INTERNAL_ERROR")),
    )

    with pytest.raises(MlflowException, match="500 Internal Server Error"):
        registry.compare(client=local_mlflow)


def test_extract_metrics_surfaces_network_connection_error(local_mlflow, monkeypatch):
    """26. Network connectivity failure during get_run is surfaced rather than swallowed."""
    m1 = {"recall_at_1pct_fpr": 0.50, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r1, model1, c1 = _log_test_run(metrics=m1)
    registry.register_candidate(r1, model1, c1, ["f"], m1, client=local_mlflow)

    m2 = {"recall_at_1pct_fpr": 0.55, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r2, model2, c2 = _log_test_run(metrics=m2)
    registry.register_candidate(r2, model2, c2, ["f"], m2, client=local_mlflow)

    monkeypatch.setattr(
        local_mlflow,
        "get_run",
        lambda run_id: (_ for _ in ()).throw(ConnectionError("Connection refused by MLflow server")),
    )

    with pytest.raises(ConnectionError, match="Connection refused"):
        registry.compare(client=local_mlflow)


def test_extract_metrics_falls_back_to_tags_when_run_does_not_exist(local_mlflow, monkeypatch):
    """27. When run metadata genuinely does not exist (e.g. deleted), falls back cleanly to version tags."""
    m1 = {"recall_at_1pct_fpr": 0.50, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r1, model1, c1 = _log_test_run(metrics=m1)
    registry.register_candidate(r1, model1, c1, ["f"], m1, client=local_mlflow)

    m2 = {"recall_at_1pct_fpr": 0.55, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r2, model2, c2 = _log_test_run(metrics=m2)
    registry.register_candidate(r2, model2, c2, ["f"], m2, client=local_mlflow)

    # Simulate deleted/missing run in MLflow store
    from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

    monkeypatch.setattr(
        local_mlflow,
        "get_run",
        lambda run_id: (_ for _ in ()).throw(MlflowException("Run not found", error_code=RESOURCE_DOES_NOT_EXIST)),
    )

    decision = registry.compare(client=local_mlflow)
    assert decision.changed
    assert "promote" in decision.reason


def test_genuinely_missing_metrics_rejects_comparison(local_mlflow):
    """28. Genuinely missing gate metric rejects comparison with explanatory reason."""
    # Champion has all metrics
    m1 = {"recall_at_1pct_fpr": 0.50, "lift_over_rules": 0.35, "pr_auc": 0.65, "ece": 0.03, "typology_detection": 0.85}
    r1, model1, c1 = _log_test_run(metrics=m1)
    registry.register_candidate(r1, model1, c1, ["f"], m1, client=local_mlflow)

    # Challenger is missing ece
    m2_incomplete = {"recall_at_1pct_fpr": 0.55, "lift_over_rules": 0.35, "pr_auc": 0.65, "typology_detection": 0.85}
    r2, model2, c2 = _log_test_run(metrics=m2_incomplete)
    registry.register_candidate(r2, model2, c2, ["f"], m2_incomplete, client=local_mlflow)

    decision = registry.compare(client=local_mlflow)
    assert not decision.changed
    assert "missing" in decision.reason
    assert "ece" in decision.reason
