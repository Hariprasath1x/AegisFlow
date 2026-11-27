"""Unit tests for M3 typology evaluation (Objective 5.4)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from aegis_ml.evaluation.typology_eval import (
    TYPOLOGIES,
    episode_detection,
    main,
)
from aegis_ml.scenarios.generate_typologies import (
    generate_typology_datasets,
    main as generator_main,
)


@pytest.fixture
def sample_episodes():
    return pd.DataFrame(
        [
            # Smurfing: 2 episodes
            {
                "episode_id": "smurf_1",
                "typology": "smurfing",
                "transaction_ids": ["tx_1", "tx_2", "tx_3", "tx_4", "tx_5", "tx_6", "tx_7", "tx_8", "tx_9", "tx_10"],
            },
            {
                "episode_id": "smurf_2",
                "typology": "smurfing",
                "transaction_ids": ["tx_11", "tx_12", "tx_13", "tx_14", "tx_15"],
            },
            # Layering: 1 episode
            {
                "episode_id": "layer_1",
                "typology": "layering",
                "transaction_ids": ["tx_21", "tx_22", "tx_23", "tx_24"],
            },
            # Round tripping: 1 episode
            {
                "episode_id": "round_1",
                "typology": "round_tripping",
                "transaction_ids": ["tx_31", "tx_32", "tx_33"],
            },
        ]
    )


def test_episode_detection_basic(sample_episodes):
    # smurf_1 has 10 txns; flag 3 (30%) -> detected
    # smurf_2 has 5 txns; flag 1 (20%) -> not detected
    # layer_1 has 4 txns; flag 2 (50%) -> detected
    # round_1 has 3 txns; flag 1 (33.3%) -> detected
    flagged = {"tx_1", "tx_2", "tx_3", "tx_11", "tx_21", "tx_22", "tx_31"}
    res = episode_detection(sample_episodes, flagged, min_share=0.30)

    assert res["smurfing"]["episodes"] == 2
    assert res["smurfing"]["detected"] == 1
    assert res["smurfing"]["rate"] == 0.50

    assert res["layering"]["episodes"] == 1
    assert res["layering"]["detected"] == 1
    assert res["layering"]["rate"] == 1.0

    assert res["round_tripping"]["episodes"] == 1
    assert res["round_tripping"]["detected"] == 1
    assert res["round_tripping"]["rate"] == 1.0

    # Overall rate = mean(0.50, 1.0, 1.0) = 2.5 / 3 = 0.8333...
    expected_overall = (0.50 + 1.0 + 1.0) / 3.0
    assert pytest.approx(res["overall"]["rate"], rel=1e-4) == expected_overall


def test_min_share_boundary():
    # 10 transactions: exactly 30% needed
    df = pd.DataFrame(
        [
            {
                "episode_id": "ep1",
                "typology": "smurfing",
                "transaction_ids": [f"t{i}" for i in range(10)],
            }
        ]
    )

    # 2/10 = 20% < 30% -> not detected
    res_2 = episode_detection(df, {"t0", "t1"}, min_share=0.30)
    assert res_2["smurfing"]["detected"] == 0
    assert res_2["smurfing"]["rate"] == 0.0

    # 3/10 = 30% >= 30% -> detected
    res_3 = episode_detection(df, {"t0", "t1", "t2"}, min_share=0.30)
    assert res_3["smurfing"]["detected"] == 1
    assert res_3["smurfing"]["rate"] == 1.0


def test_custom_min_share(sample_episodes):
    # With min_share=0.50:
    # smurf_1: 3/10 (30%) < 50% -> not detected
    # layer_1: 2/4 (50%) >= 50% -> detected
    flagged = {"tx_1", "tx_2", "tx_3", "tx_21", "tx_22"}
    res = episode_detection(sample_episodes, flagged, min_share=0.50)
    assert res["smurfing"]["detected"] == 0
    assert res["layering"]["detected"] == 1


def test_empty_episodes_dataframe():
    empty_df = pd.DataFrame(columns=["episode_id", "typology", "transaction_ids"])
    res = episode_detection(empty_df, {"tx_1"}, min_share=0.30)
    for typ in TYPOLOGIES:
        assert res[typ]["episodes"] == 0
        assert res[typ]["detected"] == 0
        assert res[typ]["rate"] is None
    assert res["overall"]["rate"] == 0.0


def test_missing_typologies_partial():
    # Only smurfing present
    df = pd.DataFrame(
        [
            {
                "episode_id": "s1",
                "typology": "smurfing",
                "transaction_ids": ["tx_1", "tx_2"],
            }
        ]
    )
    res = episode_detection(df, {"tx_1"}, min_share=0.30)
    assert res["smurfing"]["episodes"] == 1
    assert res["smurfing"]["detected"] == 1
    assert res["smurfing"]["rate"] == 1.0
    assert res["layering"]["rate"] is None
    assert res["round_tripping"]["rate"] is None
    # Overall should only average the non-None rates
    assert res["overall"]["rate"] == 1.0

def test_type_conversion_string_ids():
    # Integer transaction IDs matched against string flagged IDs
    df = pd.DataFrame(
        [
            {
                "episode_id": "s1",
                "typology": "smurfing",
                "transaction_ids": [101, 102, 103],
            }
        ]
    )
    res = episode_detection(df, {"101"}, min_share=0.30)
    assert res["smurfing"]["detected"] == 1
    assert res["smurfing"]["rate"] == 1.0


def test_input_validation():
    with pytest.raises(ValueError, match="episodes must be a pandas DataFrame"):
        episode_detection(None, {"tx_1"})

    with pytest.raises(ValueError, match="flagged_ids must be a set, list, or tuple"):
        episode_detection(pd.DataFrame(), "tx_1")

    with pytest.raises(ValueError, match="min_share must be a float between 0 and 1.0"):
        episode_detection(pd.DataFrame(), {"tx_1"}, min_share=0)

    with pytest.raises(ValueError, match="min_share must be a float between 0 and 1.0"):
        episode_detection(pd.DataFrame(), {"tx_1"}, min_share=1.5)


def test_main_missing_prerequisites(tmp_path, monkeypatch):
    monkeypatch.setattr("aegis_ml.evaluation.typology_eval.GROUND_TRUTH", tmp_path / "missing.parquet")
    exit_code = main()
    assert exit_code == 2

    # Provide ground truth but missing stream path
    gt_file = tmp_path / "gt.parquet"
    pd.DataFrame().to_parquet(gt_file)
    monkeypatch.setattr("aegis_ml.evaluation.typology_eval.GROUND_TRUTH", gt_file)
    monkeypatch.setattr("aegis_ml.evaluation.typology_eval.STREAM_PATH", tmp_path / "missing_stream.parquet")
    assert main() == 2

    # Provide stream path but missing metrics path
    stream_file = tmp_path / "stream.parquet"
    pd.DataFrame().to_parquet(stream_file)
    monkeypatch.setattr("aegis_ml.evaluation.typology_eval.STREAM_PATH", stream_file)
    monkeypatch.setattr("aegis_ml.evaluation.typology_eval.METRICS_PATH", tmp_path / "missing_metrics.json")
    assert main() == 2


def test_main_success_path_and_gate(tmp_path, monkeypatch):
    gt_file = tmp_path / "gt.parquet"
    stream_file = tmp_path / "stream.parquet"
    metrics_file = tmp_path / "metrics.json"

    episodes = pd.DataFrame(
        [
            {"episode_id": "e1", "typology": "smurfing", "transaction_ids": ["t1", "t2"]},
            {"episode_id": "e2", "typology": "layering", "transaction_ids": ["t3", "t4"]},
            {"episode_id": "e3", "typology": "round_tripping", "transaction_ids": ["t5", "t6"]},
        ]
    )
    episodes.to_parquet(gt_file)

    stream = pd.DataFrame(
        [
            {"transaction_id": "t1", "amount": 100.0},
            {"transaction_id": "t2", "amount": 200.0},
            {"transaction_id": "t3", "amount": 300.0},
            {"transaction_id": "t4", "amount": 400.0},
            {"transaction_id": "t5", "amount": 500.0},
            {"transaction_id": "t6", "amount": 600.0},
        ]
    )
    stream.to_parquet(stream_file)

    metrics_content = {
        "champion": "xgboost",
        "results": {
            "xgboost": {
                "threshold_at_1pct_fpr": 0.5,
            }
        },
        "frequency_encoding": {},
    }
    metrics_file.write_text(json.dumps(metrics_content))

    monkeypatch.setattr("aegis_ml.evaluation.typology_eval.GROUND_TRUTH", gt_file)
    monkeypatch.setattr("aegis_ml.evaluation.typology_eval.STREAM_PATH", stream_file)
    monkeypatch.setattr("aegis_ml.evaluation.typology_eval.METRICS_PATH", metrics_file)
    monkeypatch.setattr("aegis_ml.evaluation.typology_eval.REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr("aegis_ml.evaluation.typology_eval.ARTIFACTS_DIR", tmp_path / "artifacts")

    mock_model = MagicMock()
    # High probabilities -> all transactions flagged -> 100% detection -> exit 0
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9]] * 6)

    with patch("mlflow.sklearn.load_model", return_value=mock_model), \
         patch("mlflow.tracking.MlflowClient"), \
         patch("aegis_ml.training.matrix.build", return_value=(pd.DataFrame({"x": [1]*6}), {})):
        exit_code = main()
        assert exit_code == 0
        assert (tmp_path / "artifacts" / "typologies.json").exists()
        assert (tmp_path / "reports" / "m3_typologies.json").exists()

    # Low probabilities -> no transactions flagged -> 0% detection -> exit 1
    mock_model.predict_proba.return_value = np.array([[0.9, 0.1]] * 6)
    with patch("mlflow.sklearn.load_model", return_value=mock_model), \
         patch("mlflow.tracking.MlflowClient"), \
         patch("aegis_ml.training.matrix.build", return_value=(pd.DataFrame({"x": [1]*6}), {})):
        exit_code = main()
        assert exit_code == 1


def test_deterministic_generation_same_seed(tmp_path):
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"
    tx1, ep1 = generate_typology_datasets(out_dir=dir1, seed=42, episodes_per_typology=5, background_txns=50)
    tx2, ep2 = generate_typology_datasets(out_dir=dir2, seed=42, episodes_per_typology=5, background_txns=50)
    pd.testing.assert_frame_equal(tx1, tx2)
    pd.testing.assert_frame_equal(ep1, ep2)


def test_different_seeds_variation(tmp_path):
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"
    tx1, ep1 = generate_typology_datasets(out_dir=dir1, seed=100, episodes_per_typology=5, background_txns=50)
    tx2, ep2 = generate_typology_datasets(out_dir=dir2, seed=200, episodes_per_typology=5, background_txns=50)
    assert not tx1["amount"].equals(tx2["amount"])


def test_required_typology_presence(tmp_path):
    tx, ep = generate_typology_datasets(out_dir=tmp_path, seed=42, episodes_per_typology=4, background_txns=40)
    ep_types = set(ep["typology"])
    assert ep_types == set(TYPOLOGIES)
    tx_types = set(tx[tx["is_laundering"] == 1]["typology"])
    assert tx_types == set(TYPOLOGIES)


def test_episode_and_transaction_id_consistency(tmp_path):
    tx, ep = generate_typology_datasets(out_dir=tmp_path, seed=42, episodes_per_typology=5, background_txns=50)
    all_txn_ids = set(tx["transaction_id"])
    for _, row in ep.iterrows():
        for tid in row["transaction_ids"]:
            assert tid in all_txn_ids


def test_schema_correctness_and_matrix_build_compatibility(tmp_path):
    from aegis_ml.training.matrix import build
    tx, ep = generate_typology_datasets(out_dir=tmp_path, seed=42, episodes_per_typology=3, background_txns=30)
    assert tx["ts"].is_monotonic_increasing
    assert (tx["amount"] > 0).all()
    # Verify build() can compile features on this stream
    X, mapping = build(tx, mapping=None)
    assert len(X) == len(tx)
    assert len(X.columns) == 28


def test_generator_cli_execution(tmp_path):
    out_dir = tmp_path / "cli_out"
    rc = generator_main(["--out", str(out_dir), "--seed", "999", "--episodes", "2", "--background", "10"])
    assert rc == 0
    assert (out_dir / "synthetic_transactions.parquet").exists()
    assert (out_dir / "typology_episodes.parquet").exists()


def test_ci_workflow_retrain_validity():
    import pathlib
    import yaml
    workflow_path = pathlib.Path(".github/workflows/retrain.yml")
    assert workflow_path.exists()
    content = yaml.safe_load(workflow_path.read_text())
    assert content["name"] == "retrain"
    triggers = content.get("on") if "on" in content else content.get(True)
    assert triggers is not None
    assert "pull_request" in triggers
    assert "gates" in content["jobs"]
    steps = content["jobs"]["gates"]["steps"]
    step_names = [s.get("name") for s in steps if "name" in s]
    assert "Unit tests (no data needed)" in step_names
    assert "Pull data" in step_names
    assert "Train and gate" in step_names
    assert "Champion/challenger comparison (report only, never promotes)" in step_names
    pull_step = next(s for s in steps if s.get("name") == "Pull data")
    assert "localstore" in pull_step["run"]


def test_graph_and_deterministic_detectors(tmp_path):
    from aegis_ml.evaluation.typology_eval import (
        detect_smurfing,
        detect_layering,
        detect_round_tripping,
        detect_typologies,
    )
    tx, ep = generate_typology_datasets(out_dir=tmp_path, seed=42, episodes_per_typology=5, background_txns=50)
    flagged_s = detect_smurfing(tx)
    flagged_l = detect_layering(tx)
    flagged_r = detect_round_tripping(tx)
    flagged_all = detect_typologies(tx)

    assert len(flagged_s) > 0
    assert len(flagged_l) > 0
    assert len(flagged_r) > 0
    assert flagged_all == (flagged_s | flagged_l | flagged_r)

    # Evaluate detection on generated episodes
    res = episode_detection(ep, flagged_all)
    assert res["smurfing"]["rate"] == 1.0
    assert res["layering"]["rate"] == 1.0
    assert res["round_tripping"]["rate"] == 1.0
    assert res["overall"]["rate"] == 1.0
