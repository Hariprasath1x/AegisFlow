"""Unit tests for M2 Step 2.10: canonical scenario seeding, pre-T0 cutoff, and PIT Parquet CLI."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import ipaddress
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import sqlalchemy as sa
import yaml

from aegis_ml.features import definitions as D
from aegis_ml.features.dataset_builder import build_supervised_training_dataset
from aegis_ml.features.offline import (
    FORBIDDEN_PIT_TARGET_COLUMNS,
    PIT_TRACE_COLUMNS,
    build_pit_feature_table,
    export_features_pit,
    main as offline_main,
    prepare,
    validate_pit_output,
    write_pit_parquet,
)
from aegis_ml.scenarios.constants import OBSERVATION_CUTOFF, SEED
from aegis_ml.scenarios.generator import ManifestGenerator
from db.seed.seed_known_fraud import (
    CANONICAL_MANIFEST_HASH,
    EXPECTED_COUNTS,
    customer_external_ref_for_account,
    customer_id_for_account,
    derive_edges_for_transactions,
    derive_known_fraud_cutoff,
    load_canonical_manifest,
    main as seed_main,
    summarize_pre_t0_eligibility,
)

ROOT = Path(__file__).resolve().parents[3]


def test_canonical_manifest_hash_and_customer_derivation() -> None:
    """Verify canonical manifest hash, exact entity counts, and deterministic customer IDs."""
    manifest = load_canonical_manifest()
    assert manifest.manifest_hash == CANONICAL_MANIFEST_HASH

    accounts = [a for sc in manifest.scenarios for a in sc.accounts]
    txns = [t for sc in manifest.scenarios for t in sc.transactions]
    labels = [l for sc in manifest.scenarios for l in sc.labels]

    assert len(manifest.scenarios) == EXPECTED_COUNTS["scenarios"]
    assert len(accounts) == EXPECTED_COUNTS["accounts"]
    assert len(txns) == EXPECTED_COUNTS["transactions"]
    assert len(labels) == EXPECTED_COUNTS["labels"]

    cust_ids = {customer_id_for_account(a.account_id) for a in accounts}
    cust_refs = {customer_external_ref_for_account(a) for a in accounts}
    assert len(cust_ids) == EXPECTED_COUNTS["accounts"]
    assert len(cust_refs) == EXPECTED_COUNTS["accounts"]
    assert all(len(ref) <= 64 and not ref.startswith("AF-M1-DEMO-") for ref in cust_refs)

    edges = derive_edges_for_transactions(txns)
    assert len(edges) == EXPECTED_COUNTS["edges"]
    assert int(sum(e["weight"] for e in edges.values())) == 455


def test_canonical_manifest_loader_fails_closed_on_tampering(tmp_path: Path) -> None:
    """Verify manifest loader rejects missing files or modified content/hash."""
    with pytest.raises(FileNotFoundError):
        load_canonical_manifest(tmp_path / "missing.json")

    manifest = ManifestGenerator(seed=SEED).generate()
    payload = manifest.model_dump(mode="json")
    payload["manifest_hash"] = "0" * 64
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 verification failed"):
        load_canonical_manifest(tampered_path)


def test_deterministic_pre_t0_cutoff_and_params_yaml_alignment() -> None:
    """Verify derived pre-T0 cutoff excludes future fraud labels and matches params.yaml."""
    manifest = load_canonical_manifest()
    cutoff = derive_known_fraud_cutoff(manifest)
    assert cutoff == datetime(2025, 6, 14, 0, 0, 0, tzinfo=timezone.utc)

    params_cfg = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    assert "features" in params_cfg
    assert params_cfg["features"]["known_fraud_cutoff"] == "2025-06-14T00:00:00Z"

    summary = summarize_pre_t0_eligibility(manifest, cutoff)
    assert summary["pre_t0_account_count"] == 6
    assert summary["pre_t0_label_count"] == 6
    assert summary["post_t0_fraud_label_count"] == 51
    assert summary["clean_label_count"] == 45

    with pytest.raises(ValueError, match="Timezone-naive"):
        derive_known_fraud_cutoff(manifest, override="2025-06-14T00:00:00")


def test_seeder_cli_argument_failures(tmp_path: Path, capsys) -> None:
    """Verify seeder CLI fails closed with exit code 2 on bad inputs."""
    rc_missing = seed_main(["--manifest-json", str(tmp_path / "missing.json")])
    assert rc_missing == 2
    assert "not found" in capsys.readouterr().err

    rc_naive = seed_main(["--cutoff", "2025-06-14T00:00:00", "--db-url", "postgresql://u:p@localhost/db"])
    assert rc_naive == 2
    assert "Timezone-naive" in capsys.readouterr().err


def test_prepare_coalesces_ip_id_ip_address_and_nullable_entities() -> None:
    """Verify IP feature parity across ip_id, INET ip_address, and nullable IP values."""
    t0 = datetime(2026, 2, 1, 10, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=10)
    t2 = t0 + timedelta(minutes=20)
    shared_ip_uuid = uuid4()

    df_via_ip_id = pd.DataFrame(
        [
            {"txn_id": "t1", "account_id": "A1", "ts": t0, "amount": Decimal("10.00"), "ip_id": shared_ip_uuid, "ip_address": None},
            {"txn_id": "t2", "account_id": "A2", "ts": t1, "amount": Decimal("20.00"), "ip_id": str(shared_ip_uuid), "ip_address": ""},
            {"txn_id": "t3", "account_id": "A2", "ts": t2, "amount": Decimal("30.00"), "ip_id": None, "ip_address": None},
        ]
    )
    df_via_ip_addr = pd.DataFrame(
        [
            {"txn_id": "t1", "account_id": "A1", "ts": t0, "amount": Decimal("10.00"), "ip_address": ipaddress.ip_interface(f"198.51.100.7/32")},
            {"txn_id": "t2", "account_id": "A2", "ts": t1, "amount": Decimal("20.00"), "ip_address": "198.51.100.7"},
            {"txn_id": "t3", "account_id": "A2", "ts": t2, "amount": Decimal("30.00"), "ip_address": None},
        ]
    )

    prep_addr = prepare(df_via_ip_addr)
    assert prep_addr["ip_address"].tolist() == ["198.51.100.7", "198.51.100.7", None]

    pit_from_id = build_pit_feature_table(df_via_ip_id)
    pit_from_addr = build_pit_feature_table(df_via_ip_addr)

    assert pit_from_id["distinct_ips_24h"].tolist() == [1.0, 1.0, 1.0]
    assert pit_from_id["ip_accounts_7d"].tolist() == [1.0, 2.0, 0.0]
    pd.testing.assert_frame_equal(
        pit_from_id[D.FEATURE_LIST],
        pit_from_addr[D.FEATURE_LIST],
        check_exact=True,
    )


def test_pit_table_temporal_boundaries_and_snapshot_traceability() -> None:
    """Verify strict as_of filtering and backward-only graph snapshot timestamp traceability."""
    t0 = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    t_future = datetime(2026, 3, 1, 15, 0, tzinfo=timezone.utc)

    txns = pd.DataFrame(
        [
            {
                "txn_id": "txn-0",
                "account_id": "acc-1",
                "ts": t0,
                "amount": Decimal("50.00"),
                "destination_account_id": None,
                "merchant_id": "mer-1",
                "device_id": "dev-1",
                "ip_id": None,
            },
            {
                "txn_id": "txn-1",
                "account_id": "acc-1",
                "ts": t1,
                "amount": Decimal("100.00"),
                "destination_account_id": "acc-2",
                "merchant_id": None,
                "device_id": "dev-1",
                "ip_id": "ip-1",
            },
            {
                "txn_id": "txn-future",
                "account_id": "acc-1",
                "ts": t_future,
                "amount": Decimal("999.00"),
                "destination_account_id": None,
                "merchant_id": None,
                "device_id": "dev-2",
                "ip_id": None,
            },
        ]
    )

    s_at_11 = datetime(2026, 3, 1, 11, 0, tzinfo=timezone.utc)
    s_at_14 = datetime(2026, 3, 1, 14, 0, tzinfo=timezone.utc)
    snaps = pd.DataFrame(
        [
            {"account_id": "acc-1", "snapshot_at": s_at_11, "component_size": 4.0, "dist_to_fraud": 2.0, "risk_rank": 0.25},
            {"account_id": "acc-1", "snapshot_at": s_at_14, "component_size": 99.0, "dist_to_fraud": 0.0, "risk_rank": 0.99},
        ]
    )

    as_of = datetime(2026, 3, 1, 13, 0, tzinfo=timezone.utc)
    pit = build_pit_feature_table(txns, snaps, as_of=as_of)

    assert pit["txn_id"].tolist() == ["txn-0", "txn-1"]
    assert list(pit.columns) == list(PIT_TRACE_COLUMNS) + list(D.FEATURE_LIST)
    assert set(FORBIDDEN_PIT_TARGET_COLUMNS).isdisjoint(pit.columns)

    # Row 0 (10:00) is before S1 (11:00) -> NaT snapshot timestamp & defaults
    assert pd.isna(pit.loc[0, "feature_snapshot_at"])
    assert pit.loc[0, "graph_component_size"] == 1.0
    assert pit.loc[0, "graph_dist_to_fraud"] == -1.0

    # Row 1 (12:00) gets S1 (11:00), never S2 (14:00)
    assert pit.loc[1, "feature_snapshot_at"] == pd.Timestamp(s_at_11)
    assert pit.loc[1, "graph_component_size"] == 4.0
    assert pit.loc[1, "graph_dist_to_fraud"] == 2.0
    assert pit.loc[1, "graph_risk_rank"] == 0.25
    assert pit.loc[1, "computed_as_of"] == pd.Timestamp(as_of)


def test_pit_parquet_determinism_empty_schema_and_atomic_write(tmp_path: Path, monkeypatch) -> None:
    """Verify deterministic Parquet write/reload, typed empty output, and atomic cleanup."""
    empty_pit = build_pit_feature_table(pd.DataFrame(), None)
    empty_file = tmp_path / "empty.parquet"
    write_pit_parquet(empty_pit, empty_file)
    reloaded_empty = pd.read_parquet(empty_file)
    assert len(reloaded_empty) == 0
    assert list(reloaded_empty.columns) == list(PIT_TRACE_COLUMNS) + list(D.FEATURE_LIST)

    t0 = datetime(2026, 2, 10, 8, 30, tzinfo=timezone.utc)
    txns = pd.DataFrame(
        [
            {"txn_id": "t-b", "account_id": "a-1", "ts": t0 + timedelta(seconds=30), "amount": Decimal("40.00")},
            {"txn_id": "t-a", "account_id": "a-1", "ts": t0, "amount": Decimal("25.00")},
        ]
    )
    pit1 = build_pit_feature_table(txns)
    pit2 = build_pit_feature_table(txns.iloc[::-1].reset_index(drop=True))
    p1 = tmp_path / "pit1.parquet"
    p2 = tmp_path / "pit2.parquet"
    write_pit_parquet(pit1, p1)
    write_pit_parquet(pit2, p2)

    df1 = pd.read_parquet(p1)
    df2 = pd.read_parquet(p2)
    pd.testing.assert_frame_equal(df1, df2, check_exact=True)
    assert pq.read_table(p1).schema.field("device_id").type == "string"

    # Atomic failure protection preserves prior output and cleans up temp file
    original_bytes = p1.read_bytes()

    def _fail_write(*args, **kwargs):
        raise OSError("simulated disk full")

    monkeypatch.setattr("aegis_ml.features.offline.pq.write_table", _fail_write)
    with pytest.raises(OSError, match="simulated disk full"):
        write_pit_parquet(pit1, p1)
    assert p1.read_bytes() == original_bytes
    assert not list(tmp_path.glob(".features_pit_tmp_*"))


def test_validate_pit_output_rejects_target_leakage_and_future_snapshots() -> None:
    """Verify PIT validator fails closed on target leakage or temporal violations."""
    t0 = datetime(2026, 2, 10, 8, 30, tzinfo=timezone.utc)
    pit = build_pit_feature_table(
        pd.DataFrame([{"txn_id": "t-1", "account_id": "a-1", "ts": t0, "amount": Decimal("10.00")}])
    )

    leaked_target = pit.copy()
    leaked_target["is_fraud"] = 1
    with pytest.raises(ValueError, match="supervised target columns"):
        validate_pit_output(leaked_target)

    leaked_snap = pit.copy()
    leaked_snap["feature_snapshot_at"] = pd.Timestamp(t0 + timedelta(minutes=1))
    with pytest.raises(ValueError, match="Temporal leakage"):
        validate_pit_output(leaked_snap)


def test_offline_cli_sqlite_export_preserves_caller_engine_and_dataset_builder(tmp_path: Path, capsys) -> None:
    """Verify DB export preserves caller Engine, CLI returns valid exit codes, and dataset_builder stays intact."""
    db_path = tmp_path / "local_txns.db"
    eng = sa.create_engine(f"sqlite:///{db_path}")
    with eng.begin() as conn:
        conn.execute(
            sa.text(
                """
                CREATE TABLE transactions (
                    txn_id TEXT PRIMARY KEY,
                    occurred_at TIMESTAMP NOT NULL,
                    account_id TEXT NOT NULL,
                    destination_account_id TEXT,
                    merchant_id TEXT,
                    device_id TEXT,
                    ip_id TEXT,
                    ip_address TEXT,
                    amount NUMERIC NOT NULL
                );
                """
            )
        )
        conn.execute(
            sa.text(
                """
                CREATE TABLE graph_feature_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    snapshot_at TIMESTAMP NOT NULL,
                    features TEXT NOT NULL
                );
                """
            )
        )
        conn.execute(
            sa.text(
                """
                INSERT INTO transactions VALUES
                    ('txn-1', '2026-01-10T10:00:00+00:00', 'acc-1', NULL, NULL, 'dev-1', 'ip-1', NULL, 100.00),
                    ('txn-2', '2026-01-10T10:05:00+00:00', 'acc-1', NULL, NULL, 'dev-1', NULL, '198.51.100.5', 200.00);
                """
            )
        )
        conn.execute(
            sa.text(
                """
                INSERT INTO graph_feature_snapshots VALUES
                    ('s-1', 'acc-1', '2026-01-10T09:00:00+00:00', '{"component_size": 5, "dist_to_fraud": 2, "risk_rank": 0.4}');
                """
            )
        )

    out_path = tmp_path / "features_pit.parquet"
    summary = export_features_pit(out_path, engine=eng)
    assert summary["rows"] == 2
    assert summary["num_features"] == 19
    # Caller-owned engine must remain usable
    with eng.connect() as conn:
        assert conn.execute(sa.text("SELECT COUNT(*) FROM transactions")).scalar() == 2
    eng.dispose()

    cli_out = tmp_path / "cli_pit.parquet"
    rc = offline_main(["--out", str(cli_out), "--db-url", f"sqlite:///{db_path}"])
    assert rc == 0
    assert "2 rows x 19 features" in capsys.readouterr().out
    assert len(pd.read_parquet(cli_out)) == 2

    rc_bad = offline_main(["--out", str(cli_out), "--db-url", f"sqlite:///{db_path}", "--as-of", "2026-01-10T10:00:00"])
    assert rc_bad == 2
    assert "Timezone-naive" in capsys.readouterr().err

    # Confirm dataset_builder three-state semantics remain unchanged
    manifest = load_canonical_manifest()
    pit_df, sup_df, out_df = build_supervised_training_dataset(
        transactions=pd.DataFrame([t.model_dump() for sc in manifest.scenarios for t in sc.transactions]),
        labels=pd.DataFrame([l.model_dump() for sc in manifest.scenarios for l in sc.labels]),
        accounts=pd.DataFrame([a.model_dump() for sc in manifest.scenarios for a in sc.accounts]),
        evaluation_time=OBSERVATION_CUTOFF,
    )
    assert len(pit_df) == 405
    assert len(sup_df) == 174
    assert len(out_df) == 405
