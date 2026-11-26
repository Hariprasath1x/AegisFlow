"""Unit tests for Step 2.12 packaging artifacts and read-only scripts/verify_m2.py."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts import verify_m2

ROOT = Path(__file__).resolve().parents[3]


def test_required_step_2_12_files_exist_and_have_key_contracts() -> None:
    """Verify all Step 2.12 packaging files exist and avoid unsafe patterns."""
    for rel in verify_m2.REQUIRED_FILES:
        assert (ROOT / rel).is_file(), f"Missing required M2 file: {rel}"

    ing_df = (ROOT / "services/ingestion/Dockerfile").read_text(encoding="utf-8")
    assert "python:3.12-slim" in ing_df
    assert "COPY services/ingestion/src /app/services/ingestion/src" in ing_df
    assert "COPY ml/src /app/ml/src" in ing_df
    assert "EXPOSE 8001" in ing_df
    assert "/health" in ing_df

    graph_df = (ROOT / "ml/Dockerfile.graph").read_text(encoding="utf-8")
    assert "python:3.12-slim" in graph_df
    assert "ml/requirements-graph.txt" in graph_df
    assert "aegis_ml.features.graph_features" in graph_df

    graph_reqs = (ROOT / "ml/requirements-graph.txt").read_text(encoding="utf-8")
    assert "loguru==0.7.3" in graph_reqs
    assert "networkx==3.4.2" in graph_reqs
    assert "psycopg[binary]==3.2.3" in graph_reqs

    compose_m2 = (ROOT / "docker-compose.m2.yml").read_text(encoding="utf-8")
    assert "POSTGRES_USER" not in compose_m2
    assert "POSTGRES_PASSWORD" not in compose_m2
    assert "POSTGRES_DB" not in compose_m2
    assert "aegisnet" in compose_m2
    assert "postgres:5432" in compose_m2
    assert "redis://redis:6379/0" in compose_m2


def test_verify_m2_read_only_success_with_mocked_stack(monkeypatch) -> None:
    """Verify verify_m2.py passes when stack mocks report healthy M2 state without writes."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5433/aegisflow")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6380/0")

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.execute.side_effect = [
        MagicMock(fetchone=MagicMock(return_value=("m2_0003",))),
        MagicMock(fetchone=MagicMock(return_value=(405,))),
        MagicMock(fetchone=MagicMock(return_value=(0,))),
        MagicMock(fetchone=MagicMock(return_value=(True,))),
        MagicMock(fetchone=MagicMock(return_value=(6,))),
    ]

    mock_rds = MagicMock()
    mock_rds.ping.return_value = True
    mock_rds.scan_iter.side_effect = [
        iter([b"feat:acct:1"]),
        iter([b"gfeat:acct:1"]),
    ]

    with (
        patch("scripts.verify_m2.psycopg.connect", return_value=mock_conn),
        patch("scripts.verify_m2.redis.Redis.from_url", return_value=mock_rds),
        patch("scripts.verify_m2._check_http_probe", side_effect=[(True, "(200)"), (True, "(200)")]),
    ):
        all_ok, checks = verify_m2.run_verification(skip_tests=True, quiet=True)

    assert all_ok is True
    assert mock_conn.read_only is True
    assert any(c["name"].startswith("pre-T0 known fraud seeded") and c["passed"] for c in checks)


def test_verify_m2_reports_failures_honestly(monkeypatch) -> None:
    """Verify missing endpoints, empty DB, or missing Redis keys fail verify_m2 cleanly."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5433/aegisflow")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6380/0")

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.execute.side_effect = [
        MagicMock(fetchone=MagicMock(return_value=("m2_0003",))),
        MagicMock(fetchone=MagicMock(return_value=(10,))),
        MagicMock(fetchone=MagicMock(return_value=(0,))),
        MagicMock(fetchone=MagicMock(return_value=(False,))),
        MagicMock(fetchone=MagicMock(return_value=(0,))),
    ]

    mock_rds = MagicMock()
    mock_rds.ping.return_value = True
    mock_rds.scan_iter.side_effect = [iter([]), iter([])]

    with (
        patch("scripts.verify_m2.psycopg.connect", return_value=mock_conn),
        patch("scripts.verify_m2.redis.Redis.from_url", return_value=mock_rds),
        patch("scripts.verify_m2._check_http_probe", return_value=(False, "(Connection refused)")),
    ):
        all_ok, checks = verify_m2.run_verification(skip_tests=True, quiet=True)

    assert all_ok is False
    failed = {c["name"] for c in checks if not c["passed"]}
    assert "graph snapshot within the last hour (snapshot_at)" in failed
    assert "feat:acct:* keys present" in failed
    assert "gfeat:acct:* keys present" in failed
    assert "ingestion /health" in failed
    assert "ingestion /ready" in failed


def test_resolve_known_fraud_cutoff_reads_params_yaml() -> None:
    cutoff = verify_m2.resolve_known_fraud_cutoff(ROOT)
    assert cutoff == datetime(2025, 6, 14, 0, 0, 0, tzinfo=timezone.utc)
