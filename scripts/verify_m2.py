#!/usr/bin/env python3
"""Read-only M2 Definition-of-Done verification script for AegisFlow.

Verifies repository files, M2 unit/API/integration suites (unless --skip-tests),
PostgreSQL schema & data invariants (in a read-only transaction), Redis keyspace
presence (scan-only, zero writes), and ingestion HTTP /health and /ready probes.

Usage:
    PYTHONPATH=services/ingestion/src:ml/src:. python scripts/verify_m2.py
    PYTHONPATH=services/ingestion/src:ml/src:. python scripts/verify_m2.py --skip-tests
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence
from urllib.parse import quote_plus
import urllib.request

import psycopg
import redis
import yaml

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES: tuple[str, ...] = (
    "ml/src/aegis_ml/features/definitions.py",
    "ml/src/aegis_ml/features/offline.py",
    "ml/src/aegis_ml/features/graph_features.py",
    "ml/src/aegis_ml/features/dataset_builder.py",
    "services/ingestion/src/ingestion/config.py",
    "services/ingestion/src/ingestion/schemas.py",
    "services/ingestion/src/ingestion/persistence.py",
    "services/ingestion/src/ingestion/online_features.py",
    "services/ingestion/src/ingestion/locks.py",
    "services/ingestion/src/ingestion/recovery.py",
    "services/ingestion/src/ingestion/deps.py",
    "services/ingestion/src/ingestion/main.py",
    "services/ingestion/src/ingestion/api/routes.py",
    "services/ingestion/Dockerfile",
    "services/ingestion/requirements.txt",
    "ml/Dockerfile.graph",
    "ml/requirements-graph.txt",
    "db/migrations/versions/m2_0001_feature_store.py",
    "db/migrations/versions/m2_0002_transaction_schema_alignment.py",
    "db/migrations/versions/m2_0003_ground_truth_incidents.py",
    "db/seed/seed_known_fraud.py",
    "docker-compose.m2.yml",
    "docs/adr/0004-feature-store.md",
    "dvc.yaml",
    "params.yaml",
    "tests/m2/unit/test_m2_parity.py",
    "tests/m2/unit/test_m2_offline.py",
    "tests/m2/unit/test_graph_features.py",
    "tests/m2/unit/test_ingestion_api.py",
    "tests/m2/unit/test_step_2_10_pit_and_seed.py",
    "tests/m2/api/test_m1_to_api_ingestion.py",
    "tests/m2/api/test_pending_feature_recovery.py",
    "tests/m2/integration/test_m2_migration.py",
    "tests/m2/integration/test_m2_0002_migration.py",
    "tests/m2/integration/test_m2_0003_migration.py",
    "tests/m2/integration/test_graph_features_job.py",
    "tests/m2/integration/test_step_2_10_seed_and_pit_db.py",
)

DEFAULT_KNOWN_FRAUD_CUTOFF = "2025-06-14T00:00:00Z"


def _load_dotenv(repo_root: Path = ROOT) -> None:
    """Populate unset environment variables from repo-root .env without overriding env."""
    env_path = repo_root / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def resolve_database_dsn() -> str | None:
    """Resolve libpq PostgreSQL DSN from environment / .env without hardcoded secrets."""
    _load_dotenv()
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url.replace("postgresql+psycopg://", "postgresql://", 1)

    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASS")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5433")
    dbname = os.getenv("DB_NAME", "aegisflow")
    if user and password:
        return f"postgresql://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{dbname}"
    return None


def resolve_redis_url() -> str:
    """Resolve Redis URL from environment / .env (defaults to Docker host port 6380)."""
    _load_dotenv()
    return os.getenv("REDIS_URL", "redis://localhost:6380/0")


def resolve_known_fraud_cutoff(repo_root: Path = ROOT) -> datetime:
    """Resolve pre-T0 known-fraud cutoff from params.yaml, env, or canonical default."""
    raw = os.getenv("KNOWN_FRAUD_CUTOFF")
    if not raw:
        params_file = repo_root / "params.yaml"
        if params_file.is_file():
            cfg = yaml.safe_load(params_file.read_text(encoding="utf-8")) or {}
            raw = (cfg.get("features") or {}).get("known_fraud_cutoff")
    if not raw:
        raw = DEFAULT_KNOWN_FRAUD_CUTOFF
    raw_str = str(raw).strip()
    if raw_str.endswith("Z"):
        raw_str = raw_str[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw_str)
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"Timezone-naive known_fraud_cutoff rejected: {raw!r}")
    return dt.astimezone(timezone.utc)


def _check_http_probe(url: str, timeout_s: float = 5.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            status = getattr(resp, "status", 200)
            return status == 200, f"({status})"
    except Exception as exc:
        return False, f"({str(exc)[:80]})"


def run_verification(
    *,
    repo_root: Path = ROOT,
    skip_tests: bool = False,
    ingest_url: str | None = None,
    quiet: bool = False,
) -> tuple[bool, list[dict[str, Any]]]:
    """Execute read-only M2 checks and return `(all_passed, check_records)`."""
    _load_dotenv(repo_root)
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        ok = bool(passed)
        checks.append({"name": name, "passed": ok, "detail": detail})
        if not quiet:
            tag = "OK" if ok else "XX"
            suffix = f" {detail}" if detail else ""
            print(f"[{tag}] {name}{suffix}")

    # 1. Static file presence
    for rel_path in REQUIRED_FILES:
        record(f"file {rel_path}", (repo_root / rel_path).is_file())

    # 2. M2 unit, API, and integration test suites
    dsn = resolve_database_dsn()
    if not skip_tests:
        env = {**os.environ, "PYTHONPATH": "services/ingestion/src:ml/src:."}
        unit_api = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests/m2/unit", "tests/m2/api"],
            cwd=repo_root,
            env=env,
        )
        record(
            "unit + API tests (incl. exact parity over 5,000 transactions)",
            unit_api.returncode == 0,
        )
        if not dsn:
            record("DATABASE_URL configured for integration tests", False, "(missing in env/.env)")
        else:
            integ = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "tests/m2/integration"],
                cwd=repo_root,
                env=env,
            )
            record("integration tests (migrations, edges, FK, graph job, Step 2.10)", integ.returncode == 0)

    # 3. Read-only PostgreSQL checks
    if not dsn:
        record("postgres connection", False, "(DATABASE_URL / DB_* not configured)")
    else:
        try:
            with psycopg.connect(dsn, connect_timeout=5) as conn:
                conn.read_only = True
                alembic_rev = conn.execute("SELECT version_num FROM alembic_version").fetchone()
                rev_val = alembic_rev[0] if alembic_rev else None
                record("postgres alembic head (m2_0003)", rev_val == "m2_0003", f"({rev_val})")

                n_tx = conn.execute("SELECT count(*) FROM transactions").fetchone()[0]
                record("transactions stored", n_tx > 0, f"({n_tx})")

                dup_edges = conn.execute(
                    """
                    SELECT count(*) FROM (
                        SELECT 1 FROM entity_edges
                        GROUP BY src_type, src_id, dst_type, dst_id, edge_type
                        HAVING count(*) > 1
                    ) d
                    """
                ).fetchone()[0]
                record("no duplicate entity_edges", dup_edges == 0, f"({dup_edges} duplicates)")

                recent_snap = conn.execute(
                    "SELECT max(snapshot_at) > now() - interval '65 minutes' FROM graph_feature_snapshots"
                ).fetchone()[0]
                record("graph snapshot within the last hour (snapshot_at)", bool(recent_snap))

                cutoff = resolve_known_fraud_cutoff(repo_root)
                kf_count = conn.execute(
                    """
                    SELECT count(*) FROM ground_truth_labels
                    WHERE is_fraud IS TRUE
                      AND confidence >= 0.85
                      AND labeled_at < %s
                    """,
                    (cutoff,),
                ).fetchone()[0]
                record(
                    f"pre-T0 known fraud seeded (ground_truth_labels < {cutoff.strftime('%Y-%m-%dT%H:%M:%SZ')})",
                    kf_count > 0,
                    f"({kf_count})",
                )
        except Exception as exc:
            record("postgres read-only verification", False, f"({str(exc)[:100]})")

    # 4. Read-only Redis checks
    redis_url = resolve_redis_url()
    try:
        rds = redis.Redis.from_url(redis_url, socket_timeout=5)
        rds.ping()
        record("redis connectivity", True)
        has_feat = next(rds.scan_iter("feat:acct:*", count=1000), None) is not None
        has_gfeat = next(rds.scan_iter("gfeat:acct:*", count=1000), None) is not None
        record("feat:acct:* keys present", has_feat)
        record("gfeat:acct:* keys present", has_gfeat)
    except Exception as exc:
        record("redis read-only verification", False, f"({str(exc)[:100]})")

    # 5. Ingestion HTTP probes
    base_url = (ingest_url or os.getenv("INGEST_URL", "http://localhost:8001")).rstrip("/")
    for path in ("/health", "/ready"):
        ok_http, detail = _check_http_probe(f"{base_url}{path}")
        record(f"ingestion {path}", ok_http, detail)

    all_passed = all(c["passed"] for c in checks)
    if not quiet:
        passed_n = sum(1 for c in checks if c["passed"])
        print(f"\n{passed_n}/{len(checks)} checks passed")
    return all_passed, checks


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only M2 Definition-of-Done verification.")
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running pytest suites and verify static files + live stack only.",
    )
    parser.add_argument(
        "--ingest-url",
        type=str,
        default=None,
        help="Override ingestion base URL (default: INGEST_URL or http://localhost:8001).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print verification summary as JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    all_passed, checks = run_verification(
        skip_tests=args.skip_tests,
        ingest_url=args.ingest_url,
        quiet=args.json,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "passed": sum(1 for c in checks if c["passed"]),
                    "total": len(checks),
                    "all_passed": all_passed,
                    "checks": checks,
                },
                indent=2,
            )
        )
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
