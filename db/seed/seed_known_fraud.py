#!/usr/bin/env python3
"""Deterministic, idempotent database seeder and pre-T0 cutoff verifier for AegisFlow M2.

Seeds the canonical six-scenario known-fraud manifest (30 accounts, 30 synthetic
customers satisfying accounts.customer_id FK, 405 transactions, 44 derived
entity edges, and 102 ground-truth labels) into the existing PostgreSQL schema
(`m2_0001` through `m2_0003`) using `ground_truth_labels` directly.

Usage:
    PYTHONPATH=services/ingestion/src:ml/src:. python db/seed/seed_known_fraud.py
    PYTHONPATH=services/ingestion/src:ml/src:. python db/seed/seed_known_fraud.py --verify-only
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence
from urllib.parse import quote_plus
from uuid import UUID, uuid5

import psycopg
from psycopg.rows import dict_row

from aegis_ml.scenarios.constants import SCENARIO_NAMESPACE, SEED
from aegis_ml.scenarios.generator import ManifestGenerator, verify_manifest_hash
from aegis_ml.scenarios.models import (
    EvidenceProvenance,
    ScenarioDefinition,
    ScenarioManifest,
    SyntheticAccount,
    SyntheticLabel,
    SyntheticTransaction,
)

CANONICAL_MANIFEST_HASH: str = (
    "25490a2b679ae60bcab42662cda3799f23ef79a62ef964c72398200614b50b6c"
)
SEED_SOURCE_TAG: str = "aegisflow_m2_scenarios_v2"
DEFAULT_FRAUD_CONFIDENCE_THRESHOLD: float = 0.85

EXPECTED_COUNTS: dict[str, int] = {
    "scenarios": 6,
    "customers": 30,
    "accounts": 30,
    "transactions": 405,
    "edges": 44,
    "labels": 102,
}

_REQUIRED_SCHEMA_COLUMNS: dict[str, set[str]] = {
    "customers": {
        "customer_id",
        "external_ref",
        "full_name",
        "country_code",
        "kyc_status",
        "risk_rating",
        "is_pep",
        "onboarded_at",
    },
    "accounts": {
        "account_id",
        "customer_id",
        "account_number",
        "account_type",
        "currency",
        "opened_on",
        "last_active_on",
        "status",
    },
    "transactions": {
        "txn_id",
        "external_ref",
        "account_id",
        "destination_account_id",
        "counterparty_id",
        "merchant_id",
        "ip_id",
        "channel",
        "direction",
        "amount",
        "currency",
        "device_id",
        "ip_address",
        "occurred_at",
        "raw_payload",
    },
    "entity_edges": {
        "src_type",
        "src_id",
        "dst_type",
        "dst_id",
        "edge_type",
        "weight",
        "first_seen_at",
        "last_seen_at",
        "properties",
    },
    "ground_truth_labels": {
        "label_id",
        "txn_id",
        "account_id",
        "label_target",
        "is_fraud",
        "fraud_type",
        "confidence",
        "source",
        "labeled_at",
        "notes",
        "incident_start_at",
        "incident_end_at",
    },
}


def _load_dotenv() -> None:
    """Populate unset environment variables from repository-root .env if present."""
    for parent in Path(__file__).resolve().parents:
        env_path = parent / ".env"
        if env_path.is_file():
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
            return


def normalize_psycopg_url(url: str) -> str:
    """Convert SQLAlchemy psycopg URL to a libpq/psycopg3 connection URI."""
    if url.startswith("postgresql+psycopg://"):
        return "postgresql://" + url[len("postgresql+psycopg://") :]
    return url


def resolve_database_url(db_url: str | None = None) -> str:
    """Resolve PostgreSQL connection URI from explicit argument, env, or .env."""
    _load_dotenv()
    if db_url:
        return normalize_psycopg_url(db_url)

    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return normalize_psycopg_url(env_url)

    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASS")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5433")
    dbname = os.getenv("DB_NAME", "aegisflow")
    if user and password:
        return (
            f"postgresql://{quote_plus(user)}:{quote_plus(password)}"
            f"@{host}:{port}/{dbname}"
        )

    raise ValueError(
        "Database connection URL is not configured. Provide --db-url, set DATABASE_URL, "
        "or configure DB_USER/DB_PASS/DB_HOST/DB_PORT/DB_NAME."
    )


def _normalize_utc(ts: datetime | str) -> datetime:
    """Parse or coerce a timestamp to timezone-aware UTC, rejecting naive inputs."""
    if isinstance(ts, str):
        raw = ts.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
    elif isinstance(ts, datetime):
        dt = ts
    else:
        raise TypeError(f"Expected datetime or ISO-8601 string, got {type(ts)!r}")

    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"Timezone-naive timestamp is rejected: {ts!r}")
    return dt.astimezone(timezone.utc)


def customer_id_for_account(account_id: UUID) -> UUID:
    """Derive deterministic customer UUID for a synthetic manifest account."""
    return uuid5(SCENARIO_NAMESPACE, f"customer:{account_id}")


def customer_external_ref_for_account(account: SyntheticAccount) -> str:
    """Derive deterministic unique customers.external_ref (<= 64 chars)."""
    ref = f"AF-M2-SYN-CUST-{account.account_number}"
    if len(ref) > 64:
        raise ValueError(f"Derived customer external_ref exceeds 64 chars: {ref!r}")
    return ref


def load_canonical_manifest(
    manifest_json: str | Path | None = None,
    *,
    expected_hash: str | None = CANONICAL_MANIFEST_HASH,
) -> ScenarioManifest:
    """Load and cryptographically verify the canonical synthetic scenario manifest."""
    if manifest_json is not None:
        manifest_path = Path(manifest_json)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest JSON file not found: {manifest_path}")
        raw_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = ScenarioManifest.model_validate(raw_data)
    else:
        manifest = ManifestGenerator(seed=SEED).generate()

    if not verify_manifest_hash(manifest):
        raise ValueError(
            f"Manifest SHA-256 verification failed for hash={manifest.manifest_hash!r}."
        )
    if expected_hash is not None and manifest.manifest_hash != expected_hash:
        raise ValueError(
            f"Manifest SHA-256 mismatch: expected canonical {expected_hash}, "
            f"got {manifest.manifest_hash}."
        )
    return manifest


def derive_known_fraud_cutoff(
    manifest: ScenarioManifest,
    override: datetime | str | None = None,
    *,
    confidence_threshold: float = DEFAULT_FRAUD_CONFIDENCE_THRESHOLD,
) -> datetime:
    """Derive deterministic pre-T0 known-fraud cutoff timestamp (UTC).

    When not overridden, selects the UTC day boundary immediately following
    the earliest confirmed fraud wave (`mule_ring`, confirmed at
    `2025-06-13T23:59:59Z`), yielding `2025-06-14T00:00:00Z`.
    At this boundary, all 6 earliest confirmed fraud-ring accounts are eligible
    as pre-T0 graph seeds while all 41 direct positive transaction fraud cases
    and 51 subsequent fraud labels remain strictly post-T0.
    """
    if override is not None:
        return _normalize_utc(override)

    env_cut = os.getenv("KNOWN_FRAUD_CUTOFF")
    if env_cut:
        return _normalize_utc(env_cut)

    eligible_fraud = [
        lbl
        for sc in manifest.scenarios
        for lbl in sc.labels
        if lbl.is_fraud and lbl.confidence >= confidence_threshold
    ]
    if not eligible_fraud:
        raise ValueError("Cannot derive known-fraud cutoff: manifest contains no fraud labels.")

    earliest_date = min(lbl.labeled_at.astimezone(timezone.utc).date() for lbl in eligible_fraud)
    wave_max = max(
        lbl.labeled_at.astimezone(timezone.utc)
        for lbl in eligible_fraud
        if lbl.labeled_at.astimezone(timezone.utc).date() == earliest_date
    )
    return wave_max.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)


def summarize_pre_t0_eligibility(
    manifest: ScenarioManifest,
    cutoff: datetime,
    *,
    confidence_threshold: float = DEFAULT_FRAUD_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Compute pre-T0 vs post-T0 label and account counts aligned with Step 2.9 rules."""
    norm_cutoff = _normalize_utc(cutoff)
    txn_by_id: dict[UUID, SyntheticTransaction] = {
        t.txn_id: t for sc in manifest.scenarios for t in sc.transactions
    }

    pre_t0_label_ids: list[str] = []
    pre_t0_account_ids: set[str] = set()
    post_t0_fraud_labels = 0
    clean_labels = 0

    for sc in manifest.scenarios:
        for lbl in sc.labels:
            if not lbl.is_fraud:
                clean_labels += 1
                continue

            if lbl.confidence < confidence_threshold:
                continue

            labeled_utc = lbl.labeled_at.astimezone(timezone.utc)
            inc_start_ok = (
                lbl.incident_start_at is None
                or lbl.incident_start_at.astimezone(timezone.utc) < norm_cutoff
            )

            if lbl.label_target == "account" and lbl.account_id is not None:
                if labeled_utc < norm_cutoff and inc_start_ok:
                    pre_t0_label_ids.append(str(lbl.label_id))
                    pre_t0_account_ids.add(str(lbl.account_id))
                else:
                    post_t0_fraud_labels += 1
            elif lbl.label_target == "transaction" and lbl.txn_id is not None:
                target_txn = txn_by_id[lbl.txn_id]
                txn_utc = target_txn.occurred_at.astimezone(timezone.utc)
                if labeled_utc < norm_cutoff and txn_utc < norm_cutoff and inc_start_ok:
                    pre_t0_label_ids.append(str(lbl.label_id))
                    pre_t0_account_ids.add(str(target_txn.account_id))
                else:
                    post_t0_fraud_labels += 1

    return {
        "cutoff": norm_cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pre_t0_label_count": len(pre_t0_label_ids),
        "pre_t0_account_count": len(pre_t0_account_ids),
        "pre_t0_account_ids": sorted(pre_t0_account_ids),
        "post_t0_fraud_label_count": post_t0_fraud_labels,
        "clean_label_count": clean_labels,
    }


def derive_edges_for_transactions(
    transactions: Sequence[SyntheticTransaction],
) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    """Aggregate entity_edges keys, weights, and first/last timestamps from transactions."""
    edges: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}

    def _add(key: tuple[str, str, str, str, str], ts: datetime) -> None:
        if key not in edges:
            edges[key] = {"weight": 1.0, "first_seen": ts, "last_seen": ts}
        else:
            rec = edges[key]
            rec["weight"] += 1.0
            if ts < rec["first_seen"]:
                rec["first_seen"] = ts
            if ts > rec["last_seen"]:
                rec["last_seen"] = ts

    for txn in transactions:
        acc_str = str(txn.account_id)
        ts_utc = txn.occurred_at.astimezone(timezone.utc)
        if txn.device_id:
            _add(("account", acc_str, "device", str(txn.device_id), "uses_device"), ts_utc)
        if txn.ip_id is not None:
            _add(("account", acc_str, "ip", str(txn.ip_id), "uses_ip"), ts_utc)
        if txn.merchant_id is not None:
            _add(("account", acc_str, "merchant", str(txn.merchant_id), "pays_merchant"), ts_utc)
        if txn.channel == "transfer" and txn.destination_account_id is not None:
            _add(
                ("account", acc_str, "account", str(txn.destination_account_id), "transfers_to"),
                ts_utc,
            )

    return edges


def _validate_db_schema(cur: psycopg.Cursor[Any]) -> None:
    """Verify PostgreSQL has the complete m2_0001..m2_0003 schema before seeding."""
    cur.execute(
        """
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = ANY(%s);
        """,
        (list(_REQUIRED_SCHEMA_COLUMNS.keys()),),
    )
    cols_by_table: dict[str, set[str]] = {}
    for row in cur.fetchall():
        tbl = row["table_name"] if isinstance(row, dict) else row[0]
        col = row["column_name"] if isinstance(row, dict) else row[1]
        cols_by_table.setdefault(tbl, set()).add(col)

    for tbl, required in _REQUIRED_SCHEMA_COLUMNS.items():
        present = cols_by_table.get(tbl)
        if present is None:
            raise RuntimeError(
                f"Required table {tbl!r} missing in PostgreSQL; run `make migrate` first."
            )
        missing = sorted(required - present)
        if missing:
            raise RuntimeError(
                f"Table {tbl!r} is missing required columns {missing}; verify Alembic m2_0001..m2_0003."
            )


def _row_val(row: Any, key: str, idx: int) -> Any:
    return row[key] if isinstance(row, dict) else row[idx]


def verify_persisted_manifest(
    cur: psycopg.Cursor[Any],
    manifest: ScenarioManifest,
) -> dict[str, Any]:
    """Reconcile persisted PostgreSQL records against the canonical manifest and hash."""
    expected_accounts = [a for sc in manifest.scenarios for a in sc.accounts]
    expected_txns = [t for sc in manifest.scenarios for t in sc.transactions]
    expected_labels = [l for sc in manifest.scenarios for l in sc.labels]

    acc_ids = [a.account_id for a in expected_accounts]
    cust_ids = [customer_id_for_account(a.account_id) for a in expected_accounts]
    txn_ids = [t.txn_id for t in expected_txns]
    lbl_ids = [l.label_id for l in expected_labels]

    # 1. Verify customers
    cur.execute(
        """
        SELECT customer_id, external_ref
        FROM customers
        WHERE customer_id = ANY(%s::uuid[]);
        """,
        (cust_ids,),
    )
    cust_rows = {_row_val(r, "customer_id", 0): _row_val(r, "external_ref", 1) for r in cur.fetchall()}
    if len(cust_rows) != len(expected_accounts):
        raise ValueError(
            f"Customer count mismatch: expected {len(expected_accounts)}, found {len(cust_rows)}"
        )

    # 2. Verify accounts
    cur.execute(
        """
        SELECT account_id, customer_id, account_number, account_type, currency, opened_on, status
        FROM accounts
        WHERE account_id = ANY(%s::uuid[]);
        """,
        (acc_ids,),
    )
    acc_rows: dict[UUID, dict[str, Any]] = {}
    for r in cur.fetchall():
        aid = _row_val(r, "account_id", 0)
        acc_rows[aid] = {
            "account_id": aid,
            "customer_id": _row_val(r, "customer_id", 1),
            "account_number": _row_val(r, "account_number", 2),
            "account_type": _row_val(r, "account_type", 3),
            "currency": _row_val(r, "currency", 4),
            "opened_on": _row_val(r, "opened_on", 5),
            "status": _row_val(r, "status", 6),
        }
    if len(acc_rows) != len(expected_accounts):
        raise ValueError(
            f"Account count mismatch: expected {len(expected_accounts)}, found {len(acc_rows)}"
        )

    # 3. Verify transactions
    cur.execute(
        """
        SELECT txn_id, external_ref, account_id, destination_account_id, counterparty_id,
               merchant_id, ip_id, channel, direction, amount, currency,
               device_id, host(ip_address) AS ip_address, occurred_at, raw_payload
        FROM transactions
        WHERE txn_id = ANY(%s::uuid[]);
        """,
        (txn_ids,),
    )
    txn_rows: dict[UUID, dict[str, Any]] = {}
    for r in cur.fetchall():
        tid = _row_val(r, "txn_id", 0)
        raw_payload = _row_val(r, "raw_payload", 14) or {}
        if isinstance(raw_payload, str):
            raw_payload = json.loads(raw_payload)
        txn_rows[tid] = {
            "txn_id": tid,
            "external_ref": _row_val(r, "external_ref", 1),
            "account_id": _row_val(r, "account_id", 2),
            "destination_account_id": _row_val(r, "destination_account_id", 3),
            "counterparty_id": _row_val(r, "counterparty_id", 4),
            "merchant_id": _row_val(r, "merchant_id", 5),
            "ip_id": _row_val(r, "ip_id", 6),
            "channel": _row_val(r, "channel", 7),
            "direction": _row_val(r, "direction", 8),
            "amount": Decimal(str(_row_val(r, "amount", 9))),
            "currency": _row_val(r, "currency", 10),
            "device_id": _row_val(r, "device_id", 11),
            "ip_address": _row_val(r, "ip_address", 12),
            "occurred_at": _normalize_utc(_row_val(r, "occurred_at", 13)),
            "raw_payload": raw_payload,
        }
    if len(txn_rows) != len(expected_txns):
        raise ValueError(
            f"Transaction count mismatch: expected {len(expected_txns)}, found {len(txn_rows)}"
        )

    # 4. Verify ground-truth labels
    cur.execute(
        """
        SELECT label_id, txn_id, account_id, label_target, is_fraud, fraud_type,
               confidence, source, labeled_at, notes, incident_start_at, incident_end_at
        FROM ground_truth_labels
        WHERE label_id = ANY(%s::uuid[]);
        """,
        (lbl_ids,),
    )
    lbl_rows: dict[UUID, dict[str, Any]] = {}
    for r in cur.fetchall():
        lid = _row_val(r, "label_id", 0)
        inc_start = _row_val(r, "incident_start_at", 10)
        inc_end = _row_val(r, "incident_end_at", 11)
        lbl_rows[lid] = {
            "label_id": lid,
            "txn_id": _row_val(r, "txn_id", 1),
            "account_id": _row_val(r, "account_id", 2),
            "label_target": _row_val(r, "label_target", 3),
            "is_fraud": bool(_row_val(r, "is_fraud", 4)),
            "fraud_type": _row_val(r, "fraud_type", 5),
            "confidence": float(_row_val(r, "confidence", 6)),
            "source": _row_val(r, "source", 7),
            "labeled_at": _normalize_utc(_row_val(r, "labeled_at", 8)),
            "notes": _row_val(r, "notes", 9),
            "incident_start_at": _normalize_utc(inc_start) if inc_start is not None else None,
            "incident_end_at": _normalize_utc(inc_end) if inc_end is not None else None,
        }
    if len(lbl_rows) != len(expected_labels):
        raise ValueError(
            f"Label count mismatch: expected {len(expected_labels)}, found {len(lbl_rows)}"
        )

    # 5. Reconstruct scenario manifest directly from DB rows and assert field parity + SHA-256
    reconstructed_scenarios: list[ScenarioDefinition] = []
    for sc in manifest.scenarios:
        rec_accts: list[SyntheticAccount] = []
        for acc in sc.accounts:
            row = acc_rows[acc.account_id]
            expected_cid = customer_id_for_account(acc.account_id)
            if row["customer_id"] != expected_cid:
                raise ValueError(f"Account {acc.account_id} customer FK drift: {row['customer_id']} != {expected_cid}")
            if cust_rows[expected_cid] != customer_external_ref_for_account(acc):
                raise ValueError(f"Customer external_ref drift for account {acc.account_id}")
            opened_str = (
                row["opened_on"].isoformat()
                if isinstance(row["opened_on"], date)
                else str(row["opened_on"])
            )
            rec_acc = SyntheticAccount(
                account_id=row["account_id"],
                scenario_id=sc.scenario_id,
                account_number=row["account_number"],
                account_type=row["account_type"],
                currency=row["currency"],
                opened_on=opened_str,
                status=row["status"],
                is_mule=acc.is_mule,
            )
            if rec_acc != acc:
                raise ValueError(f"Persisted account mismatch for {acc.account_id}")
            rec_accts.append(rec_acc)

        rec_txns: list[SyntheticTransaction] = []
        for txn in sc.transactions:
            row = txn_rows[txn.txn_id]
            rp = row["raw_payload"]
            ev = EvidenceProvenance.model_validate(rp["evidence"]) if rp.get("evidence") else None
            rec_txn = SyntheticTransaction(
                txn_id=row["txn_id"],
                external_ref=row["external_ref"],
                account_id=row["account_id"],
                scenario_tag=rp.get("scenario_tag", txn.scenario_tag),
                channel=row["channel"],
                direction=row["direction"],
                amount=row["amount"],
                currency=row["currency"],
                occurred_at=row["occurred_at"],
                destination_account_id=row["destination_account_id"],
                merchant_id=row["merchant_id"],
                ip_id=row["ip_id"],
                device_id=row["device_id"],
                ip_address=row["ip_address"],
                evidence=ev,
            )
            if rec_txn != txn:
                raise ValueError(f"Persisted transaction mismatch for {txn.txn_id}")
            rec_txns.append(rec_txn)

        rec_lbls: list[SyntheticLabel] = []
        for lbl in sc.labels:
            row = lbl_rows[lbl.label_id]
            if not row["notes"]:
                raise ValueError(f"Persisted label {lbl.label_id} missing provenance notes JSON")
            ev = EvidenceProvenance.model_validate(json.loads(row["notes"]))
            rec_lbl = SyntheticLabel(
                label_id=row["label_id"],
                txn_id=row["txn_id"],
                account_id=row["account_id"],
                label_target=row["label_target"],
                is_fraud=row["is_fraud"],
                fraud_type=row["fraud_type"],
                confidence=row["confidence"],
                source=row["source"],
                labeled_at=row["labeled_at"],
                incident_start_at=row["incident_start_at"],
                incident_end_at=row["incident_end_at"],
                evidence=ev,
            )
            if rec_lbl != lbl:
                raise ValueError(f"Persisted label mismatch for {lbl.label_id}")
            rec_lbls.append(rec_lbl)

        reconstructed_scenarios.append(
            ScenarioDefinition(
                scenario_id=sc.scenario_id,
                typology=sc.typology,
                accounts=rec_accts,
                transactions=rec_txns,
                labels=rec_lbls,
            )
        )

    reconstructed_manifest = ScenarioManifest(
        manifest_version=manifest.manifest_version,
        seed=manifest.seed,
        observation_cutoff=manifest.observation_cutoff,
        manifest_reference_at=manifest.manifest_reference_at,
        manifest_hash=manifest.manifest_hash,
        scenarios=reconstructed_scenarios,
    )
    if not verify_manifest_hash(reconstructed_manifest):
        raise ValueError(
            "Reconstructed DB manifest failed canonical SHA-256 verification."
        )

    # 6. Verify derived entity edges and exact weights
    expected_edges = derive_edges_for_transactions(expected_txns)
    src_account_strs = sorted({str(a.account_id) for a in expected_accounts})
    cur.execute(
        """
        SELECT src_type, src_id, dst_type, dst_id, edge_type,
               weight, first_seen_at, last_seen_at
        FROM entity_edges
        WHERE src_type = 'account' AND src_id = ANY(%s::text[]);
        """,
        (src_account_strs,),
    )
    persisted_edges: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for r in cur.fetchall():
        key = (
            _row_val(r, "src_type", 0),
            _row_val(r, "src_id", 1),
            _row_val(r, "dst_type", 2),
            _row_val(r, "dst_id", 3),
            _row_val(r, "edge_type", 4),
        )
        persisted_edges[key] = {
            "weight": float(_row_val(r, "weight", 5)),
            "first_seen": _normalize_utc(_row_val(r, "first_seen_at", 6)),
            "last_seen": _normalize_utc(_row_val(r, "last_seen_at", 7)),
        }

    if set(persisted_edges.keys()) != set(expected_edges.keys()):
        missing_e = set(expected_edges.keys()) - set(persisted_edges.keys())
        extra_e = set(persisted_edges.keys()) - set(expected_edges.keys())
        raise ValueError(
            f"Entity edge set mismatch (missing={len(missing_e)}, extra={len(extra_e)})"
        )

    for key, exp in expected_edges.items():
        got = persisted_edges[key]
        if abs(got["weight"] - exp["weight"]) > 1e-9:
            raise ValueError(
                f"Entity edge weight mismatch for {key}: expected {exp['weight']}, got {got['weight']}"
            )
        if got["first_seen"] != exp["first_seen"] or got["last_seen"] != exp["last_seen"]:
            raise ValueError(
                f"Entity edge timestamp bounds mismatch for {key}: expected [{exp['first_seen']}, {exp['last_seen']}], got [{got['first_seen']}, {got['last_seen']}]"
            )

    return {
        "customers": len(cust_rows),
        "accounts": len(acc_rows),
        "transactions": len(txn_rows),
        "edges": len(persisted_edges),
        "total_edge_weight": int(sum(v["weight"] for v in persisted_edges.values())),
        "labels": len(lbl_rows),
        "manifest_hash": manifest.manifest_hash,
        "hash_verified": True,
    }


def seed_manifest(
    conn_or_dsn: psycopg.Connection[Any] | str,
    manifest: ScenarioManifest | None = None,
    *,
    cutoff: datetime | str | None = None,
    verify_only: bool = False,
) -> dict[str, Any]:
    """Seed canonical synthetic scenarios idempotently and verify DB reconciliation.

    Does not close a caller-supplied psycopg Connection.
    """
    active_manifest = manifest if manifest is not None else load_canonical_manifest()
    if not verify_manifest_hash(active_manifest):
        raise ValueError("Cannot seed unverified manifest with invalid SHA-256 hash.")

    resolved_cutoff = derive_known_fraud_cutoff(active_manifest, override=cutoff)
    pre_t0 = summarize_pre_t0_eligibility(active_manifest, resolved_cutoff)

    owns_conn = isinstance(conn_or_dsn, str)
    conn = (
        psycopg.connect(normalize_psycopg_url(conn_or_dsn), row_factory=dict_row)
        if owns_conn
        else conn_or_dsn
    )

    try:
        if verify_only:
            with conn.cursor() as cur:
                _validate_db_schema(cur)
                reconciled = verify_persisted_manifest(cur, active_manifest)
            return {
                "mode": "verify_only",
                "inserted": {
                    "customers": 0,
                    "accounts": 0,
                    "transactions": 0,
                    "edges_upserted": 0,
                    "labels": 0,
                },
                "reconciled": reconciled,
                "pre_t0": pre_t0,
            }

        with conn.transaction():
            with conn.cursor() as cur:
                _validate_db_schema(cur)

                all_accounts = [a for sc in active_manifest.scenarios for a in sc.accounts]
                all_txns_with_sc = [
                    (sc.scenario_id, t)
                    for sc in active_manifest.scenarios
                    for t in sc.transactions
                ]
                all_labels = [l for sc in active_manifest.scenarios for l in sc.labels]

                last_active_by_acc: dict[UUID, date] = {}
                for _, txn in all_txns_with_sc:
                    d = txn.occurred_at.astimezone(timezone.utc).date()
                    prev = last_active_by_acc.get(txn.account_id)
                    if prev is None or d > prev:
                        last_active_by_acc[txn.account_id] = d

                inserted_customers = 0
                for acc in all_accounts:
                    cid = customer_id_for_account(acc.account_id)
                    cref = customer_external_ref_for_account(acc)
                    risk = (
                        "low"
                        if acc.scenario_id == "legitimate_clean"
                        else ("high" if acc.is_mule else "medium")
                    )
                    full_name = (
                        f"Synthetic {acc.scenario_id.replace('_', ' ').title()} "
                        f"{acc.account_number.rsplit('-', 1)[-1]}"
                    )
                    onboarded = datetime.fromisoformat(f"{acc.opened_on}T00:00:00+00:00")
                    cur.execute(
                        """
                        INSERT INTO customers (
                            customer_id, external_ref, full_name, country_code,
                            kyc_status, risk_rating, is_pep, onboarded_at
                        ) VALUES (%s, %s, %s, 'IN', 'verified', %s, FALSE, %s)
                        ON CONFLICT (customer_id) DO NOTHING
                        RETURNING customer_id;
                        """,
                        (cid, cref, full_name, risk, onboarded),
                    )
                    if cur.fetchone() is not None:
                        inserted_customers += 1

                inserted_accounts = 0
                for acc in all_accounts:
                    cid = customer_id_for_account(acc.account_id)
                    opened_date = date.fromisoformat(acc.opened_on)
                    last_active = last_active_by_acc.get(acc.account_id)
                    cur.execute(
                        """
                        INSERT INTO accounts (
                            account_id, customer_id, account_number, account_type,
                            currency, opened_on, last_active_on, status
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (account_id) DO NOTHING
                        RETURNING account_id;
                        """,
                        (
                            acc.account_id,
                            cid,
                            acc.account_number,
                            acc.account_type,
                            acc.currency,
                            opened_date,
                            last_active,
                            acc.status,
                        ),
                    )
                    if cur.fetchone() is not None:
                        inserted_accounts += 1

                newly_inserted_txns: list[SyntheticTransaction] = []
                for scenario_id, txn in all_txns_with_sc:
                    payload = json.dumps(
                        {
                            "seed_source": SEED_SOURCE_TAG,
                            "manifest_hash": active_manifest.manifest_hash,
                            "scenario_id": scenario_id,
                            "scenario_tag": txn.scenario_tag,
                            "evidence": (
                                txn.evidence.model_dump(mode="json")
                                if txn.evidence is not None
                                else None
                            ),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    cur.execute(
                        """
                        INSERT INTO transactions (
                            txn_id, external_ref, account_id, destination_account_id,
                            counterparty_id, merchant_id, ip_id, channel, direction,
                            amount, currency, device_id, ip_address, occurred_at, raw_payload
                        ) VALUES (
                            %s, %s, %s, %s,
                            NULL, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, CAST(%s AS jsonb)
                        )
                        ON CONFLICT (txn_id) DO NOTHING
                        RETURNING txn_id;
                        """,
                        (
                            txn.txn_id,
                            txn.external_ref,
                            txn.account_id,
                            txn.destination_account_id,
                            txn.merchant_id,
                            txn.ip_id,
                            txn.channel,
                            txn.direction,
                            txn.amount,
                            txn.currency,
                            txn.device_id,
                            txn.ip_address,
                            txn.occurred_at,
                            payload,
                        ),
                    )
                    if cur.fetchone() is not None:
                        newly_inserted_txns.append(txn)

                # Decision D6 parity: upsert edges ONLY from newly inserted transactions
                edges_upserted = 0
                if newly_inserted_txns:
                    delta_edges = derive_edges_for_transactions(newly_inserted_txns)
                    edge_props = json.dumps(
                        {"seed_source": SEED_SOURCE_TAG}, sort_keys=True, separators=(",", ":")
                    )
                    for (st, si, dt, di, et), meta in sorted(delta_edges.items()):
                        cur.execute(
                            """
                            INSERT INTO entity_edges (
                                src_type, src_id, dst_type, dst_id, edge_type,
                                weight, first_seen_at, last_seen_at, properties
                            ) VALUES (
                                %s, %s, %s, %s, %s,
                                %s, %s, %s, CAST(%s AS jsonb)
                            )
                            ON CONFLICT (src_type, src_id, dst_type, dst_id, edge_type)
                            DO UPDATE SET
                                weight = entity_edges.weight + EXCLUDED.weight,
                                first_seen_at = LEAST(entity_edges.first_seen_at, EXCLUDED.first_seen_at),
                                last_seen_at = GREATEST(entity_edges.last_seen_at, EXCLUDED.last_seen_at);
                            """,
                            (
                                st,
                                si,
                                dt,
                                di,
                                et,
                                meta["weight"],
                                meta["first_seen"],
                                meta["last_seen"],
                                edge_props,
                            ),
                        )
                        edges_upserted += 1

                inserted_labels = 0
                for lbl in all_labels:
                    notes_json = json.dumps(
                        lbl.evidence.model_dump(mode="json"),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    cur.execute(
                        """
                        INSERT INTO ground_truth_labels (
                            label_id, txn_id, account_id, label_target, is_fraud,
                            fraud_type, confidence, source, labeled_at, notes,
                            incident_start_at, incident_end_at
                        ) VALUES (
                            %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s,
                            %s, %s
                        )
                        ON CONFLICT (label_id) DO NOTHING
                        RETURNING label_id;
                        """,
                        (
                            lbl.label_id,
                            lbl.txn_id,
                            lbl.account_id,
                            lbl.label_target,
                            lbl.is_fraud,
                            lbl.fraud_type,
                            lbl.confidence,
                            lbl.source,
                            lbl.labeled_at,
                            notes_json,
                            lbl.incident_start_at,
                            lbl.incident_end_at,
                        ),
                    )
                    if cur.fetchone() is not None:
                        inserted_labels += 1

                reconciled = verify_persisted_manifest(cur, active_manifest)

        return {
            "mode": "seed",
            "inserted": {
                "customers": inserted_customers,
                "accounts": inserted_accounts,
                "transactions": len(newly_inserted_txns),
                "edges_upserted": edges_upserted,
                "labels": inserted_labels,
            },
            "reconciled": reconciled,
            "pre_t0": pre_t0,
        }
    finally:
        if owns_conn:
            conn.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for canonical scenario database seeding."""
    parser = argparse.ArgumentParser(
        description="Idempotently seed canonical M2 synthetic scenarios and verify pre-T0 known fraud."
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="Optional PostgreSQL connection URL (defaults to DATABASE_URL / .env).",
    )
    parser.add_argument(
        "--manifest-json",
        type=str,
        default=None,
        help="Optional path to exported canonical ScenarioManifest JSON.",
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default=None,
        help="Optional ISO-8601 UTC override for pre-T0 known_fraud_cutoff.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify persisted DB counts and manifest hash parity without writing.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit summary output as formatted JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for seeding known-fraud scenarios into PostgreSQL."""
    args = parse_args(argv)
    try:
        manifest = load_canonical_manifest(args.manifest_json)
        dsn = resolve_database_url(args.db_url)
        summary = seed_manifest(
            dsn,
            manifest=manifest,
            cutoff=args.cutoff,
            verify_only=args.verify_only,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: database seed failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        rec = summary["reconciled"]
        ins = summary["inserted"]
        pre = summary["pre_t0"]
        print(
            f"T0 = {pre['cutoff']} -> {pre['pre_t0_account_count']} pre-T0 known-fraud accounts "
            f"({pre['pre_t0_label_count']} labels), {pre['post_t0_fraud_label_count']} post-T0 fraud labels, "
            f"{pre['clean_label_count']} clean labels"
        )
        print(
            f"Reconciled DB state: customers={rec['customers']}, accounts={rec['accounts']}, "
            f"transactions={rec['transactions']}, edges={rec['edges']} (weight={rec['total_edge_weight']}), "
            f"labels={rec['labels']}, hash={rec['manifest_hash']} [VERIFIED]"
        )
        if summary["mode"] == "seed":
            print(
                f"Inserted this run: customers={ins['customers']}, accounts={ins['accounts']}, "
                f"transactions={ins['transactions']}, edges_upserted={ins['edges_upserted']}, "
                f"labels={ins['labels']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
