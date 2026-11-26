#!/usr/bin/env python
"""Generate M1-compatible transactions and stream them through the M2 Ingestion API.

Preserves M1 PostgreSQL entity seeding (`scripts/seed_db.py`) while driving real
`POST /api/v1/transactions` requests in strict chronological `(ts, id)` order per
account so PostgreSQL `transactions`, `entity_edges`, and Redis online velocity
features stay synchronized.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any, Literal
import urllib.error
import urllib.request
from uuid import UUID, NAMESPACE_DNS, uuid5

import psycopg
from psycopg.rows import dict_row

M1_CHANNELS = ("card", "wire", "upi", "ach", "atm")
_CHANNEL_MAP: dict[str, Literal["card", "transfer", "wallet"]] = {
    "card": "card",
    "atm": "card",
    "wire": "transfer",
    "ach": "transfer",
    "upi": "wallet",
    "transfer": "transfer",
    "wallet": "wallet",
}
_DEMO_PREFIX = "AF-M1-DEMO-"


def _load_dotenv_defaults() -> None:
    """Load unset environment variables from repo-root .env."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
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


def normalize_psycopg_url(url: str) -> str:
    """Convert SQLAlchemy psycopg URL to a plain libpq/psycopg DSN."""
    if url.startswith("postgresql+psycopg://"):
        return "postgresql://" + url[len("postgresql+psycopg://"):]
    return url


def map_m1_channel_to_api(channel: str) -> Literal["card", "transfer", "wallet"]:
    """Map legacy M1 transaction channels onto M2 TransactionIn channels."""
    normalized = channel.strip().lower()
    if normalized not in _CHANNEL_MAP:
        raise ValueError(f"Unsupported M1 channel for M2 API ingestion: {channel!r}")
    return _CHANNEL_MAP[normalized]


def ensure_m1_entities(conn: psycopg.Connection, num_entities: int = 60) -> list[dict[str, Any]]:
    """Return deterministic M1 demo account/counterparty pairs, seeding any missing ones."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT a.account_id, a.account_number, c.counterparty_id
            FROM accounts AS a
            JOIN counterparties AS c ON c.account_id = a.account_id
            WHERE a.account_number LIKE %s
              AND c.beneficiary_ref LIKE %s
            ORDER BY a.account_number ASC
            """,
            (f"{_DEMO_PREFIX}ACCOUNT-%", f"{_DEMO_PREFIX}BENEFICIARY-%"),
        )
        existing = list(cur.fetchall())
        if len(existing) >= num_entities:
            return existing[:num_entities]

        rng = random.Random(20261010)
        now = datetime.now(timezone.utc)
        high_risk = ("AE", "VG", "PA", "CY", "MT")

        by_num = {row["account_number"]: row for row in existing}
        for i in range(num_entities):
            cust_ref = f"{_DEMO_PREFIX}CUSTOMER-{i:04d}"
            acct_num = f"{_DEMO_PREFIX}ACCOUNT-{i:04d}"
            ben_ref = f"{_DEMO_PREFIX}BENEFICIARY-{i:04d}"
            if acct_num in by_num:
                continue

            cur.execute("SELECT customer_id FROM customers WHERE external_ref = %s", (cust_ref,))
            cust_row = cur.fetchone()
            if cust_row is None:
                cur.execute(
                    """
                    INSERT INTO customers (
                        external_ref, full_name, country_code, kyc_status,
                        risk_rating, is_pep, onboarded_at
                    )
                    VALUES (%s, %s, 'IN', 'verified', %s, %s, %s)
                    RETURNING customer_id
                    """,
                    (
                        cust_ref,
                        f"Demo Customer {i:04d}",
                        rng.choice(["low", "medium", "high"]),
                        i % 17 == 0,
                        now - timedelta(days=rng.randint(200, 1500)),
                    ),
                )
                cust_id = cur.fetchone()["customer_id"]
            else:
                cust_id = cust_row["customer_id"]

            cur.execute("SELECT account_id FROM accounts WHERE account_number = %s", (acct_num,))
            acct_row = cur.fetchone()
            if acct_row is None:
                cur.execute(
                    """
                    INSERT INTO accounts (
                        customer_id, account_number, account_type, currency,
                        opened_on, last_active_on, status
                    )
                    VALUES (%s, %s, %s, 'INR', %s, %s, 'active')
                    RETURNING account_id
                    """,
                    (
                        cust_id,
                        acct_num,
                        rng.choice(["current", "savings", "card", "wallet"]),
                        (now - timedelta(days=rng.randint(200, 1500))).date(),
                        (now - timedelta(days=rng.randint(0, 30))).date(),
                    ),
                )
                acct_id = cur.fetchone()["account_id"]
            else:
                acct_id = acct_row["account_id"]

            cur.execute(
                "SELECT counterparty_id FROM counterparties WHERE beneficiary_ref = %s AND account_id = %s",
                (ben_ref, acct_id),
            )
            cp_row = cur.fetchone()
            if cp_row is None:
                country = high_risk[i % len(high_risk)] if i % 19 == 0 else "IN"
                cur.execute(
                    """
                    INSERT INTO counterparties (
                        account_id, beneficiary_ref, beneficiary_name,
                        country_code, first_seen_at, txn_count
                    )
                    VALUES (%s, %s, %s, %s, %s, 0)
                    RETURNING counterparty_id
                    """,
                    (acct_id, ben_ref, f"Demo Beneficiary {i:04d}", country, now - timedelta(days=30)),
                )
                cp_id = cur.fetchone()["counterparty_id"]
            else:
                cp_id = cp_row["counterparty_id"]

            by_num[acct_num] = {
                "account_id": acct_id,
                "account_number": acct_num,
                "counterparty_id": cp_id,
            }

        conn.commit()
        return [by_num[k] for k in sorted(by_num)][:num_entities]


def _format_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def build_m1_api_transactions(
    entities: list[dict[str, Any]],
    *,
    count: int = 120,
    seed: int = 20261010,
    end_time: datetime | None = None,
    min_start_time: datetime | None = None,
    window_minutes: int = 45,
    batch_id: str = "m1-api-stream-v1",
) -> list[dict[str, Any]]:
    """Build chronologically ordered M2 TransactionIn payloads from M1 entities."""
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    if count == 0:
        return []
    if not entities:
        raise ValueError("entities must not be empty")
    if window_minutes <= 0:
        raise ValueError(f"window_minutes must be > 0, got {window_minutes}")

    rng = random.Random(seed)
    ref_end = (end_time or datetime.now(timezone.utc)).astimezone(timezone.utc)
    start_time = ref_end - timedelta(minutes=window_minutes)
    if min_start_time is not None:
        floor_ts = min_start_time.astimezone(timezone.utc)
        if floor_ts > start_time:
            start_time = floor_ts
        if ref_end <= start_time:
            ref_end = start_time + timedelta(seconds=max(count, 60))
    step = (ref_end - start_time) / max(count, 1)

    merchants = [str(uuid5(NAMESPACE_DNS, f"aegisflow-merchant-{i}")) for i in range(12)]
    devices = [str(uuid5(NAMESPACE_DNS, f"aegisflow-device-{i}")) for i in range(24)]
    ips = [str(uuid5(NAMESPACE_DNS, f"aegisflow-ip-{i}")) for i in range(18)]

    n_entities = len(entities)
    staged: list[tuple[datetime, str, dict[str, Any]]] = []

    for i in range(count):
        acct_idx = i % n_entities
        entity = entities[acct_idx]
        m1_channel = M1_CHANNELS[i % len(M1_CHANNELS)] if i < len(M1_CHANNELS) else rng.choice(M1_CHANNELS)
        channel = map_m1_channel_to_api(m1_channel)
        if channel == "transfer" and n_entities < 2:
            channel = "card"

        amount = (
            Decimal(rng.randint(500001, 900000)) if i % 25 == 0 else Decimal(rng.randint(100, 100000))
        ).quantize(Decimal("0.01"))
        ts = start_time + step * i + timedelta(milliseconds=(i % 97) + 1)
        txn_uuid = str(uuid5(NAMESPACE_DNS, f"{batch_id}-{seed}-{i:06d}"))

        device_id = devices[(acct_idx + (i // n_entities)) % len(devices)]
        ip_id = ips[(acct_idx // 2 + i) % len(ips)]

        if channel == "transfer":
            dest_idx = (acct_idx + 1 + (i % max(1, n_entities - 1))) % n_entities
            if dest_idx == acct_idx:
                dest_idx = (acct_idx + 1) % n_entities
            counterparty_id: str | None = str(entities[dest_idx]["account_id"])
            merchant_id: str | None = None
        elif channel == "card":
            counterparty_id = str(entity["counterparty_id"])
            merchant_id = merchants[(acct_idx + i) % len(merchants)]
        else:  # wallet
            counterparty_id = str(entity["counterparty_id"])
            merchant_id = None

        payload: dict[str, Any] = {
            "id": txn_uuid,
            "ts": _format_utc(ts),
            "account_id": str(entity["account_id"]),
            "counterparty_id": counterparty_id,
            "merchant_id": merchant_id,
            "device_id": device_id,
            "ip_id": ip_id,
            "amount": f"{amount:.2f}",
            "currency": "INR",
            "channel": channel,
        }
        staged.append((ts, txn_uuid, payload))

    staged.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in staged]


@dataclass
class IngestionSummary:
    generated: int = 0
    submitted: int = 0
    accepted_new: int = 0
    accepted_duplicate: int = 0
    rejected: int = 0
    distinct_accounts: int = 0
    channels: dict[str, int] = field(default_factory=dict)
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    errors: list[dict[str, Any]] = field(default_factory=list)


def _http_get_json(url: str, timeout: float) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return resp.status, body
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except Exception:
            body = {"raw": raw}
        return exc.code, body


def submit_transactions(
    payloads: list[dict[str, Any]],
    *,
    api_url: str,
    service_key: str,
    timeout: float = 10.0,
    check_ready: bool = True,
) -> IngestionSummary:
    """Submit ordered payloads sequentially to POST /api/v1/transactions."""
    base_url = api_url.rstrip("/")
    if check_ready:
        for path in ("/health", "/ready"):
            status_code, body = _http_get_json(f"{base_url}{path}", timeout=timeout)
            if status_code != 200:
                raise RuntimeError(f"Ingestion API {path} returned HTTP {status_code}: {body}")

    summary = IngestionSummary(generated=len(payloads))
    latencies_ms: list[float] = []
    accounts: set[str] = set()
    channel_counts: dict[str, int] = {}

    for payload in payloads:
        channel = str(payload["channel"])
        channel_counts[channel] = channel_counts.get(channel, 0) + 1
        accounts.add(str(payload["account_id"]))

        req = urllib.request.Request(
            f"{base_url}/api/v1/transactions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Service-Key": service_key,
            },
            method="POST",
        )
        t0 = time.perf_counter()
        summary.submitted += 1
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(elapsed_ms)
                data = json.loads(resp.read().decode("utf-8"))
                if resp.status == 201 and not data.get("duplicate", False):
                    summary.accepted_new += 1
                elif resp.status == 200 or data.get("duplicate", False):
                    summary.accepted_duplicate += 1
                else:
                    summary.rejected += 1
                    if len(summary.errors) < 10:
                        summary.errors.append({"id": payload["id"], "status": resp.status, "body": data})
        except urllib.error.HTTPError as exc:
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            summary.rejected += 1
            if len(summary.errors) < 10:
                summary.errors.append(
                    {
                        "id": payload["id"],
                        "status": exc.code,
                        "detail": exc.read().decode("utf-8", errors="replace"),
                    }
                )
        except Exception as exc:
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            summary.rejected += 1
            if len(summary.errors) < 10:
                summary.errors.append({"id": payload["id"], "error": str(exc)})

    summary.distinct_accounts = len(accounts)
    summary.channels = channel_counts
    if latencies_ms:
        summary.mean_latency_ms = round(statistics.mean(latencies_ms), 2)
        sorted_lat = sorted(latencies_ms)
        p95_idx = min(len(sorted_lat) - 1, max(0, int(0.95 * len(sorted_lat)) - 1))
        summary.p95_latency_ms = round(sorted_lat[p95_idx], 2)
    return summary


def _latest_api_txn_time(conn: psycopg.Connection, account_ids: list[UUID]) -> datetime | None:
    """Return latest occurred_at among non-M1-SQL transactions for these accounts."""
    if not account_ids:
        return None
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT MAX(occurred_at)
            FROM transactions
            WHERE account_id = ANY(%s)
              AND (external_ref IS NULL OR external_ref NOT LIKE %s)
            """,
            (account_ids, f"{_DEMO_PREFIX}TXN-%"),
        )
        row = cur.fetchone()
        return row[0] if row and row[0] is not None else None


def main(argv: list[str] | None = None) -> int:
    _load_dotenv_defaults()
    parser = argparse.ArgumentParser(description="Stream M1 demo transactions to the M2 Ingestion API.")
    parser.add_argument("--count", type=int, default=120, help="Number of transactions to generate.")
    parser.add_argument("--seed", type=int, default=20261010, help="Deterministic RNG seed.")
    parser.add_argument("--window-minutes", type=int, default=45, help="Recent UTC event window width.")
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Optional deterministic UUID namespace tag (defaults to a fresh batch per invocation).",
    )
    parser.add_argument(
        "--end-time",
        default=None,
        help="Optional ISO-8601 UTC end timestamp for reproducible replay.",
    )
    parser.add_argument("--api-url", default=os.getenv("INGESTION_API_URL", "http://localhost:8001"))
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", "postgresql://aegis:aegis_dev_2026@localhost:5433/aegisflow"),
    )
    parser.add_argument("--service-key", default=os.getenv("SERVICE_KEY", "dev-service-key"))
    parser.add_argument("--dry-run", action="store_true", help="Build payloads without sending HTTP requests.")
    args = parser.parse_args(argv)

    end_time = datetime.fromisoformat(args.end_time.replace("Z", "+00:00")) if args.end_time else None
    batch_id = args.batch_id or f"m1-api-stream-{int(time.time() * 1000)}"

    dsn = normalize_psycopg_url(args.database_url)
    with psycopg.connect(dsn) as conn:
        entities = ensure_m1_entities(conn, num_entities=60)
        latest_ts = None if args.end_time else _latest_api_txn_time(
            conn, [row["account_id"] for row in entities]
        )

    min_start = (latest_ts + timedelta(milliseconds=10)) if latest_ts is not None else None
    payloads = build_m1_api_transactions(
        entities,
        count=args.count,
        seed=args.seed,
        end_time=end_time,
        min_start_time=min_start,
        window_minutes=args.window_minutes,
        batch_id=batch_id,
    )
    if args.dry_run:
        print(json.dumps({"generated": len(payloads), "sample": payloads[:2]}, indent=2))
        return 0

    summary = submit_transactions(
        payloads,
        api_url=args.api_url,
        service_key=args.service_key,
    )
    print(json.dumps(asdict(summary), indent=2))
    return 0 if summary.rejected == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
