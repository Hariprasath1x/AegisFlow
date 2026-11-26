#!/usr/bin/env python
"""Recover PostgreSQL-persisted API transactions missing Redis online velocity updates.

Uses per-account Redis locks (`lock:acct:{account_id}`) and re-fetches durable API rows
within the 8-day (`KEY_TTL_S`) Redis retention horizon so interrupted or failed runs can
be safely re-executed with zero duplicate velocity contributions.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import sys
from uuid import UUID

import redis
from sqlalchemy import create_engine

from aegis_ml.features import definitions as D
from ingestion import config
from ingestion.recovery import recover_pending_features


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconcile pending online feature updates from durable PostgreSQL rows."
    )
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", config.DATABASE_URL))
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", config.REDIS_URL))
    parser.add_argument("--account-id", default=None, help="Limit recovery to a single account UUID.")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=D.KEY_TTL_S // 86_400,
        help="Lookback window in days matching Redis 8-day retention (0 disables cutoff).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report pending transactions and affected accounts without mutating Redis.",
    )
    args = parser.parse_args(argv)

    if args.lookback_days < 0:
        parser.error("--lookback-days must be >= 0")

    acct_uuid = UUID(args.account_id) if args.account_id else None
    lookback_seconds = args.lookback_days * 86_400 if args.lookback_days > 0 else None

    engine = create_engine(args.database_url, pool_pre_ping=True)
    r = redis.Redis.from_url(args.redis_url, decode_responses=False)
    try:
        summary = recover_pending_features(
            engine,
            r,
            account_id=acct_uuid,
            lookback_seconds=lookback_seconds,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "note": "Recovery did not complete; re-running make recover-features is safe.",
                }
            ),
            file=sys.stderr,
        )
        return 1
    finally:
        r.close()
        engine.dispose()

    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
