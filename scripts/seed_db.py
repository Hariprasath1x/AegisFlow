#!/usr/bin/env python
"""Seed AegisFlow reference data and a demo transaction set."""

from __future__ import annotations
import os
import random
import uuid
import psycopg
from datetime import datetime, timedelta, timezone
from psycopg.rows import dict_row

DSN = (
    f"host={os.getenv('DB_HOST', 'localhost')} port={os.getenv('DB_PORT', '5432')} "
    f"dbname={os.getenv('DB_NAME', 'aegisflow')} user={os.getenv('DB_USER', 'aegis')} "
    f"password={os.getenv('DB_PASS', 'aegis_dev_2026')}"
)

RULES = [
    (
        "R001_HIGH_VALUE",
        1,
        "Single transaction above reporting threshold",
        25,
        '{"threshold_amount": 500000}',
    ),
    (
        "R002_VELOCITY_1H",
        1,
        "More than N transactions in a rolling hour",
        20,
        '{"max_txn_per_hour": 12}',
    ),
    (
        "R003_STRUCTURING",
        1,
        "Multiple sub-threshold debits summing above the report limit",
        35,
        '{"window_hours": 24, "report_limit": 1000000}',
    ),
    (
        "R004_NEW_BENEFICIARY",
        1,
        "High-value transfer to a beneficiary first seen recently",
        15,
        '{"account_age_days": 7}',
    ),
    ("R005_HIGH_RISK_COUNTRY", 1, "Counterparty in a high-risk jurisdiction", 30, "{}"),
    (
        "R006_DORMANT_REACTIVATION",
        1,
        "Activity on an account dormant beyond N days",
        20,
        '{"dormancy_days": 180}',
    ),
]

WATCHLIST = [
    ("internal", "Vela Holdings Ltd", "organisation", "CY"),
    ("internal", "Arun Bhaskar", "individual", "IN"),
    ("ofac", "Northbridge Trading FZE", "organisation", "AE"),
]

HIGH_RISK = ["KP", "IR", "SY", "MM"]


def seed_rules(cur) -> None:
    cur.executemany(
        """INSERT INTO rule_definitions (rule_code, version, description, weight, params)
           VALUES (%s, %s, %s, %s, %s)
           ON CONFLICT (rule_code, version) DO NOTHING;""",
        RULES,
    )


def main() -> None:
    print(f"Connecting to DB to seed rules...")
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            seed_rules(cur)
            conn.commit()
    print("Seed complete.")


if __name__ == "__main__":
    main()
