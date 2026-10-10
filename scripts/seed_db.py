
#!/usr/bin/env python
"""Seed AegisFlow reference data and a reproducible demo ledger."""

from __future__ import annotations

import os
import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import psycopg
from psycopg.rows import dict_row


DSN = (
    f"host={os.getenv('DB_HOST', 'localhost')} "
    f"port={os.getenv('DB_PORT', '5433')} "
    f"dbname={os.getenv('DB_NAME', 'aegisflow')} "
    f"user={os.getenv('DB_USER', 'aegis')} "
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
        "High-value transfer to a recently seen beneficiary",
        15,
        '{"account_age_days": 7}',
    ),
    (
        "R005_HIGH_RISK_COUNTRY",
        1,
        "Counterparty in a high-risk jurisdiction",
        30,
        "{}",
    ),
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

DEMO_PREFIX = "AF-M1-DEMO-"
NUM_TRANSACTIONS = 1200
NUM_CUSTOMERS = 60
HIGH_RISK = ["KP", "IR", "SY", "MM"]


def seed_rules(cur):
    """Insert rule definitions without duplicating existing versions."""
    cur.executemany(
        """
        INSERT INTO rule_definitions
            (rule_code, version, description, weight, params)
        VALUES (%s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (rule_code, version) DO NOTHING
        """,
        RULES,
    )


def seed_watchlist(cur):
    """Replace only the watchlist records owned by this demo seed."""
    cur.execute(
        """
        DELETE FROM watchlist_entries
        WHERE list_name = 'aegisflow_m1_demo'
        """
    )

    cur.executemany(
        """
        INSERT INTO watchlist_entries
            (list_name, entity_name, entity_type, country_code)
        VALUES (%s, %s, %s, %s)
        """,
        [
            ("aegisflow_m1_demo", name, entity_type, country)
            for _, name, entity_type, country in WATCHLIST
        ],
    )


def seed_demo_ledger(cur):
    """Generate customers, accounts, counterparties and demo transactions."""

    # Delete only records created by this seed, in foreign-key-safe order.
    cur.execute(
        "DELETE FROM transactions WHERE external_ref LIKE %s",
        (f"{DEMO_PREFIX}%",),
    )
    cur.execute(
        "DELETE FROM counterparties WHERE beneficiary_ref LIKE %s",
        (f"{DEMO_PREFIX}%",),
    )
    cur.execute(
        "DELETE FROM accounts WHERE account_number LIKE %s",
        (f"{DEMO_PREFIX}%",),
    )
    cur.execute(
        "DELETE FROM customers WHERE external_ref LIKE %s",
        (f"{DEMO_PREFIX}%",),
    )

    rng = random.Random(20261010)
    now = datetime.now(timezone.utc)

    # 1. Customers
    customer_ids = []

    for i in range(NUM_CUSTOMERS):
        cur.execute(
            """
            INSERT INTO customers
                (
                    external_ref,
                    full_name,
                    country_code,
                    kyc_status,
                    risk_rating,
                    is_pep,
                    onboarded_at
                )
            VALUES (%s, %s, %s, 'verified', %s, %s, %s)
            RETURNING customer_id
            """,
            (
                f"{DEMO_PREFIX}CUSTOMER-{i:04d}",
                f"Demo Customer {i:04d}",
                "IN",
                rng.choice(["low", "medium", "high"]),
                i % 17 == 0,
                now - timedelta(days=rng.randint(200, 1500)),
            ),
        )

        # dict_row returns a mapping, so access the named column.
        customer_ids.append(cur.fetchone()["customer_id"])

    # 2. Accounts
    account_ids = []

    for i, customer_id in enumerate(customer_ids):
        cur.execute(
            """
            INSERT INTO accounts
                (
                    customer_id,
                    account_number,
                    account_type,
                    currency,
                    opened_on,
                    last_active_on,
                    status
                )
            VALUES (%s, %s, %s, 'INR', %s, %s, 'active')
            RETURNING account_id
            """,
            (
                customer_id,
                f"{DEMO_PREFIX}ACCOUNT-{i:04d}",
                rng.choice(["current", "savings", "card", "wallet"]),
                (now - timedelta(days=rng.randint(200, 1500))).date(),
                (now - timedelta(days=rng.randint(0, 30))).date(),
            ),
        )

        account_ids.append(cur.fetchone()["account_id"])

    # 3. Counterparties
    counterparty_ids = []

    for i, account_id in enumerate(account_ids):
        country = HIGH_RISK[i % len(HIGH_RISK)] if i % 19 == 0 else "IN"

        cur.execute(
            """
            INSERT INTO counterparties
                (
                    account_id,
                    beneficiary_ref,
                    beneficiary_name,
                    country_code,
                    first_seen_at,
                    txn_count
                )
            VALUES (%s, %s, %s, %s, %s, 0)
            RETURNING counterparty_id
            """,
            (
                account_id,
                f"{DEMO_PREFIX}BENEFICIARY-{i:04d}",
                f"Demo Beneficiary {i:04d}",
                country,
                now - timedelta(days=rng.randint(0, 365)),
            ),
        )

        counterparty_ids.append(cur.fetchone()["counterparty_id"])

    # 4. Transactions
    channels = ["card", "wire", "upi", "ach", "atm"]

    for i in range(NUM_TRANSACTIONS):
        account_index = i % NUM_CUSTOMERS

        # Include high-value examples for future rule-scoring tests.
        if i % 100 == 0:
            amount = Decimal(rng.randint(500001, 900000)).quantize(
                Decimal("0.01")
            )
        else:
            amount = Decimal(rng.randint(100, 100000)).quantize(
                Decimal("0.01")
            )

        occurred_at = now - timedelta(
            minutes=rng.randint(0, 60 * 24 * 30)
        )

        cur.execute(
            """
            INSERT INTO transactions
                (
                    external_ref,
                    account_id,
                    counterparty_id,
                    channel,
                    direction,
                    amount,
                    currency,
                    country_code,
                    occurred_at,
                    raw_payload
                )
            VALUES (%s, %s, %s, %s, %s, %s, 'INR', %s, %s, %s::jsonb)
            """,
            (
                f"{DEMO_PREFIX}TXN-{i:06d}",
                account_ids[account_index],
                counterparty_ids[account_index],
                rng.choice(channels),
                rng.choice(["debit", "credit"]),
                amount,
                "IN",
                occurred_at,
                '{"dataset":"synthetic","seed":"aegisflow-m1"}',
            ),
        )

    # Update counterparty transaction counts using this demo's transactions.
    cur.execute(
        """
        UPDATE counterparties AS cp
        SET txn_count = counts.n
        FROM (
            SELECT counterparty_id, COUNT(*)::int AS n
            FROM transactions
            WHERE external_ref LIKE %s
            GROUP BY counterparty_id
        ) AS counts
        WHERE cp.counterparty_id = counts.counterparty_id
        """,
        (f"{DEMO_PREFIX}%",),
    )


def main():
    print("Seeding AegisFlow M1 reference and demo data...")

    with psycopg.connect(DSN, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            seed_rules(cur)
            seed_watchlist(cur)
            seed_demo_ledger(cur)

            cur.execute(
                """
                SELECT COUNT(*) AS n
                FROM transactions
                WHERE external_ref LIKE %s
                """,
                (f"{DEMO_PREFIX}%",),
            )
            txn_count = cur.fetchone()["n"]

            cur.execute(
                """
                SELECT COUNT(*) AS n
                FROM watchlist_entries
                WHERE list_name = 'aegisflow_m1_demo'
                """
            )
            watchlist_count = cur.fetchone()["n"]

    # The connection context commits on success and rolls back on failure.
    print("Seed complete.")
    print(f"Demo transactions: {txn_count}")
    print(f"Demo watchlist entries: {watchlist_count}")


if __name__ == "__main__":
    main()
