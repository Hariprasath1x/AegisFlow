#!/usr/bin/env python
"""M1 acceptance checks for AegisFlow. Exit code 0 means the milestone is done."""

from __future__ import annotations

import os
import subprocess
import sys

import psycopg

DSN = (
    f"host={os.getenv('DB_HOST', 'localhost')} "
    f"port={os.getenv('DB_PORT', '5433')} "
    f"dbname={os.getenv('DB_NAME', 'aegisflow')} "
    f"user={os.getenv('DB_USER', 'aegis')} "
    f"password={os.getenv('DB_PASS', 'aegis_dev_2026')}"
)

EXPECTED_TABLES = {
    "customers",
    "accounts",
    "counterparties",
    "transactions",
    "velocity_snapshots",
    "rule_definitions",
    "rule_hits",
    "risk_assessments",
    "alerts",
    "cases",
    "case_alerts",
    "case_notes",
    "watchlist_entries",
    "audit_log",
}

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"{status} {label}{suffix}")

    if not ok:
        failures.append(label)


def main() -> int:
    print("AegisFlow M1 Acceptance Verification")
    print("=" * 40)

    try:
        with psycopg.connect(DSN) as conn, conn.cursor() as cur:
            # 1. Verify all 14 expected tables.
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_type = 'BASE TABLE'
                """
            )
            found = {row[0] for row in cur.fetchall()}
            missing = EXPECTED_TABLES - found

            check(
                "schema: all 14 tables present",
                not missing,
                f"missing {sorted(missing)}" if missing else "14/14",
            )

            # 2. Verify the rule catalogue.
            cur.execute("SELECT COUNT(*) FROM rule_definitions")
            rule_count = cur.fetchone()[0]

            check(
                "seed: rule catalogue",
                rule_count >= 6,
                f"{rule_count} rules",
            )

            # 3. Verify the demo ledger.
            cur.execute("SELECT COUNT(*) FROM transactions")
            transaction_count = cur.fetchone()[0]

            check(
                "seed: demo ledger",
                transaction_count >= 1000,
                f"{transaction_count} transactions",
            )

            # 4. Verify the watchlist.
            cur.execute("SELECT COUNT(*) FROM watchlist_entries")
            watchlist_count = cur.fetchone()[0]

            check(
                "seed: watchlist",
                watchlist_count >= 3,
                f"{watchlist_count} entries",
            )

    except Exception as exc:
        check("database: connection and acceptance queries", False, str(exc))
        print("\nM1 NOT COMPLETE — database verification failed")
        return 1

    # 5. Run the domain unit tests using the current Python environment.
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        capture_output=True,
        text=True,
    )

    output = (result.stdout or "").strip()
    if result.stderr.strip():
        output = f"{output}\n{result.stderr.strip()}".strip()

    detail = output.splitlines()[-1] if output else "No test output"

    check(
        "domain: unit tests",
        result.returncode == 0,
        detail,
    )

    print()

    if failures:
        print(f"M1 NOT COMPLETE — {len(failures)} check(s) failed")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("M1 COMPLETE — safe to tag v0.1.0-M1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
