"""Shared support module for AegisFlow M2 testing and deterministic data generation.

Provides a reproducible synthetic transaction stream generator with explicit
per-account window boundary fixtures (exact 60s, exact 3,600s, 59s, 61s, 3599s,
3601s, timestamp ties, missing entities, and entity sharing).
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any
import uuid
import numpy as np
import pandas as pd

# Authoritative UUID5 namespace for reproducible synthetic identifiers
_UUID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
_DEFAULT_START_TIME = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def stream(
    n: int = 5000,
    accounts: int = 300,
    devices: int = 150,
    ips: int = 100,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a deterministic synthetic transaction stream with explicit boundary fixtures.

    Parameters:
        n: Total number of transactions (default 5,000).
        accounts: Size of the account pool (default 300).
        devices: Size of the device pool (default 150).
        ips: Size of the IP pool (default 100).
        seed: Random seed for RNG and deterministic UUID generation (default 7).

    Returns:
        pd.DataFrame sorted strictly by event time and identifier: (ts, id).
    """
    rng = np.random.default_rng(seed)
    start_dt = _DEFAULT_START_TIME

    # Deterministic entity identifier pools
    acc_pool = [
        f"acc_{uuid.uuid5(_UUID_NAMESPACE, f'acc:{seed}:{i}')}"
        for i in range(accounts)
    ]
    dev_pool = [
        f"dev_{uuid.uuid5(_UUID_NAMESPACE, f'dev:{seed}:{i}')}"
        for i in range(devices)
    ]
    ip_pool = [
        f"10.0.{i // 256}.{i % 256}"
        for i in range(ips)
    ]

    # Explicit per-account boundary fixtures
    controlled: list[dict[str, Any]] = []
    t_base = start_dt

    # 1. Exact 60-second window boundary on acc_pool[0]
    controlled.append({
        "account_id": acc_pool[0],
        "ts": t_base,
        "amount_minor": 10000,
        "device_id": dev_pool[0],
        "ip_address": ip_pool[0],
    })
    controlled.append({
        "account_id": acc_pool[0],
        "ts": t_base + timedelta(seconds=60),
        "amount_minor": 20000,
        "device_id": dev_pool[0],
        "ip_address": ip_pool[0],
    })

    # 2. Exact 3,600-second window boundary on acc_pool[1]
    controlled.append({
        "account_id": acc_pool[1],
        "ts": t_base,
        "amount_minor": 15000,
        "device_id": dev_pool[1],
        "ip_address": ip_pool[1],
    })
    controlled.append({
        "account_id": acc_pool[1],
        "ts": t_base + timedelta(seconds=3600),
        "amount_minor": 25000,
        "device_id": dev_pool[1],
        "ip_address": ip_pool[1],
    })

    # 3. Just inside (59s) on acc_pool[2]
    controlled.append({
        "account_id": acc_pool[2],
        "ts": t_base,
        "amount_minor": 11000,
        "device_id": dev_pool[2],
        "ip_address": ip_pool[2],
    })
    controlled.append({
        "account_id": acc_pool[2],
        "ts": t_base + timedelta(seconds=59),
        "amount_minor": 12000,
        "device_id": dev_pool[2],
        "ip_address": ip_pool[2],
    })

    # 4. Just outside (61s) on acc_pool[3]
    controlled.append({
        "account_id": acc_pool[3],
        "ts": t_base,
        "amount_minor": 13000,
        "device_id": dev_pool[3],
        "ip_address": ip_pool[3],
    })
    controlled.append({
        "account_id": acc_pool[3],
        "ts": t_base + timedelta(seconds=61),
        "amount_minor": 14000,
        "device_id": dev_pool[3],
        "ip_address": ip_pool[3],
    })

    # 5. Just inside (3599s) on acc_pool[4]
    controlled.append({
        "account_id": acc_pool[4],
        "ts": t_base,
        "amount_minor": 21000,
        "device_id": dev_pool[4],
        "ip_address": ip_pool[4],
    })
    controlled.append({
        "account_id": acc_pool[4],
        "ts": t_base + timedelta(seconds=3599),
        "amount_minor": 22000,
        "device_id": dev_pool[4],
        "ip_address": ip_pool[4],
    })

    # 6. Just outside (3601s) on acc_pool[5]
    controlled.append({
        "account_id": acc_pool[5],
        "ts": t_base,
        "amount_minor": 23000,
        "device_id": dev_pool[5],
        "ip_address": ip_pool[5],
    })
    controlled.append({
        "account_id": acc_pool[5],
        "ts": t_base + timedelta(seconds=3601),
        "amount_minor": 24000,
        "device_id": dev_pool[5],
        "ip_address": ip_pool[5],
    })

    # 7. Same-timestamp transactions on same account (acc_pool[6])
    controlled.append({
        "account_id": acc_pool[6],
        "ts": t_base + timedelta(seconds=120),
        "amount_minor": 31000,
        "device_id": dev_pool[6],
        "ip_address": ip_pool[6],
    })
    controlled.append({
        "account_id": acc_pool[6],
        "ts": t_base + timedelta(seconds=120),
        "amount_minor": 32000,
        "device_id": dev_pool[6],
        "ip_address": ip_pool[6],
    })

    # 8. Same-timestamp transactions across different accounts (acc_pool[7] & acc_pool[8])
    controlled.append({
        "account_id": acc_pool[7],
        "ts": t_base + timedelta(seconds=240),
        "amount_minor": 41000,
        "device_id": dev_pool[7],
        "ip_address": ip_pool[7],
    })
    controlled.append({
        "account_id": acc_pool[8],
        "ts": t_base + timedelta(seconds=240),
        "amount_minor": 42000,
        "device_id": dev_pool[8],
        "ip_address": ip_pool[8],
    })

    # 9. Device and IP sharing across accounts (acc_pool[9] & acc_pool[10])
    controlled.append({
        "account_id": acc_pool[9],
        "ts": t_base + timedelta(seconds=300),
        "amount_minor": 51000,
        "device_id": dev_pool[7],
        "ip_address": ip_pool[7],
    })
    controlled.append({
        "account_id": acc_pool[10],
        "ts": t_base + timedelta(seconds=360),
        "amount_minor": 52000,
        "device_id": dev_pool[7],
        "ip_address": ip_pool[7],
    })

    # 10. Missing entity values on acc_pool[11]
    controlled.append({
        "account_id": acc_pool[11],
        "ts": t_base + timedelta(seconds=400),
        "amount_minor": 61000,
        "device_id": None,
        "ip_address": ip_pool[8],
    })
    controlled.append({
        "account_id": acc_pool[11],
        "ts": t_base + timedelta(seconds=450),
        "amount_minor": 62000,
        "device_id": dev_pool[8],
        "ip_address": None,
    })
    controlled.append({
        "account_id": acc_pool[11],
        "ts": t_base + timedelta(seconds=500),
        "amount_minor": 63000,
        "device_id": None,
        "ip_address": None,
    })

    # Fill remainder of stream with chronological pseudo-random transactions
    num_controlled = len(controlled)
    num_random = max(0, n - num_controlled)

    # Specified gaps including timestamp ties (0s), immediate (1s, 5s), windows (59s, 60s, 61s, 600s, 3600s, 86400s)
    gaps_pool = [0, 1, 5, 30, 59, 60, 61, 600, 3600, 86400]
    gap_weights = [0.05, 0.20, 0.20, 0.20, 0.05, 0.05, 0.05, 0.10, 0.05, 0.05]

    bg_rows: list[dict[str, Any]] = []
    current_ts = t_base + timedelta(seconds=1)
    # Background accounts pool (leaves controlled accounts strictly for boundary fixtures)
    bg_accounts = acc_pool[12:] if len(acc_pool) > 12 else acc_pool

    for _ in range(num_random):
        gap = int(rng.choice(gaps_pool, p=gap_weights))
        current_ts = current_ts + timedelta(seconds=gap)

        acc = rng.choice(bg_accounts)
        dev = None if rng.random() < 0.15 else rng.choice(dev_pool)
        ip = None if rng.random() < 0.15 else rng.choice(ip_pool)
        amt_minor = int(rng.integers(100, 500000))

        bg_rows.append({
            "account_id": acc,
            "ts": current_ts,
            "amount_minor": amt_minor,
            "device_id": dev,
            "ip_address": ip,
        })

    all_rows = controlled + bg_rows

    # Generate deterministic UUID5 identifiers and amount fields
    for i, r in enumerate(all_rows):
        txn_uuid = str(uuid.uuid5(_UUID_NAMESPACE, f"txn:{seed}:{i:06d}"))
        r["id"] = txn_uuid
        r["txn_id"] = txn_uuid
        r["timestamp"] = r["ts"]
        r["amount"] = r["amount_minor"] / 100.0

    df = pd.DataFrame(all_rows)
    # Strictly sort by (ts, id)
    df = df.sort_values(by=["ts", "id"]).reset_index(drop=True)
    return df
