"""Reproducible synthetic scenario generator for laundering typologies and ground-truth episodes.

Generates:
1. data/processed/synthetic_transactions.parquet
2. data/processed/typology_episodes.parquet

Required for Milestone 3 Step 3.12 typology evaluation (Objective 5.4).
Can be executed from a clean checkout without Kaggle credentials.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import pathlib
import sys
from typing import Any
import numpy as np
import pandas as pd

from aegis_ml.evaluation.typology_eval import TYPOLOGIES

# Fixed seed matching scenario namespace default
DEFAULT_SEED: int = 20261010
DEFAULT_OUT: pathlib.Path = pathlib.Path("data/processed")


def generate_typology_datasets(
    out_dir: pathlib.Path | str = DEFAULT_OUT,
    seed: int = DEFAULT_SEED,
    episodes_per_typology: int = 20,
    background_txns: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate deterministic synthetic transactions and ground-truth laundering episodes.

    Returns:
        tuple of (synthetic_transactions_df, typology_episodes_df)
    """
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    base_time = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)

    all_txns: list[dict[str, Any]] = []
    all_episodes: list[dict[str, Any]] = []

    txn_counter = 0

    def next_txn_id(prefix: str) -> str:
        nonlocal txn_counter
        txn_counter += 1
        return f"txn_{prefix}_{txn_counter:07d}"

    # 1. Generate Smurfing Episodes (structuring into a single beneficiary)
    for ep_idx in range(episodes_per_typology):
        ep_id = f"ep_smurfing_{ep_idx:04d}"
        target_account = f"acc_mule_dest_{ep_idx:04d}"
        num_deposits = int(rng.integers(6, 12))
        ep_start_offset = float(rng.uniform(0, 20 * 86400))  # within first 20 days
        ep_start_time = base_time + timedelta(seconds=ep_start_offset)

        ep_txn_ids: list[str] = []
        ep_total_amount = 0.0

        for dep_i in range(num_deposits):
            txn_id = next_txn_id("smurf")
            src_account = f"acc_smurf_src_{ep_idx:04d}_{dep_i:02d}"
            # Deposit amounts structured just below reporting thresholds
            amount = float(np.round(rng.uniform(7500.0, 9800.0), 2))
            txn_time = ep_start_time + timedelta(minutes=float(dep_i * rng.uniform(8, 25)))

            ep_txn_ids.append(txn_id)
            ep_total_amount += amount

            all_txns.append(
                {
                    "transaction_id": txn_id,
                    "ts": txn_time,
                    "account_id": src_account,
                    "destination_account_id": target_account,
                    "amount": amount,
                    "channel": "ach" if dep_i % 2 == 0 else "wire",
                    "card_network": "none",
                    "card_type": "none",
                    "device_type": "mobile" if rng.random() > 0.3 else "desktop",
                    "email_domain": rng.choice(["gmail.com", "yahoo.com", "protonmail.com"]),
                    "region": rng.choice(["US", "EU", "APAC"]),
                    "device_id": f"dev_smurf_{ep_idx:04d}_{dep_i:02d}",
                    "ip_address": f"198.51.100.{int(rng.integers(1, 254))}",
                    "is_laundering": 1,
                    "typology": "smurfing",
                    "episode_id": ep_id,
                }
            )

        all_episodes.append(
            {
                "episode_id": ep_id,
                "typology": "smurfing",
                "transaction_ids": ep_txn_ids,
                "num_transactions": len(ep_txn_ids),
                "total_amount": float(np.round(ep_total_amount, 2)),
                "started_at": ep_start_time,
                "ended_at": ep_start_time + timedelta(minutes=float(num_deposits * 25)),
            }
        )

    # 2. Generate Layering Episodes (multi-hop rapid sequential transfers)
    for ep_idx in range(episodes_per_typology):
        ep_id = f"ep_layering_{ep_idx:04d}"
        num_hops = int(rng.integers(4, 8))
        ep_start_offset = float(rng.uniform(0, 20 * 86400))
        ep_start_time = base_time + timedelta(seconds=ep_start_offset)

        chain_accounts = [f"acc_layer_{ep_idx:04d}_{h:02d}" for h in range(num_hops + 1)]
        initial_amount = float(np.round(rng.uniform(25000.0, 75000.0), 2))
        current_amount = initial_amount

        ep_txn_ids = []
        ep_total_amount = 0.0

        for h in range(num_hops):
            txn_id = next_txn_id("layer")
            # Slight amount reduction simulating transfer fee or structuring retention
            current_amount = float(np.round(current_amount * rng.uniform(0.96, 0.99), 2))
            txn_time = ep_start_time + timedelta(minutes=float(h * rng.uniform(10, 30)))

            ep_txn_ids.append(txn_id)
            ep_total_amount += current_amount

            all_txns.append(
                {
                    "transaction_id": txn_id,
                    "ts": txn_time,
                    "account_id": chain_accounts[h],
                    "destination_account_id": chain_accounts[h + 1],
                    "amount": current_amount,
                    "channel": "transfer" if h % 2 == 0 else "wire",
                    "card_network": "none",
                    "card_type": "none",
                    "device_type": "desktop",
                    "email_domain": rng.choice(["corporate-net.com", "consulting.org", "mail.com"]),
                    "region": rng.choice(["US", "EU"]),
                    "device_id": f"dev_layer_{ep_idx:04d}_{h:02d}",
                    "ip_address": f"203.0.113.{int(rng.integers(1, 254))}",
                    "is_laundering": 1,
                    "typology": "layering",
                    "episode_id": ep_id,
                }
            )

        all_episodes.append(
            {
                "episode_id": ep_id,
                "typology": "layering",
                "transaction_ids": ep_txn_ids,
                "num_transactions": len(ep_txn_ids),
                "total_amount": float(np.round(ep_total_amount, 2)),
                "started_at": ep_start_time,
                "ended_at": ep_start_time + timedelta(minutes=float(num_hops * 30)),
            }
        )

    # 3. Generate Round-Tripping Episodes (circular funds returning to source)
    for ep_idx in range(episodes_per_typology):
        ep_id = f"ep_round_tripping_{ep_idx:04d}"
        num_hops = int(rng.integers(3, 5))
        ep_start_offset = float(rng.uniform(0, 20 * 86400))
        ep_start_time = base_time + timedelta(seconds=ep_start_offset)

        ring_accounts = [f"acc_round_{ep_idx:04d}_{h:02d}" for h in range(num_hops)]
        # Last transfer returns funds back to the first account in the ring
        dest_accounts = ring_accounts[1:] + [ring_accounts[0]]

        ep_amount = float(np.round(rng.uniform(30000.0, 90000.0), 2))
        ep_txn_ids = []
        ep_total_amount = 0.0

        for h in range(num_hops):
            txn_id = next_txn_id("round")
            txn_time = ep_start_time + timedelta(hours=float(h * rng.uniform(1.5, 4.0)))

            ep_txn_ids.append(txn_id)
            ep_total_amount += ep_amount

            all_txns.append(
                {
                    "transaction_id": txn_id,
                    "ts": txn_time,
                    "account_id": ring_accounts[h],
                    "destination_account_id": dest_accounts[h],
                    "amount": ep_amount,
                    "channel": "wire",
                    "card_network": "none",
                    "card_type": "none",
                    "device_type": "desktop",
                    "email_domain": rng.choice(["finholdings.com", "importexport.net", "globaltrade.org"]),
                    "region": "US",
                    "device_id": f"dev_round_{ep_idx:04d}_{h:02d}",
                    "ip_address": f"192.0.2.{int(rng.integers(1, 254))}",
                    "is_laundering": 1,
                    "typology": "round_tripping",
                    "episode_id": ep_id,
                }
            )

        all_episodes.append(
            {
                "episode_id": ep_id,
                "typology": "round_tripping",
                "transaction_ids": ep_txn_ids,
                "num_transactions": len(ep_txn_ids),
                "total_amount": float(np.round(ep_total_amount, 2)),
                "started_at": ep_start_time,
                "ended_at": ep_start_time + timedelta(hours=float(num_hops * 4.0)),
            }
        )

    # 4. Generate Background Legitimate Retail/Banking Transactions
    channels = ["card", "card", "card", "ach", "wire", "wallet"]
    card_nets = ["visa", "mastercard", "amex"]
    card_types = ["debit", "credit"]
    domains = ["gmail.com", "yahoo.com", "outlook.com", "icloud.com"]
    regions = ["US", "US", "US", "EU", "APAC"]

    for b_idx in range(background_txns):
        txn_id = next_txn_id("clean")
        acct_id = f"acc_retail_{int(rng.integers(1, 150)):04d}"
        dest_id = f"merch_retail_{int(rng.integers(1, 300)):04d}"
        offset = float(rng.uniform(0, 25 * 86400))
        txn_time = base_time + timedelta(seconds=offset)
        amt = float(np.round(rng.exponential(scale=65.0) + 5.0, 2))

        ch = str(rng.choice(channels))
        all_txns.append(
            {
                "transaction_id": txn_id,
                "ts": txn_time,
                "account_id": acct_id,
                "destination_account_id": dest_id,
                "amount": amt,
                "channel": ch,
                "card_network": str(rng.choice(card_nets)) if ch == "card" else "none",
                "card_type": str(rng.choice(card_types)) if ch == "card" else "none",
                "device_type": str(rng.choice(["mobile", "desktop", "tablet"])),
                "email_domain": str(rng.choice(domains)),
                "region": str(rng.choice(regions)),
                "device_id": f"dev_retail_{int(rng.integers(1, 200)):04d}",
                "ip_address": f"10.0.{int(rng.integers(1, 50))}.{int(rng.integers(1, 254))}",
                "is_laundering": 0,
                "typology": "none",
                "episode_id": "none",
            }
        )

    # Convert to DataFrames
    txns_df = pd.DataFrame(all_txns)
    # Strictly sort chronologically by timestamp
    txns_df = txns_df.sort_values("ts").reset_index(drop=True)

    episodes_df = pd.DataFrame(all_episodes)
    episodes_df = episodes_df.sort_values("started_at").reset_index(drop=True)

    # Persist as Parquet files
    txns_path = out_path / "synthetic_transactions.parquet"
    episodes_path = out_path / "typology_episodes.parquet"

    txns_df.to_parquet(txns_path, index=False)
    episodes_df.to_parquet(episodes_path, index=False)

    return txns_df, episodes_df


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic transactions and ground-truth laundering episodes."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT),
        help="Target directory for generated Parquet files (default: data/processed)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic PRNG seed (default: 20261010)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes to generate per typology (default: 20)",
    )
    parser.add_argument(
        "--background",
        type=int,
        default=1000,
        help="Number of background clean transactions (default: 1000)",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    opts = parse_args(args)
    txns, episodes = generate_typology_datasets(
        out_dir=opts.out,
        seed=opts.seed,
        episodes_per_typology=opts.episodes,
        background_txns=opts.background,
    )
    print(
        f"Generated {len(txns)} transactions and {len(episodes)} episodes "
        f"across typologies {TYPOLOGIES} into {opts.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
