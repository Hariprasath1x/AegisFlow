# ADR-0004: Feature Store Parity and Entity Graph Rules (M2)

**Status:** Accepted (M2)
**Date:** 2026-11-26
**Note on Numbering:** The M2 Implementation Guide labels this ADR `0001-feature-store.md`; numbered `0004` here to avoid colliding with existing `0001-kafka-as-transport.md`.

## Context

AegisFlow Milestone 2 introduces a dual-path feature store (`ml/src/aegis_ml/features/definitions.py` declaring 19 canonical features) compiled to an atomic Redis Lua script (`services/ingestion/src/ingestion/online_features.py`) for online serving and a point-in-time batch builder (`ml/src/aegis_ml/features/offline.py`) for offline training, plus an undirected entity graph (`ml/src/aegis_ml/features/graph_features.py`). Online/offline skew or label leakage would invalidate downstream M3 models.

## Decisions (D1–D8)

- **D1 — Exact float round-tripping (`%.17g`):**
  Redis Lua truncates returned numbers to integers. All numeric outputs from the Lua script leave Redis formatted via `string.format('%.17g', x)` and are parsed back as IEEE-754 `float64` in Python, guaranteeing bit-exact parity (`check_exact=True`) across 5,000 replayed transactions.

- **D2 — Event-time semantics:**
  All sliding windows, recency (`seconds_since_prev`), and calendar features use the transaction's own timezone-aware UTC event timestamp (`ts` / `occurred_at`), never Redis server `TIME` or wall-clock ingestion time. Offline and online paths share `definitions.epoch_seconds()` so 10x replay streams match batch computation identically.

- **D3 — Velocity window interval `(ts - w, ts]` and `(ts, id)` tie ordering:**
  Velocity and distinct/sharing windows are half-open `(ts - w, ts]` with the current transaction included (`include_current=True`) and ties ordered deterministically by `(ts, id)` in both `offline.prepare()` and online replay.

- **D4 — Idempotency (`SET seen:txn:{id} NX`) and integer minor units:**
  Monetary amounts are converted to 64-bit integer minor units (`definitions.amount_minor`) before entering Lua or two-pointer cumulative sums, eliminating floating-point drift. Inside Lua, `SET seen:txn:{id} '1' NX EX <KEY_TTL_S>` prevents duplicate feature updates on retried requests; per-account Redis lease locks (`services/ingestion/src/ingestion/locks.py`) and `recovery.py` reconcile pending Redis updates if Redis fails after PostgreSQL commit.

- **D5 — Welford pre-update `amount_zscore`:**
  `amount_zscore` is computed against the account's Welford running profile (`n`, `mean`, `m2` stored in `feat:acct:{id}`) **before** the current transaction's amount updates the profile (`include_current=False`). Both `online_features.py` and `offline._compute_account_stats()` execute this exact ordering.

- **D6 — Pre-T0 known-fraud eligibility without serving-path label leakage:**
  Known-fraud seeding for graph risk features uses confirmed fraud labels strictly prior to cutoff `T0` (`features.known_fraud_cutoff` in `params.yaml`, derived as `2025-06-14T00:00:00Z` from the canonical 6-scenario manifest). In this repository, `m2_0001`–`m2_0003` store labels in `ground_truth_labels` (rather than a separate `known_fraud_entities` table), and `graph_features.fetch_fraud_account_ids()` filters `is_fraud IS TRUE`, `confidence >= 0.85`, `labeled_at <= snapshot_at`, and incident/transaction timestamps `<= snapshot_at`. Serving ingestion never reads `ground_truth_labels`.

- **D7 — Unique entity-edge key (`uq_entity_edges_identity`):**
  `entity_edges` enforces uniqueness on `(src_type, src_id, dst_type, dst_id, edge_type)` (`m2_0001_feature_store.py`). Both `PgStore.store()` and `db/seed/seed_known_fraud.py` upsert edge weights **only** when a transaction row is newly inserted (`ON CONFLICT DO NOTHING RETURNING txn_id`), ensuring HTTP retries and idempotent re-seeding never inflate edge weights.

- **D8 — Hourly graph snapshots (`snapshot_at`) and deferred entity upsert endpoints:**
  `graph_features.py` persists hourly undirected NetworkX snapshots to `graph_feature_snapshots` (column `snapshot_at`) and publishes `gfeat:acct:{id}` to Redis with `GRAPH_TTL_S` (`7200s`). Offline PIT extraction (`offline.py`) joins backward-only (`snapshot_at <= ts`) and tracks `feature_snapshot_at`. Standalone entity upsert endpoints remain deferred to keep M1 entity schemas decoupled from generic ingestion upserts.

## Known Limitations & Repository Adaptations

- `ground_truth_labels` (extended by `m2_0003`) serves as the single ground-truth table instead of creating a separate `known_fraud_entities` table.
- `graph_feature_snapshots` uses `snapshot_at` (and stores `component_size`, `dist_to_fraud`, `risk_rank` inside `features` JSONB alongside typed `pagerank`, `degree`, `clustering_coefficient` columns).
- Host-side Docker services bind PostgreSQL on `5433` and Redis on `6380`, while container-internal `docker-compose.m2.yml` connects to `postgres:5432` and `redis:6379` on `aegisnet`.
