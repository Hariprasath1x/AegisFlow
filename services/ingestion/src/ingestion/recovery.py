"""Recover pending Redis online features from durable PostgreSQL API transactions.

Preserves strict monotonic `(ts, txn_id)` validation inside `online_features.update`
and synchronizes per-account rebuilds with live ingestion via renewable, token-fenced
per-account Redis locks (`lock:acct:{account_id}`):
- An active recovery holder renews its own lease (`GET == token -> PEXPIRE`) periodically
  so slow legitimate rebuilds do not lose ownership mid-flight.
- Both account key resets and every replayed `online_features.update` execute with atomic
  server-side Lua token verification (`GET lock_key == token`); if a lease ever expires
  due to a host pause, stale mutations are rejected before touching Redis.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.engine import Engine

from aegis_ml.features import definitions as D
from ingestion import online_features
from ingestion.locks import AccountLockLostError, account_lock
from ingestion.schemas import TransactionIn

_RESET_FENCED_LUA = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then
    return 0
end
for i = 2, #KEYS do
    redis.call('DEL', KEYS[i])
end
return 1
"""


@dataclass
class RecoverySummary:
    scanned_txns: int = 0
    pending_txns: int = 0
    affected_accounts: int = 0
    tail_accounts: int = 0
    rebuild_accounts: int = 0
    replayed_txns: int = 0
    recovered_txns: int = 0
    pending_txn_ids: list[str] = field(default_factory=list)
    account_ids: list[str] = field(default_factory=list)


def _fetch_api_transactions(
    engine: Engine,
    *,
    account_id: UUID | None = None,
    lookback_seconds: int | None = D.KEY_TTL_S,
) -> list[TransactionIn]:
    """Load durable API-ingested transactions ordered by account and event chronology."""
    if lookback_seconds is not None and lookback_seconds < 0:
        raise ValueError(f"lookback_seconds must be >= 0 or None, got {lookback_seconds}")

    filters = [
        "raw_payload IS NOT NULL",
        "jsonb_typeof(raw_payload) = 'object'",
        "raw_payload ? 'id'",
        "raw_payload ? 'ts'",
        "external_ref = txn_id::text",
    ]
    params: dict[str, Any] = {}
    if account_id is not None:
        filters.append("account_id = :account_id")
        params["account_id"] = account_id
    if lookback_seconds is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=lookback_seconds)
        filters.append("occurred_at >= :cutoff")
        params["cutoff"] = cutoff

    stmt = text(
        f"""
        SELECT raw_payload
        FROM transactions
        WHERE {' AND '.join(filters)}
        ORDER BY account_id ASC, occurred_at ASC, txn_id ASC
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt, params).fetchall()

    return [TransactionIn.model_validate(payload) for (payload,) in rows]


def _find_pending_ids(r: Any, txns: list[TransactionIn]) -> set[UUID]:
    """Return subset of transaction IDs lacking `seen:txn:{id}` in Redis."""
    if not txns:
        return set()
    pipe = r.pipeline()
    for t in txns:
        pipe.exists(f"seen:txn:{t.id}")
    flags = pipe.execute()
    return {t.id for t, exists in zip(txns, flags) if not exists}


def _redis_last_marker(r: Any, account_id: UUID) -> tuple[float, str] | None:
    """Read `(last_ts, last_id)` from Redis `feat:acct:{account_id}` if present."""
    raw_ts, raw_id = r.hmget(f"feat:acct:{account_id}", "last_ts", "last_id")
    if raw_ts is None:
        return None
    last_ts = float(raw_ts.decode("utf-8") if isinstance(raw_ts, bytes) else raw_ts)
    last_id = (
        raw_id.decode("utf-8") if isinstance(raw_id, bytes) else (str(raw_id) if raw_id else "")
    )
    return (last_ts, last_id)


def _is_tail_only_pending(
    r: Any,
    account_txns: list[TransactionIn],
    pending_ids: set[UUID],
) -> bool:
    """Return True if pending events sit strictly after current Redis `(last_ts, last_id)`."""
    marker = _redis_last_marker(r, account_txns[0].account_id)
    if marker is None:
        return True

    last_ts, last_id = marker
    seen_pending = False
    for t in account_txns:
        t_ts = float(f"{D.epoch_seconds(t.ts):.6f}")
        t_key = (t_ts, str(t.id))
        if t.id in pending_ids:
            if t_key <= (last_ts, last_id):
                return False
            seen_pending = True
        elif seen_pending:
            return False
    return seen_pending


def _reset_account_keys(
    r: Any,
    account_id: UUID,
    txn_ids: list[UUID],
    *,
    lock_key: str | None = None,
    lock_token: str | None = None,
) -> None:
    """Delete account-scoped feature/velocity/distinct keys, fenced atomically if locked."""
    acct = str(account_id)
    target_keys = [
        f"feat:acct:{acct}",
        *(f"vel:acct:{acct}:{w}" for w in D.WINDOWS),
        f"dist:acct:{acct}:dev",
        f"dist:acct:{acct}:ip",
        *(f"seen:txn:{tid}" for tid in txn_ids),
    ]
    if lock_key is None or lock_token is None:
        r.delete(*target_keys)
        return

    res = r.eval(_RESET_FENCED_LUA, 1 + len(target_keys), lock_key, *target_keys, lock_token)
    ok = bool(int(res)) if isinstance(res, (int, bytes, str)) else bool(res)
    if not ok:
        raise AccountLockLostError(
            f"Lock fence rejected stale key reset for account {acct}"
        )


def _sort_chronological(txns: list[TransactionIn]) -> list[TransactionIn]:
    return sorted(txns, key=lambda item: (item.ts.astimezone(timezone.utc), str(item.id)))


def recover_pending_features(
    engine: Engine,
    r: Any,
    *,
    account_id: UUID | None = None,
    lookback_seconds: int | None = D.KEY_TTL_S,
    dry_run: bool = False,
    lock_ttl_ms: int = 5_000,
    lock_wait_timeout_s: float = 5.0,
    auto_renew: bool = True,
    renew_interval_s: float | None = None,
) -> RecoverySummary:
    """Reconcile persisted API transactions missing Redis online feature updates."""
    txns = _fetch_api_transactions(
        engine, account_id=account_id, lookback_seconds=lookback_seconds
    )
    pending_ids = _find_pending_ids(r, txns)

    by_account: dict[UUID, list[TransactionIn]] = defaultdict(list)
    for t in txns:
        by_account[t.account_id].append(t)

    affected_accts = [
        acct
        for acct, acct_txns in by_account.items()
        if any(item.id in pending_ids for item in acct_txns)
    ]

    summary = RecoverySummary(
        scanned_txns=len(txns),
        pending_txns=len(pending_ids),
        affected_accounts=len(affected_accts),
        pending_txn_ids=[str(t.id) for t in txns if t.id in pending_ids],
        account_ids=[str(acct) for acct in affected_accts],
    )

    if dry_run:
        for acct in affected_accts:
            acct_txns = _sort_chronological(by_account[acct])
            if _is_tail_only_pending(r, acct_txns, pending_ids):
                summary.tail_accounts += 1
            else:
                summary.rebuild_accounts += 1
        return summary

    # Hold per-account renewable lease + atomic Lua fence for every reset and replay write.
    for acct in affected_accts:
        with account_lock(
            r,
            acct,
            ttl_ms=lock_ttl_ms,
            wait_timeout_s=lock_wait_timeout_s,
            auto_renew=auto_renew,
            renew_interval_s=renew_interval_s,
        ) as lease:
            lease.assert_held()
            acct_txns = _sort_chronological(
                _fetch_api_transactions(
                    engine, account_id=acct, lookback_seconds=lookback_seconds
                )
            )
            if not acct_txns:
                continue
            acct_pending = _find_pending_ids(r, acct_txns)
            if not acct_pending:
                continue

            lease.assert_held()
            tail_only = _is_tail_only_pending(r, acct_txns, acct_pending)
            if tail_only:
                summary.tail_accounts += 1
                to_replay = [t for t in acct_txns if t.id in acct_pending]
            else:
                summary.rebuild_accounts += 1
                _reset_account_keys(
                    r,
                    acct,
                    [t.id for t in acct_txns],
                    lock_key=lease.key,
                    lock_token=lease.token,
                )
                to_replay = acct_txns

            with online_features.fenced_by(lease.key, lease.token):
                for t in to_replay:
                    lease.assert_held()
                    res = online_features.update(r, t.feature_input())
                    summary.replayed_txns += 1
                    if res is not None and t.id in acct_pending:
                        summary.recovered_txns += 1

    return summary
