"""Focused safety, interruption, concurrency, and retention tests for pending feature recovery."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import os
import threading
import time
from uuid import uuid4

import fakeredis
from fastapi.testclient import TestClient
import psycopg
import pytest
import redis
from sqlalchemy import create_engine

from aegis_ml.features import definitions as D
from ingestion import config, deps, online_features
import ingestion.recovery as recovery_mod
from ingestion.locks import AccountLockLostError, AccountLockTimeout, account_lock
from ingestion.main import app
from ingestion.persistence import PgStore
from ingestion.recovery import _is_tail_only_pending, recover_pending_features
from ingestion.schemas import TransactionIn


DATABASE_URL = os.getenv("DATABASE_URL", config.DATABASE_URL)
PG_URL = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")


@pytest.fixture(scope="function")
def pg_accounts_fixture():
    """Create two isolated disposable accounts and tear down all test rows on exit."""
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        store = PgStore(engine)
        if not store.ping():
            pytest.skip("PostgreSQL unavailable for recovery test")
    except Exception as exc:
        pytest.skip(f"PostgreSQL connection failed: {exc}")

    cust_id = uuid4()
    acc_target = uuid4()
    acc_other = uuid4()
    created_txns: list = []

    with psycopg.connect(PG_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO customers (customer_id, external_ref, full_name, country_code) VALUES (%s, %s, 'Recovery Cust', 'IN');",
                (cust_id, f"REC-{cust_id.hex[:16]}"),
            )
            cur.execute(
                "INSERT INTO accounts (account_id, customer_id, account_number, account_type, currency, opened_on) "
                "VALUES (%s, %s, %s, 'savings', 'INR', CURRENT_DATE);",
                (acc_target, cust_id, f"RECA1-{acc_target.hex[:16]}"),
            )
            cur.execute(
                "INSERT INTO accounts (account_id, customer_id, account_number, account_type, currency, opened_on) "
                "VALUES (%s, %s, %s, 'savings', 'INR', CURRENT_DATE);",
                (acc_other, cust_id, f"RECA2-{acc_other.hex[:16]}"),
            )
            conn.commit()

    yield engine, store, acc_target, acc_other, created_txns

    with psycopg.connect(PG_URL) as conn:
        with conn.cursor() as cur:
            ids = [str(acc_target), str(acc_other)]
            cur.execute(
                "DELETE FROM entity_edges WHERE (src_type='account' AND src_id=ANY(%s)) OR (dst_type='account' AND dst_id=ANY(%s));",
                (ids, ids),
            )
            if created_txns:
                cur.execute("DELETE FROM transactions WHERE txn_id = ANY(%s);", (created_txns,))
            cur.execute("DELETE FROM accounts WHERE customer_id = %s;", (cust_id,))
            cur.execute("DELETE FROM customers WHERE customer_id = %s;", (cust_id,))
            conn.commit()
    engine.dispose()


def _make_txn(acc_id, ts: datetime, amount: str, dev_id, ip_id) -> TransactionIn:
    return TransactionIn(
        id=uuid4(),
        ts=ts,
        account_id=acc_id,
        device_id=dev_id,
        ip_id=ip_id,
        amount=Decimal(amount),
        currency="INR",
        channel="card",
    )


def test_tail_recovery_idempotency_and_ttl(pg_accounts_fixture) -> None:
    """1, 3, 7, 9: Tail recovery, repeated no-op idempotency, and Redis TTL check."""
    engine, store, acc_target, acc_other, created_txns = pg_accounts_fixture
    r = fakeredis.FakeRedis()
    base = datetime.now(timezone.utc) - timedelta(minutes=20)
    dev_id, ip_id = uuid4(), uuid4()

    t1 = _make_txn(acc_target, base, "100.00", dev_id, ip_id)
    t2 = _make_txn(acc_target, base + timedelta(minutes=5), "250.00", dev_id, ip_id)
    t_other = _make_txn(acc_other, base + timedelta(minutes=2), "500.00", dev_id, ip_id)
    created_txns.extend([t1.id, t2.id, t_other.id])

    for item in (t1, t2, t_other):
        assert store.store(item) is True

    online_features.update(r, t1.feature_input())
    online_features.update(r, t_other.feature_input())
    assert _is_tail_only_pending(r, [t1, t2], {t2.id}) is True

    dry = recover_pending_features(engine, r, account_id=acc_target, dry_run=True)
    assert dry.pending_txns == 1
    assert dry.tail_accounts == 1
    assert dry.replayed_txns == 0

    rec = recover_pending_features(engine, r, account_id=acc_target)
    assert rec.pending_txns == 1
    assert rec.tail_accounts == 1
    assert rec.rebuild_accounts == 0
    assert rec.replayed_txns == 1
    assert rec.recovered_txns == 1

    assert r.exists(f"seen:txn:{t2.id}") == 1
    assert 0 < r.ttl(f"seen:txn:{t2.id}") <= D.KEY_TTL_S
    assert 0 < r.ttl(f"feat:acct:{acc_target}") <= D.KEY_TTL_S
    assert 0 < r.ttl(f"vel:acct:{acc_target}:1h") <= D.KEY_TTL_S
    assert int(r.hget(f"feat:acct:{acc_target}", "w_count")) == 2
    assert int(r.hget(f"feat:acct:{acc_target}", "sum_1h")) == 35000
    assert r.zcard(f"vel:acct:{acc_target}:1h") == 2

    # Subsequent run must be an idempotent no-op.
    second = recover_pending_features(engine, r, account_id=acc_target)
    assert second.pending_txns == 0
    assert second.replayed_txns == 0
    assert int(r.hget(f"feat:acct:{acc_target}", "w_count")) == 2
    assert int(r.hget(f"feat:acct:{acc_target}", "sum_1h")) == 35000


def test_mid_timeline_rebuild_restores_exact_features_and_distinct_sets(pg_accounts_fixture) -> None:
    """2, 7, 8: Mid-timeline rebuild restores velocity, distinct sets, Welford stats, and isolation."""
    engine, store, acc_target, acc_other, created_txns = pg_accounts_fixture
    r = fakeredis.FakeRedis()
    base = datetime.now(timezone.utc) - timedelta(minutes=30)
    dev_1, dev_2 = uuid4(), uuid4()
    ip_1, ip_2 = uuid4(), uuid4()

    t1 = _make_txn(acc_target, base, "100.00", dev_1, ip_1)
    t2_mid = _make_txn(acc_target, base + timedelta(minutes=10), "200.00", dev_2, ip_2)
    t3 = _make_txn(acc_target, base + timedelta(minutes=20), "300.00", dev_1, ip_1)
    t_other = _make_txn(acc_other, base + timedelta(minutes=15), "700.00", dev_1, ip_1)
    created_txns.extend([t1.id, t2_mid.id, t3.id, t_other.id])

    for item in (t1, t3, t_other, t2_mid):
        assert store.store(item) is True

    online_features.update(r, t1.feature_input())
    online_features.update(r, t3.feature_input())
    online_features.update(r, t_other.feature_input())
    with pytest.raises(ValueError, match="Out-of-order"):
        online_features.update(r, t2_mid.feature_input())

    rec = recover_pending_features(engine, r, account_id=acc_target)
    assert rec.pending_txns == 1
    assert rec.rebuild_accounts == 1
    assert rec.replayed_txns == 3
    assert rec.recovered_txns == 1

    assert int(r.hget(f"feat:acct:{acc_target}", "w_count")) == 3
    assert int(r.hget(f"feat:acct:{acc_target}", "sum_1h")) == 60000
    assert r.hget(f"feat:acct:{acc_target}", "last_id").decode() == str(t3.id)
    assert r.zcard(f"vel:acct:{acc_target}:1h") == 3
    assert r.zcard(f"dist:acct:{acc_target}:dev") == 2
    assert r.zcard(f"dist:acct:{acc_target}:ip") == 2
    assert r.zcard(f"share:dev:{dev_1}") == 2
    assert int(r.hget(f"feat:acct:{acc_other}", "w_count")) == 1
    assert int(r.hget(f"feat:acct:{acc_other}", "sum_1h")) == 70000


def test_interruption_after_reset_and_mid_replay_recovers_cleanly(
    pg_accounts_fixture, monkeypatch, capsys
) -> None:
    """4, 5, 7, 8: Crash right after key reset and crash mid-replay both recover without duplicates."""
    engine, store, acc_target, _, created_txns = pg_accounts_fixture
    r = fakeredis.FakeRedis()
    base = datetime.now(timezone.utc) - timedelta(minutes=30)
    dev_1, dev_2 = uuid4(), uuid4()
    ip_1, ip_2 = uuid4(), uuid4()

    t1 = _make_txn(acc_target, base, "100.00", dev_1, ip_1)
    t2 = _make_txn(acc_target, base + timedelta(minutes=10), "200.00", dev_2, ip_2)
    t3 = _make_txn(acc_target, base + timedelta(minutes=20), "300.00", dev_1, ip_1)
    created_txns.extend([t1.id, t2.id, t3.id])

    for item in (t1, t3, t2):
        assert store.store(item) is True

    online_features.update(r, t1.feature_input())
    online_features.update(r, t3.feature_input())

    real_update = online_features.update
    # 1. Fail on first replay call (immediately after _reset_account_keys cleared Redis keys)
    def fail_immediately(client, txn_dict):
        raise redis.ConnectionError("Simulated crash before replay")

    monkeypatch.setattr(online_features, "update", fail_immediately)
    with pytest.raises(redis.ConnectionError):
        recover_pending_features(engine, r, account_id=acc_target)

    # All account keys were cleared and 0 transactions replayed; lock must already be released.
    assert r.exists(f"lock:acct:{acc_target}") == 0
    assert r.exists(f"feat:acct:{acc_target}") == 0

    # 2. Allow t1 to succeed, then fail on t2 (partial replay interruption)
    calls = {"n": 0}

    def fail_on_second(client, txn_dict):
        calls["n"] += 1
        if calls["n"] == 2:
            raise redis.ConnectionError("Simulated mid-replay crash")
        return real_update(client, txn_dict)

    monkeypatch.setattr(online_features, "update", fail_on_second)
    with pytest.raises(redis.ConnectionError):
        recover_pending_features(engine, r, account_id=acc_target)

    assert r.exists(f"lock:acct:{acc_target}") == 0
    assert r.exists(f"seen:txn:{t1.id}") == 1
    assert r.exists(f"seen:txn:{t2.id}") == 0
    assert int(r.hget(f"feat:acct:{acc_target}", "w_count")) == 1

    # Verify CLI exits non-zero on failure and does NOT report success on stdout
    import scripts.recover_pending_features as cli_mod

    monkeypatch.setattr(
        cli_mod,
        "recover_pending_features",
        lambda *a, **kw: (_ for _ in ()).throw(redis.ConnectionError("Redis offline")),
    )
    exit_code = cli_mod.main(["--database-url", DATABASE_URL, "--account-id", str(acc_target)])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "failed" in captured.err

    # 3. Restore normal update and complete recovery after partial replay
    monkeypatch.setattr(online_features, "update", real_update)
    resumed = recover_pending_features(engine, r, account_id=acc_target)
    assert resumed.pending_txns == 2
    assert resumed.tail_accounts == 1
    assert resumed.rebuild_accounts == 0
    assert resumed.replayed_txns == 2
    assert resumed.recovered_txns == 2

    assert int(r.hget(f"feat:acct:{acc_target}", "w_count")) == 3
    assert int(r.hget(f"feat:acct:{acc_target}", "sum_1h")) == 60000
    assert r.zcard(f"vel:acct:{acc_target}:1h") == 3
    assert r.zcard(f"dist:acct:{acc_target}:dev") == 2
    assert r.zcard(f"dist:acct:{acc_target}:ip") == 2


def test_account_lock_isolation_expiry_and_concurrent_ingestion(pg_accounts_fixture, monkeypatch) -> None:
    """6: Per-account lock blocks same-account contention cleanly, never blocks other accounts, expires safely."""
    engine, store, acc_target, acc_other, created_txns = pg_accounts_fixture
    r = fakeredis.FakeRedis()

    # A. Unrelated account is never blocked; expired lease cannot release newer owner's lock
    with account_lock(r, acc_target, ttl_ms=10_000, wait_timeout_s=1.0):
        with pytest.raises(AccountLockTimeout):
            with account_lock(r, acc_target, ttl_ms=1_000, wait_timeout_s=0.05):
                pass
        # Unrelated account acquires immediately
        with account_lock(r, acc_other, ttl_ms=1_000, wait_timeout_s=0.05):
            assert r.exists(f"lock:acct:{acc_other}") == 1

    # Expired lease safety: first owner exits after TTL elapsed; second owner retains lock
    entered_second = threading.Event()
    holder2_done = threading.Event()

    def _hold_second() -> None:
        with account_lock(r, acc_target, ttl_ms=5_000, wait_timeout_s=0.5):
            entered_second.set()
            holder2_done.wait(2.0)

    t_hold = threading.Thread(target=_hold_second)
    with account_lock(r, acc_target, ttl_ms=50, wait_timeout_s=0.5):
        time.sleep(0.08)  # First lease expires cleanly
        t_hold.start()
        assert entered_second.wait(1.0)
    try:
        # Exiting first context must NOT delete second holder's lock
        assert r.exists(f"lock:acct:{acc_target}") == 1
    finally:
        holder2_done.set()
        t_hold.join(timeout=2.0)

    # B. Concurrent live POST during rebuild on acc_target vs immediate POST on acc_other
    base = datetime.now(timezone.utc) - timedelta(minutes=25)
    dev_id, ip_id = uuid4(), uuid4()
    t1 = _make_txn(acc_target, base, "100.00", dev_id, ip_id)
    t2_mid = _make_txn(acc_target, base + timedelta(minutes=10), "200.00", dev_id, ip_id)
    t3 = _make_txn(acc_target, base + timedelta(minutes=20), "300.00", dev_id, ip_id)
    t4_live = _make_txn(acc_target, base + timedelta(minutes=22), "400.00", dev_id, ip_id)
    t_other_live = _make_txn(acc_other, base + timedelta(minutes=22), "500.00", dev_id, ip_id)
    created_txns.extend([t1.id, t2_mid.id, t3.id, t4_live.id, t_other_live.id])

    for item in (t1, t3, t2_mid):
        assert store.store(item) is True
    online_features.update(r, t1.feature_input())
    online_features.update(r, t3.feature_input())

    replay_started = threading.Event()
    allow_finish_replay = threading.Event()
    real_update = online_features.update

    def gated_update(client, txn_dict):
        if txn_dict["id"] == str(t1.id) and not replay_started.is_set():
            replay_started.set()
            assert allow_finish_replay.wait(5.0)
        return real_update(client, txn_dict)

    monkeypatch.setattr(online_features, "update", gated_update)
    deps._store = store
    deps._redis = r
    client = TestClient(app)
    headers = {"X-Service-Key": config.SERVICE_KEY}

    try:
        recovery_result = {}

        def run_rec():
            recovery_result["summary"] = recover_pending_features(
                engine, r, account_id=acc_target, lock_ttl_ms=10_000, lock_wait_timeout_s=5.0
            )

        rec_thread = threading.Thread(target=run_rec)
        rec_thread.start()
        assert replay_started.wait(3.0)

        # Unrelated account POST completes immediately while acc_target rebuild is paused!
        other_resp = client.post(
            "/api/v1/transactions",
            content=t_other_live.model_dump_json(),
            headers={"Content-Type": "application/json", **headers},
        )
        assert other_resp.status_code == 201

        # Concurrent POST on acc_target waits on lock:acct:{acc_target} while recovery completes
        live_result = {}

        def post_live():
            live_result["resp"] = client.post(
                "/api/v1/transactions",
                content=t4_live.model_dump_json(),
                headers={"Content-Type": "application/json", **headers},
            )

        live_thread = threading.Thread(target=post_live)
        live_thread.start()
        time.sleep(0.05)  # Allow t4_live to commit Postgres and block on lock:acct:{acc_target}
        allow_finish_replay.set()

        rec_thread.join(timeout=5.0)
        live_thread.join(timeout=5.0)

        assert live_result["resp"].status_code == 201
        assert int(r.hget(f"feat:acct:{acc_target}", "w_count")) == 4
        assert int(r.hget(f"feat:acct:{acc_target}", "sum_1h")) == 100000
        assert r.zcard(f"vel:acct:{acc_target}:1h") == 4
        assert r.exists(f"seen:txn:{t4_live.id}") == 1
    finally:
        deps.reset_deps()


def test_eight_day_retention_horizon_enforced(pg_accounts_fixture) -> None:
    """Retention check: transactions older than 8 days (KEY_TTL_S) are skipped by default lookback."""
    engine, store, acc_target, _, created_txns = pg_accounts_fixture
    r = fakeredis.FakeRedis()
    expired_ts = datetime.now(timezone.utc) - timedelta(days=9)
    t_old = _make_txn(acc_target, expired_ts, "99.00", uuid4(), uuid4())
    created_txns.append(t_old.id)
    assert store.store(t_old) is True

    default_scan = recover_pending_features(engine, r, account_id=acc_target)
    assert default_scan.scanned_txns == 0
    assert default_scan.pending_txns == 0

    full_scan = recover_pending_features(engine, r, account_id=acc_target, lookback_seconds=None)
    assert full_scan.scanned_txns == 1
    assert full_scan.recovered_txns == 1
    assert r.exists(f"seen:txn:{t_old.id}") == 1


def test_lock_renewal_rejects_foreign_token_and_sustains_long_rebuild(
    pg_accounts_fixture, monkeypatch
) -> None:
    """Renewal extends own active lease across >2x TTL while blocking concurrent account writes and refusing foreign locks."""
    engine, store, acc_target, acc_other, created_txns = pg_accounts_fixture
    r = fakeredis.FakeRedis()

    # 1. Foreign-token protection on renewal & cleanup
    with account_lock(r, acc_target, ttl_ms=50, wait_timeout_s=0.5) as stale_lease:
        time.sleep(0.08)  # Initial lease expires cleanly
        assert r.set(stale_lease.key, "foreign-holder-token", nx=True, px=5_000)
        ttl_before = r.pttl(stale_lease.key)
        assert stale_lease.renew(ttl_ms=60_000) is False
        assert stale_lease.lost is True
        assert r.get(stale_lease.key) == b"foreign-holder-token"
        assert r.pttl(stale_lease.key) <= max(ttl_before, 5_000)
    assert r.get(f"lock:acct:{acc_target}") == b"foreign-holder-token"
    r.delete(f"lock:acct:{acc_target}")

    # 2. Prolonged rebuild holds renewed lock (>2x TTL), keeps concurrent target POST blocked, allows unrelated account
    base = datetime.now(timezone.utc) - timedelta(minutes=25)
    dev_1, dev_2 = uuid4(), uuid4()
    ip_1, ip_2 = uuid4(), uuid4()
    t1 = _make_txn(acc_target, base, "100.00", dev_1, ip_1)
    t2_mid = _make_txn(acc_target, base + timedelta(minutes=10), "200.00", dev_2, ip_2)
    t3 = _make_txn(acc_target, base + timedelta(minutes=20), "300.00", dev_1, ip_1)
    t4_live = _make_txn(acc_target, base + timedelta(minutes=22), "400.00", dev_1, ip_1)
    t_other = _make_txn(acc_other, base + timedelta(minutes=22), "500.00", dev_1, ip_1)
    created_txns.extend([t1.id, t2_mid.id, t3.id, t4_live.id, t_other.id])

    for item in (t1, t3, t2_mid):
        assert store.store(item) is True
    online_features.update(r, t1.feature_input())
    online_features.update(r, t3.feature_input())

    replay_entered = threading.Event()
    allow_finish = threading.Event()
    real_update = online_features.update

    def prolonged_update(client, txn_dict):
        if txn_dict["id"] == str(t1.id) and not replay_entered.is_set():
            replay_entered.set()
            assert allow_finish.wait(5.0)
        return real_update(client, txn_dict)

    monkeypatch.setattr(online_features, "update", prolonged_update)
    deps._store = store
    deps._redis = r
    client = TestClient(app)
    headers = {"X-Service-Key": config.SERVICE_KEY}

    try:
        rec_out = {}

        def _run_rec() -> None:
            rec_out["summary"] = recover_pending_features(
                engine,
                r,
                account_id=acc_target,
                lock_ttl_ms=80,
                lock_wait_timeout_s=2.0,
                auto_renew=True,
                renew_interval_s=0.02,
            )

        rec_thread = threading.Thread(target=_run_rec)
        rec_thread.start()
        assert replay_entered.wait(3.0)

        # Hold well past initial 80ms lease (~2.5x TTL); heartbeat renewal must keep lock active
        time.sleep(0.20)
        assert r.exists(f"lock:acct:{acc_target}") == 1

        # Concurrent attempt on same account cannot steal lock while recovery is renewing
        with pytest.raises(AccountLockTimeout):
            with account_lock(r, acc_target, ttl_ms=100, wait_timeout_s=0.03):
                pass

        # Unrelated account ingestion completes immediately with zero blocking
        other_resp = client.post(
            "/api/v1/transactions",
            content=t_other.model_dump_json(),
            headers={"Content-Type": "application/json", **headers},
        )
        assert other_resp.status_code == 201

        live_out = {}

        def _post_target() -> None:
            live_out["resp"] = client.post(
                "/api/v1/transactions",
                content=t4_live.model_dump_json(),
                headers={"Content-Type": "application/json", **headers},
            )

        live_thread = threading.Thread(target=_post_target)
        live_thread.start()
        time.sleep(0.05)  # Allow t4_live to commit Postgres and wait on active renewed lock
        allow_finish.set()

        rec_thread.join(timeout=5.0)
        live_thread.join(timeout=5.0)

        assert rec_out["summary"].recovered_txns == 1
        assert live_out["resp"].status_code == 201
        assert r.exists(f"lock:acct:{acc_target}") == 0
        assert int(r.hget(f"feat:acct:{acc_target}", "w_count")) == 4
        assert int(r.hget(f"feat:acct:{acc_target}", "sum_1h")) == 100000
        assert r.zcard(f"vel:acct:{acc_target}:1h") == 4
        assert r.zcard(f"dist:acct:{acc_target}:dev") == 2
        assert r.zcard(f"dist:acct:{acc_target}:ip") == 2
        assert r.exists(f"seen:txn:{t4_live.id}") == 1
    finally:
        deps.reset_deps()


def test_expired_recovery_lease_fences_stale_reset_and_stale_replay(
    pg_accounts_fixture, monkeypatch
) -> None:
    """When recovery lease expires due to stall, fenced Lua aborts stale reset and stale replay without data loss."""
    engine, store, acc_target, acc_other, created_txns = pg_accounts_fixture
    r = fakeredis.FakeRedis()

    base = datetime.now(timezone.utc) - timedelta(minutes=25)
    dev_id, ip_id = uuid4(), uuid4()
    t1 = _make_txn(acc_target, base, "100.00", dev_id, ip_id)
    t2_mid = _make_txn(acc_target, base + timedelta(minutes=10), "200.00", dev_id, ip_id)
    t3 = _make_txn(acc_target, base + timedelta(minutes=20), "300.00", dev_id, ip_id)
    t4_live = _make_txn(acc_target, base + timedelta(minutes=22), "400.00", dev_id, ip_id)
    t_other = _make_txn(acc_other, base + timedelta(minutes=22), "500.00", dev_id, ip_id)
    created_txns.extend([t1.id, t2_mid.id, t3.id, t4_live.id, t_other.id])

    for item in (t1, t3, t2_mid):
        assert store.store(item) is True
    online_features.update(r, t1.feature_input())
    online_features.update(r, t3.feature_input())

    # Subcase A: Stall BEFORE _reset_account_keys crosses expiry boundary (auto_renew=False simulates frozen host)
    orig_reset = recovery_mod._reset_account_keys
    gate_reset = threading.Event()
    resume_reset = threading.Event()

    def paused_reset(client, acct, ids, **kwargs):
        gate_reset.set()
        assert resume_reset.wait(5.0)
        return orig_reset(client, acct, ids, **kwargs)

    monkeypatch.setattr(recovery_mod, "_reset_account_keys", paused_reset)
    deps._store = store
    deps._redis = r
    client = TestClient(app)
    headers = {"X-Service-Key": config.SERVICE_KEY}

    try:
        stale_err = {}

        def _stale_rec() -> None:
            try:
                recover_pending_features(
                    engine,
                    r,
                    account_id=acc_target,
                    lock_ttl_ms=50,
                    lock_wait_timeout_s=1.0,
                    auto_renew=False,
                )
            except Exception as exc:
                stale_err["exc"] = exc

        th_a = threading.Thread(target=_stale_rec)
        th_a.start()
        assert gate_reset.wait(3.0)
        time.sleep(0.08)  # Force 50ms lease expiry

        # Lock has expired cleanly; concurrent ingestion for acc_target AND acc_other succeed
        assert r.exists(f"lock:acct:{acc_target}") == 0
        other_resp = client.post(
            "/api/v1/transactions",
            content=t_other.model_dump_json(),
            headers={"Content-Type": "application/json", **headers},
        )
        assert other_resp.status_code == 201
        target_resp = client.post(
            "/api/v1/transactions",
            content=t4_live.model_dump_json(),
            headers={"Content-Type": "application/json", **headers},
        )
        assert target_resp.status_code == 201
        assert int(r.hget(f"feat:acct:{acc_target}", "sum_1h")) == 80000  # t1+t3+t4

        # Unblock stale recovery -> must raise AccountLockLostError and NOT wipe t4_live
        resume_reset.set()
        th_a.join(timeout=5.0)
        assert isinstance(stale_err.get("exc"), AccountLockLostError)
        assert r.exists(f"seen:txn:{t4_live.id}") == 1
        assert int(r.hget(f"feat:acct:{acc_target}", "sum_1h")) == 80000

        # Subcase B: Stall MID-REPLAY (after reset & t1) until lease expires; Lua fence rejects t2_mid
        monkeypatch.setattr(recovery_mod, "_reset_account_keys", orig_reset)
        real_update = online_features.update
        gate_mid = threading.Event()
        resume_mid = threading.Event()
        seen_t1_replay = {"done": False}

        def paused_after_t1(client_arg, txn_dict):
            res = real_update(client_arg, txn_dict)
            if txn_dict["id"] == str(t1.id) and not seen_t1_replay["done"]:
                seen_t1_replay["done"] = True
                gate_mid.set()
                assert resume_mid.wait(5.0)
            return res

        monkeypatch.setattr(online_features, "update", paused_after_t1)
        stale_mid_err = {}

        def _stale_mid_rec() -> None:
            try:
                recover_pending_features(
                    engine,
                    r,
                    account_id=acc_target,
                    lock_ttl_ms=50,
                    lock_wait_timeout_s=1.0,
                    auto_renew=False,
                )
            except Exception as exc:
                stale_mid_err["exc"] = exc

        th_b = threading.Thread(target=_stale_mid_rec)
        th_b.start()
        assert gate_mid.wait(3.0)
        time.sleep(0.08)  # Force 50ms expiry while paused after t1

        # Simulate another holder taking ownership of the lock while paused
        assert r.set(f"lock:acct:{acc_target}", "new-owner-token", nx=True, px=5_000)
        resume_mid.set()
        th_b.join(timeout=5.0)
        assert isinstance(stale_mid_err.get("exc"), AccountLockLostError)
        # Foreign holder's lock was preserved by token-verified cleanup
        assert r.get(f"lock:acct:{acc_target}") == b"new-owner-token"
        r.delete(f"lock:acct:{acc_target}")

        # Final clean recovery converges exact full state for t1, t2_mid, t3, t4_live (no duplicates)
        monkeypatch.setattr(online_features, "update", real_update)
        final_rec = recover_pending_features(engine, r, account_id=acc_target)
        assert final_rec.pending_txns == 3  # t2_mid, t3, t4_live were un-replayed after subcase B abort
        assert int(r.hget(f"feat:acct:{acc_target}", "w_count")) == 4
        assert int(r.hget(f"feat:acct:{acc_target}", "sum_1h")) == 100000
        assert r.zcard(f"vel:acct:{acc_target}:1h") == 4
        assert r.exists(f"seen:txn:{t1.id}") == 1
        assert r.exists(f"seen:txn:{t2_mid.id}") == 1
        assert r.exists(f"seen:txn:{t3.id}") == 1
        assert r.exists(f"seen:txn:{t4_live.id}") == 1
        assert int(r.hget(f"feat:acct:{acc_other}", "w_count")) == 1
    finally:
        deps.reset_deps()
