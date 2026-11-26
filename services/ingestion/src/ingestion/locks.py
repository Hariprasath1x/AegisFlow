"""Per-account Redis distributed lease lock with token-verified renewal and fencing support."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import threading
import time
from typing import Any, Iterator
from uuid import UUID, uuid4

from loguru import logger

_RELEASE_LOCK_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""

_RENEW_LOCK_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[2]))
end
return 0
"""


class AccountLockTimeout(TimeoutError):
    """Raised when an account-scoped Redis lock cannot be acquired within timeout."""


class AccountLockLostError(RuntimeError):
    """Raised when an operation's Redis lease expired or is owned by another holder."""


def _token_matches(current: Any, token: str) -> bool:
    return current in (token, token.encode("utf-8"))


def _release_compare_and_delete(r: Any, lock_key: str, token: str) -> None:
    """Release `lock_key` only if its stored value matches `token`."""
    try:
        r.eval(_RELEASE_LOCK_LUA, 1, lock_key, token)
        return
    except Exception as lua_exc:
        # Fallback compare-and-delete via WATCH/MULTI for scripted mocks or non-Lua paths.
        try:
            with r.pipeline() as pipe:
                pipe.watch(lock_key)
                current = pipe.get(lock_key)
                if _token_matches(current, token):
                    pipe.multi()
                    pipe.delete(lock_key)
                    pipe.execute()
                else:
                    pipe.unwatch()
            return
        except Exception:
            logger.warning(
                f"Failed releasing {lock_key} (will expire via TTL): {lua_exc}"
            )


def _renew_compare_and_expire(r: Any, lock_key: str, token: str, ttl_ms: int) -> bool:
    """Extend `lock_key` TTL only if its stored value still matches `token`."""
    try:
        res = r.eval(_RENEW_LOCK_LUA, 1, lock_key, token, int(ttl_ms))
        return bool(int(res)) if isinstance(res, (int, bytes, str)) else bool(res)
    except Exception:
        with r.pipeline() as pipe:
            pipe.watch(lock_key)
            current = pipe.get(lock_key)
            if not _token_matches(current, token):
                pipe.unwatch()
                return False
            pipe.multi()
            pipe.pexpire(lock_key, int(ttl_ms))
            res = pipe.execute()
            return bool(res and res[0])


@dataclass
class AccountLease:
    """Handle representing an active per-account lease token in Redis."""

    redis_client: Any
    key: str
    token: str
    ttl_ms: int
    lost: bool = False
    renew_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def renew(self, ttl_ms: int | None = None) -> bool:
        """Renew this lease's TTL if and only if Redis still holds our token."""
        target_ttl = int(self.ttl_ms if ttl_ms is None else ttl_ms)
        with self._lock:
            if self.lost:
                return False
            try:
                ok = _renew_compare_and_expire(self.redis_client, self.key, self.token, target_ttl)
            except Exception:
                ok = False
            if not ok:
                self.lost = True
                return False
            self.renew_count += 1
            return True

    def is_held(self) -> bool:
        """Return True iff this lease is not marked lost and Redis holds our token."""
        with self._lock:
            if self.lost:
                return False
            current = self.redis_client.get(self.key)
            if not _token_matches(current, self.token):
                self.lost = True
                return False
            return True

    def assert_held(self) -> None:
        """Raise `AccountLockLostError` if this lease expired or was overtaken."""
        if not self.is_held():
            raise AccountLockLostError(
                f"Account lock lease lost for {self.key} (expired or acquired by another holder)"
            )


@contextmanager
def account_lock(
    r: Any,
    account_id: UUID | str,
    *,
    ttl_ms: int = 10_000,
    wait_timeout_s: float = 3.0,
    poll_interval_s: float = 0.01,
    auto_renew: bool = False,
    renew_interval_s: float | None = None,
) -> Iterator[AccountLease]:
    """Acquire an expiring per-account lease lock in Redis without blocking other accounts.

    Uses atomic `SET key token NX PX ttl_ms` to acquire, optional token-verified
    background lease renewal while work is active, and token-checked deletion on exit.
    """
    lock_key = f"lock:acct:{account_id}"
    token = uuid4().hex
    deadline = time.monotonic() + max(0.0, wait_timeout_s)

    while True:
        acquired = r.set(lock_key, token, nx=True, px=ttl_ms)
        if acquired:
            break
        if time.monotonic() >= deadline:
            raise AccountLockTimeout(
                f"Timed out after {wait_timeout_s:.2f}s waiting for {lock_key}"
            )
        time.sleep(poll_interval_s)

    lease = AccountLease(redis_client=r, key=lock_key, token=token, ttl_ms=ttl_ms)
    stop_event = threading.Event()
    renew_thread: threading.Thread | None = None

    if auto_renew and ttl_ms > 0:
        interval = (
            renew_interval_s
            if renew_interval_s is not None
            else max(0.01, (ttl_ms / 1000.0) / 3.0)
        )

        def _heartbeat() -> None:
            while not stop_event.wait(interval):
                if not lease.renew():
                    break

        renew_thread = threading.Thread(
            target=_heartbeat,
            name=f"lock-renew-{account_id}",
            daemon=True,
        )
        renew_thread.start()

    try:
        yield lease
    finally:
        stop_event.set()
        if renew_thread is not None:
            renew_thread.join(timeout=1.0)
        _release_compare_and_delete(r, lock_key, token)
