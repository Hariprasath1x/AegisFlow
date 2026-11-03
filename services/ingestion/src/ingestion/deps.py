"""Process-level dependency singletons for AegisFlow ingestion service."""

from __future__ import annotations

from sqlalchemy import create_engine
import redis

from ingestion import config
from ingestion.persistence import PgStore, Store

_store: Store | None = None
_redis: redis.Redis | None = None


def store() -> Store:
    """Return the process-level Store singleton, initializing it if None."""
    global _store
    if _store is None:
        engine = create_engine(
            config.DATABASE_URL,
            pool_pre_ping=True,
        )
        _store = PgStore(engine)
    return _store


def redis_client() -> redis.Redis:
    """Return the process-level Redis client singleton, initializing it if None."""
    global _redis
    if _redis is None:
        _redis = redis.Redis.from_url(
            config.REDIS_URL,
            decode_responses=False,
        )
    return _redis


def reset_deps() -> None:
    """Reset dependency singletons to None (for clean test isolation)."""
    global _store, _redis
    _store = None
    _redis = None
