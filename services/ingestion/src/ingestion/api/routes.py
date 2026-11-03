"""FastAPI routes for AegisFlow transaction ingestion, health, and observability."""

from __future__ import annotations

import secrets
import time
from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from loguru import logger
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Histogram,
    generate_latest,
)
import redis

from ingestion import config, deps, online_features
from ingestion.persistence import (
    UnknownEntity,
    DuplicateExternalRefError,
    LegacyReconstructionError,
)
from ingestion.schemas import TransactionIn, TransactionOut, compare_canonical_fields

router = APIRouter()

# ============================================================================
# Prometheus Metrics (Idempotent Registry Registration)
# ============================================================================

try:
    TRANSACTIONS_TOTAL = Counter(
        "af_transactions_total",
        "Total incoming transactions processed by ingestion endpoint",
        ["result"],
    )
except ValueError:
    TRANSACTIONS_TOTAL = REGISTRY._names_to_collectors["af_transactions_total"]

try:
    ONLINE_FEATURE_SECONDS = Histogram(
        "af_online_feature_seconds",
        "Latency of online feature computation via Redis Lua in seconds",
    )
except ValueError:
    ONLINE_FEATURE_SECONDS = REGISTRY._names_to_collectors["af_online_feature_seconds"]


# ============================================================================
# Security / Authentication Dependency
# ============================================================================


def verify_service_key(
    x_service_key: str | None = Header(None, alias="X-Service-Key"),
) -> None:
    """Validate X-Service-Key header using constant-time string comparison."""
    if not x_service_key or not secrets.compare_digest(x_service_key, config.SERVICE_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Service-Key header",
        )


# ============================================================================
# Ingestion Endpoint
# ============================================================================


@router.post(
    "/api/v1/transactions",
    dependencies=[Depends(verify_service_key)],
    response_model=TransactionOut,
)
def ingest_transaction(
    t: TransactionIn,
    response: Response,
) -> TransactionOut:
    """Ingest a transaction following the exact lifecycle:

    1. Validate payload (Pydantic).
    2. Commit transaction row and entity edges atomically in PostgreSQL.
    3. Execute Redis Lua online feature calculation.
    4. Return HTTP 201 (new) or 200 (duplicate), or HTTP 503 if Redis fails.
    """
    # 1. Validation is performed automatically by FastAPI before handler execution

    # 2. Persist in database
    store = deps.store()
    try:
        inserted = store.store(t)
    except UnknownEntity:
        TRANSACTIONS_TOTAL.labels(result="unknown_entity").inc()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unknown entity reference: transaction references a non-existent account or counterparty.",
        )
    except DuplicateExternalRefError as exc:
        logger.warning(
            "External reference conflict for transaction {}: exception={}, error_code={}",
            t.id,
            type(exc).__name__,
            exc.error_code,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Transaction external reference already exists with a different transaction ID",
        )

    # If duplicate ID, retrieve existing transaction and verify payload integrity
    if not inserted:
        try:
            existing = store.get(t.id)
        except LegacyReconstructionError as exc:
            logger.warning(
                "Legacy reconstruction failure for transaction {}: field={}, exception={}, error_code={}",
                t.id,
                exc.field,
                type(exc).__name__,
                exc.error_code,
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "Transaction ID already exists but was recorded with a legacy schema "
                    f"incompatible with the current API version (field: {exc.field}). "
                    "Manual review required."
                ),
            )

        if existing is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Transaction conflict detected but record could not be retrieved",
            )
        if not compare_canonical_fields(existing, t):
            # Conflicting payload with existing transaction ID: reject with 409
            # Do NOT update Redis, do NOT modify store, do NOT increment created/duplicate
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Transaction ID already exists with conflicting payload",
            )

    # 3. Update online feature state in Redis
    r = deps.redis_client()
    try:
        t0 = time.perf_counter()
        online_features.update(r, t.feature_input())
        ONLINE_FEATURE_SECONDS.observe(time.perf_counter() - t0)
    except (redis.RedisError, Exception) as exc:
        logger.error(f"Redis feature update failed for transaction {t.id}: {exc}")
        TRANSACTIONS_TOTAL.labels(result="features_pending").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Transaction stored; online features pending. Please retry.",
        )

    # 4. Return appropriate response
    if inserted:
        TRANSACTIONS_TOTAL.labels(result="created").inc()
        response.status_code = status.HTTP_201_CREATED
        return TransactionOut(id=t.id, decision=None, duplicate=False)
    else:
        TRANSACTIONS_TOTAL.labels(result="duplicate").inc()
        response.status_code = status.HTTP_200_OK
        return TransactionOut(id=t.id, decision=None, duplicate=True)


# ============================================================================
# Observability & Health Endpoints
# ============================================================================


@router.get("/health")
def health() -> dict[str, str]:
    """Lightweight liveness probe."""
    return {"status": "ok"}


@router.get("/ready")
def readiness(response: Response) -> dict[str, Any]:
    """Independent readiness probe verifying PostgreSQL and Redis reachability."""
    pg_ok = False
    try:
        pg_ok = bool(deps.store().ping())
    except Exception:
        pg_ok = False

    redis_ok = False
    try:
        redis_ok = bool(deps.redis_client().ping())
    except Exception:
        redis_ok = False

    if not (pg_ok and redis_ok):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "not_ready",
            "postgres": "ok" if pg_ok else "unavailable",
            "redis": "ok" if redis_ok else "unavailable",
        }

    return {"status": "ready"}


@router.get("/metrics")
def metrics() -> Response:
    """Prometheus exposition metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
