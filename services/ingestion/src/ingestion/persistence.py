"""PostgreSQL and in-memory persistence layer for transaction ingestion and entity edges."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Protocol, Sequence
from uuid import UUID
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from ingestion.schemas import TransactionIn


_M2_CHANNELS: frozenset[str] = frozenset({"card", "transfer", "wallet"})


class UnknownEntity(Exception):
    """Raised when a foreign-key constraint is violated (e.g. unknown account reference)."""


class LegacyReconstructionError(Exception):
    """Base exception for stored legacy transactions that cannot be reconstructed as M2 TransactionIn."""

    def __init__(self, field: str, error_code: str = "ERR_LEGACY_RECONSTRUCTION") -> None:
        self.field = field
        self.error_code = error_code
        super().__init__(f"Legacy reconstruction failed for field={field!r} (code={error_code})")


class LegacyTransactionError(LegacyReconstructionError):
    """Raised when a stored channel or currency cannot be represented in M2 TransactionIn schema."""

    def __init__(self, field: str, error_code: str = "ERR_LEGACY_RECONSTRUCTION") -> None:
        super().__init__(field=field, error_code=error_code)


class LegacyDeviceIdError(LegacyReconstructionError):
    """Raised when a stored device_id cannot be parsed as a UUID. Never stores or logs raw device ID strings."""

    def __init__(self, error_code: str = "ERR_LEGACY_DEVICE_ID") -> None:
        super().__init__(field="device_id", error_code=error_code)


class DuplicateExternalRefError(Exception):
    """Raised when an insert violates the unique constraint on external_ref (SQLSTATE 23505) with a differing txn_id."""

    def __init__(self, error_code: str = "ERR_EXTERNAL_REF_CONFLICT") -> None:
        self.error_code = error_code
        super().__init__(f"External reference conflict ({error_code})")


def _safe_device_uuid(raw: str | None) -> UUID | None:
    """Convert a stored device_id string to UUID.

    Returns UUID if valid, None if None or empty.
    Raises LegacyDeviceIdError if non-empty and non-UUID.
    The exception is raised outside the except block to decouple __cause__ and __context__.
    """
    if raw is None or raw == "":
        return None

    parsed: UUID | None = None
    try:
        parsed = UUID(raw)
    except (ValueError, AttributeError):
        pass

    if parsed is None:
        raise LegacyDeviceIdError(error_code="ERR_LEGACY_DEVICE_ID")

    return parsed


def edges_for(t: TransactionIn) -> list[tuple[str, str, str, str, str]]:
    """Derive entity edges for transaction t: (src_type, src_id, dst_type, dst_id, edge_type)."""
    edges: list[tuple[str, str, str, str, str]] = []
    acc_id = str(t.account_id)

    if t.device_id is not None:
        edges.append(("account", acc_id, "device", str(t.device_id), "uses_device"))

    if t.ip_id is not None:
        edges.append(("account", acc_id, "ip", str(t.ip_id), "uses_ip"))

    if t.merchant_id is not None:
        edges.append(("account", acc_id, "merchant", str(t.merchant_id), "pays_merchant"))

    if t.counterparty_id is not None and t.channel == "transfer":
        edges.append(("account", acc_id, "account", str(t.counterparty_id), "transfers_to"))

    return edges


class Store(Protocol):
    """Protocol defining transaction persistence interface."""

    def store(self, t: TransactionIn) -> bool:
        """Persist transaction and entity edges. Returns True if newly inserted, False if duplicate."""
        ...

    def get(self, txn_id: UUID) -> TransactionIn | None:
        """Retrieve existing persisted transaction by ID."""
        ...

    def ping(self) -> bool:
        """Check whether the underlying datastore is healthy and reachable."""
        ...


class PgStore:
    """PostgreSQL implementation of transaction and entity-edge persistence."""

    def __init__(self, engine: Engine | str):
        if isinstance(engine, str):
            self.engine = create_engine(engine, pool_pre_ping=True)
        else:
            self.engine = engine
        self._id_col, self._ts_col, self._has_m1_cols, self._has_raw_payload, self._has_dest_account = self._detect_columns()

    def _detect_columns(self) -> tuple[str, str, bool, bool, bool]:
        """Detect whether transactions table uses M2 guide columns or M1 schema columns."""
        try:
            inspector = inspect(self.engine)
            cols = {c["name"] for c in inspector.get_columns("transactions")}
        except Exception:
            cols = set()

        has_raw_payload = "raw_payload" in cols
        has_dest_account = "destination_account_id" in cols
        if "id" in cols:
            return "id", "ts", False, has_raw_payload, has_dest_account
        return "txn_id", "occurred_at", True, has_raw_payload, has_dest_account

    def ping(self) -> bool:
        """Execute simple probe query to verify database health."""
        try:
            with self.engine.connect() as conn:
                res = conn.execute(text("SELECT 1")).scalar()
                return res == 1
        except Exception:
            return False

    def get(self, txn_id: UUID) -> TransactionIn | None:
        """Retrieve existing persisted transaction by ID.

        Returns canonical TransactionIn if found, or None.
        Raises LegacyReconstructionError subclasses if stored data is incompatible with M2.
        """
        with self.engine.connect() as conn:
            if not self._has_m1_cols:
                stmt = text("""
                    SELECT id, ts, account_id, counterparty_id, merchant_id,
                           device_id, ip_id, amount, currency, channel
                    FROM transactions
                    WHERE id = :id;
                """)
                row = conn.execute(stmt, {"id": txn_id}).mappings().one_or_none()
                if row is None:
                    return None
                occurred = row["ts"] if row["ts"].tzinfo else row["ts"].replace(tzinfo=timezone.utc)
                raw_currency = (row["currency"] or "").strip()
                if not re.fullmatch(r"^[A-Z]{3}$", raw_currency):
                    raise LegacyTransactionError(field="currency", error_code="ERR_LEGACY_CURRENCY")
                stored_channel = row["channel"]
                if stored_channel not in _M2_CHANNELS:
                    raise LegacyTransactionError(field="channel", error_code="ERR_LEGACY_CHANNEL")
                safe_device = _safe_device_uuid(str(row["device_id"]) if row["device_id"] else None)
                return TransactionIn(
                    id=row["id"],
                    ts=occurred,
                    account_id=row["account_id"],
                    counterparty_id=row["counterparty_id"],
                    merchant_id=row["merchant_id"],
                    device_id=safe_device,
                    ip_id=row["ip_id"],
                    amount=row["amount"],
                    currency=raw_currency,
                    channel=stored_channel,
                )
            else:
                if self._has_dest_account:
                    stmt = text("""
                        SELECT txn_id, occurred_at, account_id, destination_account_id, counterparty_id,
                               merchant_id, device_id, ip_id, amount, currency, channel
                        FROM transactions
                        WHERE txn_id = :id;
                    """)
                    row = conn.execute(stmt, {"id": txn_id}).mappings().one_or_none()
                    if row is None:
                        return None

                    stored_channel = row["channel"]
                    if stored_channel not in _M2_CHANNELS:
                        raise LegacyTransactionError(field="channel", error_code="ERR_LEGACY_CHANNEL")

                    safe_device = _safe_device_uuid(row["device_id"])

                    occurred = row["occurred_at"]
                    if occurred.tzinfo is None:
                        occurred = occurred.replace(tzinfo=timezone.utc)

                    raw_currency = (row["currency"] or "").strip()
                    if not re.fullmatch(r"^[A-Z]{3}$", raw_currency):
                        raise LegacyTransactionError(field="currency", error_code="ERR_LEGACY_CURRENCY")

                    wire_counterparty = (
                        row["destination_account_id"] if stored_channel == "transfer"
                        else row["counterparty_id"]
                    )

                    return TransactionIn(
                        id=row["txn_id"],
                        ts=occurred,
                        account_id=row["account_id"],
                        counterparty_id=wire_counterparty,
                        merchant_id=row["merchant_id"],
                        device_id=safe_device,
                        ip_id=row["ip_id"],
                        amount=row["amount"],
                        currency=raw_currency,
                        channel=stored_channel,
                    )
                else:
                    stmt = text("""
                        SELECT txn_id, occurred_at, account_id, counterparty_id,
                               device_id, amount, currency, channel
                        FROM transactions
                        WHERE txn_id = :id;
                    """)
                    row = conn.execute(stmt, {"id": txn_id}).mappings().one_or_none()
                    if row is None:
                        return None

                    stored_channel = row["channel"]
                    if stored_channel not in _M2_CHANNELS:
                        raise LegacyTransactionError(field="channel", error_code="ERR_LEGACY_CHANNEL")

                    safe_device = _safe_device_uuid(row["device_id"])

                    occurred = row["occurred_at"]
                    if occurred.tzinfo is None:
                        occurred = occurred.replace(tzinfo=timezone.utc)

                    raw_currency = (row["currency"] or "").strip()
                    if not re.fullmatch(r"^[A-Z]{3}$", raw_currency):
                        raise LegacyTransactionError(field="currency", error_code="ERR_LEGACY_CURRENCY")

                    return TransactionIn(
                        id=row["txn_id"],
                        ts=occurred,
                        account_id=row["account_id"],
                        counterparty_id=row["counterparty_id"],
                        merchant_id=None,
                        device_id=safe_device,
                        ip_id=None,
                        amount=row["amount"],
                        currency=raw_currency,
                        channel=stored_channel,
                    )

    def store(self, t: TransactionIn) -> bool:
        """Atomically insert transaction row and upsert entity edges in one DB transaction."""
        edge_list = edges_for(t)

        if t.channel == "transfer":
            dest_acc_id = t.counterparty_id
            m1_cp_id = None
        else:
            dest_acc_id = None
            m1_cp_id = t.counterparty_id

        if not self._has_m1_cols:
            stmt = text("""
                INSERT INTO transactions (
                    id, ts, account_id, counterparty_id, merchant_id,
                    device_id, ip_id, amount, currency, channel
                ) VALUES (
                    :id, :ts, :account_id, :counterparty_id, :merchant_id,
                    :device_id, :ip_id, :amount, :currency, :channel
                )
                ON CONFLICT (id) DO NOTHING
                RETURNING id;
            """)
            params = {
                "id": t.id,
                "ts": t.ts,
                "account_id": t.account_id,
                "counterparty_id": t.counterparty_id,
                "merchant_id": t.merchant_id,
                "device_id": t.device_id,
                "ip_id": t.ip_id,
                "amount": t.amount,
                "currency": t.currency,
                "channel": t.channel,
            }
        else:
            if self._has_dest_account:
                stmt = text("""
                    INSERT INTO transactions (
                        txn_id, occurred_at, account_id, destination_account_id, counterparty_id,
                        merchant_id, device_id, ip_id, amount, currency, channel, direction,
                        external_ref, raw_payload
                    ) VALUES (
                        :id, :ts, :account_id, :dest_acc_id, :counterparty_id,
                        :merchant_id, :device_id, :ip_id, :amount, :currency, :channel, NULL,
                        :external_ref, CAST(:raw_payload AS jsonb)
                    )
                    ON CONFLICT (txn_id) DO NOTHING
                    RETURNING txn_id;
                """)
                params = {
                    "id": t.id,
                    "ts": t.ts,
                    "account_id": t.account_id,
                    "dest_acc_id": dest_acc_id,
                    "counterparty_id": m1_cp_id,
                    "merchant_id": t.merchant_id,
                    "device_id": str(t.device_id) if t.device_id else None,
                    "ip_id": t.ip_id,
                    "amount": t.amount,
                    "currency": t.currency,
                    "channel": t.channel,
                    "external_ref": str(t.id),
                    "raw_payload": t.model_dump_json(),
                }
            else:
                stmt = text("""
                    INSERT INTO transactions (
                        txn_id, occurred_at, account_id, counterparty_id,
                        device_id, amount, currency, channel, direction, external_ref, raw_payload
                    ) VALUES (
                        :id, :ts, :account_id, :counterparty_id,
                        :device_id, :amount, :currency, :channel, 'debit', :external_ref, CAST(:raw_payload AS jsonb)
                    )
                    ON CONFLICT (txn_id) DO NOTHING
                    RETURNING txn_id;
                """)
                params = {
                    "id": t.id,
                    "ts": t.ts,
                    "account_id": t.account_id,
                    "counterparty_id": t.counterparty_id,
                    "device_id": str(t.device_id) if t.device_id else None,
                    "amount": t.amount,
                    "currency": t.currency,
                    "channel": t.channel,
                    "external_ref": str(t.id),
                    "raw_payload": t.model_dump_json(),
                }

        try:
            with self.engine.begin() as conn:
                res = conn.execute(stmt, params).scalar_one_or_none()
                inserted = res is not None

                # Critical D6 requirement: Upsert edges ONLY if transaction was newly inserted
                if inserted and edge_list:
                    upsert_edge_stmt = text("""
                        INSERT INTO entity_edges (
                            src_type, src_id, dst_type, dst_id, edge_type,
                            weight, first_seen_at, last_seen_at, properties
                        ) VALUES (
                            :src_type, :src_id, :dst_type, :dst_id, :edge_type,
                            1.0, :ts, :ts, '{}'::jsonb
                        )
                        ON CONFLICT (src_type, src_id, dst_type, dst_id, edge_type)
                        DO UPDATE SET
                            weight = entity_edges.weight + 1.0,
                            last_seen_at = GREATEST(entity_edges.last_seen_at, EXCLUDED.last_seen_at);
                    """)
                    for st, si, dt, di, et in edge_list:
                        conn.execute(
                            upsert_edge_stmt,
                            {
                                "src_type": st,
                                "src_id": si,
                                "dst_type": dt,
                                "dst_id": di,
                                "edge_type": et,
                                "ts": t.ts,
                            },
                        )

                return inserted

        except IntegrityError as exc:
            orig = getattr(exc, "orig", None)
            sqlstate = getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
            diag = getattr(orig, "diag", None)
            constraint_name = getattr(diag, "constraint_name", None)

            # 1. Foreign key violation (SQLSTATE 23503) — classified by SQLSTATE only,
            #    never by str(exc) message content which could expose raw database text.
            if sqlstate == "23503":
                raise UnknownEntity("Foreign key constraint violated.") from exc

            # 2. Section 5 requirement: Diagnostic-only constraint identification for external_ref
            if sqlstate == "23505":
                if constraint_name == "transactions_external_ref_key":
                    raise DuplicateExternalRefError(error_code="ERR_EXTERNAL_REF_CONFLICT")
                # Unrelated unique constraint or absent diagnostic: fail closed and re-raise IntegrityError
                raise

            # Re-raise any other unexpected integrity errors (e.g. check constraints)
            raise


class MemoryStore:
    """In-memory test double for transaction and entity-edge persistence."""

    def __init__(self, known_accounts: Sequence[UUID | str] | None = None):
        self.known_accounts: set[str] = (
            {str(a) for a in known_accounts} if known_accounts is not None else set()
        )
        self.transactions: dict[UUID, TransactionIn] = {}
        self.edges: dict[tuple[str, str, str, str, str], float] = {}
        self.edge_timestamps: dict[tuple[str, str, str, str, str], tuple[datetime, datetime]] = {}
        self._healthy: bool = True

    def set_healthy(self, healthy: bool) -> None:
        """Control simulated health for readiness testing."""
        self._healthy = healthy

    def ping(self) -> bool:
        """Simulated readiness check."""
        return self._healthy

    def get(self, txn_id: UUID) -> TransactionIn | None:
        """Retrieve existing persisted transaction by ID."""
        return self.transactions.get(txn_id)

    def store(self, t: TransactionIn) -> bool:
        """Persist transaction and entity edges in memory.

        Raises UnknownEntity if account_id or counterparty_id is not in known_accounts.
        """
        # Validate known entities if known_accounts was configured
        if self.known_accounts:
            if str(t.account_id) not in self.known_accounts:
                raise UnknownEntity(f"Unknown account: {t.account_id}")
            if t.counterparty_id and str(t.counterparty_id) not in self.known_accounts:
                raise UnknownEntity(f"Unknown counterparty: {t.counterparty_id}")

        if t.id in self.transactions:
            return False

        # Mark transaction inserted
        self.transactions[t.id] = t

        # Upsert entity edges only on initial insert
        for edge in edges_for(t):
            if edge in self.edges:
                self.edges[edge] += 1.0
                first_seen, _ = self.edge_timestamps[edge]
                self.edge_timestamps[edge] = (first_seen, t.ts)
            else:
                self.edges[edge] = 1.0
                self.edge_timestamps[edge] = (t.ts, t.ts)

        return True
