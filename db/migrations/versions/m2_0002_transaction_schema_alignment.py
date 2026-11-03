"""M2 Transaction Schema Alignment

Revision ID: m2_0002
Revises: m2_0001
Create Date: 2026-11-03 16:00:00

Schema changes to transactions table:
- Conditionally align occurred_at to TIMESTAMPTZ based on pg_type catalog inspection
- Add destination_account_id UUID NULL REFERENCES accounts(account_id)
- Add merchant_id UUID NULL
- Add ip_id UUID NULL
- Expand channel check constraint to ('card', 'wire', 'upi', 'ach', 'atm', 'transfer', 'wallet')
- Make direction nullable and update check constraint to (direction IS NULL OR direction IN ('debit', 'credit'))
- Add check constraint ck_transactions_transfer_target enforcing transfer destination rules
- Add partial indexes on destination_account_id, merchant_id, and ip_id
- Sequential downgrade guards preventing silent data loss
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'm2_0002'
down_revision: Union[str, None] = 'm2_0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()

    # 1. Authoritative catalog inspection of occurred_at using pg_type.typname
    col_info = bind.execute(sa.text("""
        SELECT t.typname AS type_name
        FROM pg_attribute a
        JOIN pg_class c ON a.attrelid = c.oid
        JOIN pg_type t ON a.atttypid = t.oid
        JOIN pg_namespace n ON c.relnamespace = n.oid
        WHERE n.nspname = 'public'
          AND c.relname = 'transactions'
          AND a.attname = 'occurred_at';
    """)).mappings().one_or_none()

    if col_info is None:
        raise RuntimeError("Migration error: column transactions.occurred_at not found")

    current_typname = col_info["type_name"].lower()

    if current_typname == "timestamptz":
        # Already canonical TIMESTAMPTZ (matches scripts/init_db.sql and live database).
        # Strict no-op: do not execute 'USING occurred_at AT TIME ZONE UTC' which would shift instants.
        pass
    elif current_typname == "timestamp":
        # Legacy naive column from custom deployment; convert to TIMESTAMPTZ deterministically.
        op.execute("""
            ALTER TABLE transactions
            ALTER COLUMN occurred_at TYPE TIMESTAMPTZ
            USING occurred_at AT TIME ZONE 'UTC';
        """)
    else:
        raise RuntimeError(
            f"Migration aborted: unexpected data type for transactions.occurred_at: {current_typname!r}. "
            "Expected 'timestamptz' or 'timestamp'."
        )

    # 2. Add M2 columns
    op.add_column(
        'transactions',
        sa.Column('destination_account_id', sa.UUID(), nullable=True),
    )
    op.create_foreign_key(
        'fk_transactions_destination_account',
        'transactions',
        'accounts',
        ['destination_account_id'],
        ['account_id'],
    )

    op.add_column(
        'transactions',
        sa.Column('merchant_id', sa.UUID(), nullable=True),
    )
    op.add_column(
        'transactions',
        sa.Column('ip_id', sa.UUID(), nullable=True),
    )

    # 3. Expand channel constraint
    op.drop_constraint('transactions_channel_check', 'transactions', type_='check')
    op.create_check_constraint(
        'ck_transactions_channel',
        'transactions',
        "channel IN ('card', 'wire', 'upi', 'ach', 'atm', 'transfer', 'wallet')",
    )

    # 4. Direction nullability & constraint
    op.alter_column('transactions', 'direction', existing_type=sa.VARCHAR(6), nullable=True)
    op.drop_constraint('transactions_direction_check', 'transactions', type_='check')
    op.create_check_constraint(
        'ck_transactions_direction',
        'transactions',
        "direction IS NULL OR direction IN ('debit', 'credit')",
    )

    # 5. Transfer target consistency constraint
    op.create_check_constraint(
        'ck_transactions_transfer_target',
        'transactions',
        """(channel = 'transfer' AND destination_account_id IS NOT NULL AND counterparty_id IS NULL)
           OR
           (channel != 'transfer' AND destination_account_id IS NULL)""",
    )

    # 6. Partial indexes for M2 entity routing
    op.create_index(
        'idx_txn_dest_account',
        'transactions',
        ['destination_account_id'],
        postgresql_where=sa.text('destination_account_id IS NOT NULL'),
    )
    op.create_index(
        'idx_txn_merchant_id',
        'transactions',
        ['merchant_id'],
        postgresql_where=sa.text('merchant_id IS NOT NULL'),
    )
    op.create_index(
        'idx_txn_ip_id',
        'transactions',
        ['ip_id'],
        postgresql_where=sa.text('ip_id IS NOT NULL'),
    )


def downgrade() -> None:
    bind = op.get_bind()

    # Sequential Downgrade Guards (first failure aborts immediately)

    # Guard 1: M2-only channels
    m2_channels = bind.execute(sa.text(
        "SELECT COUNT(*) FROM transactions WHERE channel IN ('transfer', 'wallet');"
    )).scalar()
    if m2_channels > 0:
        raise RuntimeError(
            f"Downgrade aborted: {m2_channels} row(s) contain M2-only channels "
            "('transfer', 'wallet'). Revert would drop these values."
        )

    # Guard 2: Populated destination_account_id
    dest_count = bind.execute(sa.text(
        "SELECT COUNT(*) FROM transactions WHERE destination_account_id IS NOT NULL;"
    )).scalar()
    if dest_count > 0:
        raise RuntimeError(
            f"Downgrade aborted: {dest_count} row(s) contain destination_account_id values. "
            "Dropping the column would destroy this data."
        )

    # Guard 3: Populated merchant_id
    merch_count = bind.execute(sa.text(
        "SELECT COUNT(*) FROM transactions WHERE merchant_id IS NOT NULL;"
    )).scalar()
    if merch_count > 0:
        raise RuntimeError(
            f"Downgrade aborted: {merch_count} row(s) contain merchant_id values. "
            "Dropping the column would destroy this data."
        )

    # Guard 4: Populated ip_id
    ip_count = bind.execute(sa.text(
        "SELECT COUNT(*) FROM transactions WHERE ip_id IS NOT NULL;"
    )).scalar()
    if ip_count > 0:
        raise RuntimeError(
            f"Downgrade aborted: {ip_count} row(s) contain ip_id values. "
            "Dropping the column would destroy this data."
        )

    # Guard 5: NULL direction
    null_dir = bind.execute(sa.text(
        "SELECT COUNT(*) FROM transactions WHERE direction IS NULL;"
    )).scalar()
    if null_dir > 0:
        raise RuntimeError(
            f"Downgrade aborted: {null_dir} row(s) have NULL direction. "
            "Restoring NOT NULL constraint would violate data integrity. "
            "Fabricating 'debit' is prohibited."
        )

    # All guards passed — execute schema rollback
    op.drop_index('idx_txn_ip_id', table_name='transactions')
    op.drop_index('idx_txn_merchant_id', table_name='transactions')
    op.drop_index('idx_txn_dest_account', table_name='transactions')

    op.drop_constraint('ck_transactions_transfer_target', 'transactions', type_='check')

    op.drop_constraint('ck_transactions_direction', 'transactions', type_='check')
    op.alter_column('transactions', 'direction', existing_type=sa.VARCHAR(6), nullable=False)
    op.create_check_constraint(
        'transactions_direction_check',
        'transactions',
        "direction IN ('debit', 'credit')",
    )

    op.drop_constraint('ck_transactions_channel', 'transactions', type_='check')
    op.create_check_constraint(
        'transactions_channel_check',
        'transactions',
        "channel IN ('card', 'wire', 'upi', 'ach', 'atm')",
    )

    op.drop_constraint('fk_transactions_destination_account', 'transactions', type_='foreignkey')

    op.drop_column('transactions', 'destination_account_id')
    op.drop_column('transactions', 'ip_id')
    op.drop_column('transactions', 'merchant_id')

    # Option B: occurred_at remains canonical TIMESTAMPTZ (matches scripts/init_db.sql).
