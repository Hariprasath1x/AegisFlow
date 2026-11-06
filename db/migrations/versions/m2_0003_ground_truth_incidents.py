"""M2 Ground Truth Incident Boundaries and Multi-Incident Indexing

Revision ID: m2_0003
Revises: m2_0002
Create Date: 2026-11-06 23:30:00

Schema changes to ground_truth_labels:
- Add incident_start_at TIMESTAMPTZ NULL
- Add incident_end_at TIMESTAMPTZ NULL
- Add check constraint ck_ground_truth_incident_paired_boundaries:
    (incident_start_at IS NULL AND incident_end_at IS NULL) OR
    (incident_start_at IS NOT NULL AND incident_end_at IS NOT NULL)
- Add check constraint ck_ground_truth_incident_chronology:
    incident_start_at IS NULL OR incident_end_at IS NULL OR incident_start_at <= incident_end_at
- Add check constraint ck_ground_truth_incident_discovery:
    incident_start_at IS NULL OR incident_start_at <= labeled_at
- Drop legacy uq_ground_truth_account unique index
- Create uq_ground_truth_account_timeless:
    UNIQUE (account_id, COALESCE(fraud_type, '')) WHERE label_target = 'account' AND incident_start_at IS NULL
- Create uq_ground_truth_account_incident:
    UNIQUE (account_id, COALESCE(fraud_type, ''), incident_start_at) WHERE label_target = 'account' AND incident_start_at IS NOT NULL
- Create b-tree lookup index idx_ground_truth_incident_window:
    (account_id, incident_start_at, incident_end_at) WHERE label_target = 'account' AND incident_start_at IS NOT NULL
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'm2_0003'
down_revision: Union[str, None] = 'm2_0002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add dedicated UTC incident boundary columns
    op.add_column(
        'ground_truth_labels',
        sa.Column('incident_start_at', sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        'ground_truth_labels',
        sa.Column('incident_end_at', sa.DateTime(timezone=True), nullable=True)
    )

    # 2. Add domain integrity check constraints
    op.create_check_constraint(
        'ck_ground_truth_incident_paired_boundaries',
        'ground_truth_labels',
        '(incident_start_at IS NULL AND incident_end_at IS NULL) OR '
        '(incident_start_at IS NOT NULL AND incident_end_at IS NOT NULL)'
    )
    op.create_check_constraint(
        'ck_ground_truth_incident_chronology',
        'ground_truth_labels',
        'incident_start_at IS NULL OR incident_end_at IS NULL OR incident_start_at <= incident_end_at'
    )
    op.create_check_constraint(
        'ck_ground_truth_incident_discovery',
        'ground_truth_labels',
        'incident_start_at IS NULL OR incident_start_at <= labeled_at'
    )

    # 3. Replace legacy account unique index with dual partial unique indexes
    op.drop_index('uq_ground_truth_account', table_name='ground_truth_labels')

    op.create_index(
        'uq_ground_truth_account_timeless',
        'ground_truth_labels',
        ['account_id', sa.text("COALESCE(fraud_type, '')")],
        unique=True,
        postgresql_where=sa.text("label_target = 'account' AND incident_start_at IS NULL")
    )
    op.create_index(
        'uq_ground_truth_account_incident',
        'ground_truth_labels',
        ['account_id', sa.text("COALESCE(fraud_type, '')"), 'incident_start_at'],
        unique=True,
        postgresql_where=sa.text("label_target = 'account' AND incident_start_at IS NOT NULL")
    )

    # 4. Partial index for point-in-time range lookups
    op.create_index(
        'idx_ground_truth_incident_window',
        'ground_truth_labels',
        ['account_id', 'incident_start_at', 'incident_end_at'],
        unique=False,
        postgresql_where=sa.text("label_target = 'account' AND incident_start_at IS NOT NULL")
    )


def downgrade() -> None:
    # Downgrade guard: check if duplicate incidents exist before restoring legacy index
    bind = op.get_bind()
    dups = bind.execute(sa.text("""
        SELECT account_id, COALESCE(fraud_type, '') AS fraud_typology, COUNT(*) as cnt
        FROM ground_truth_labels
        WHERE label_target = 'account'
        GROUP BY account_id, COALESCE(fraud_type, '')
        HAVING COUNT(*) > 1;
    """)).fetchall()

    if dups:
        raise RuntimeError(
            f"Downgrade aborted: {len(dups)} account(s) have multiple incidents of the same fraud typology. "
            "Restoring legacy uq_ground_truth_account would cause unique key violations."
        )

    # 1. Drop incident indexes
    op.drop_index('idx_ground_truth_incident_window', table_name='ground_truth_labels')
    op.drop_index('uq_ground_truth_account_incident', table_name='ground_truth_labels')
    op.drop_index('uq_ground_truth_account_timeless', table_name='ground_truth_labels')

    # 2. Restore legacy account unique index
    op.create_index(
        'uq_ground_truth_account',
        'ground_truth_labels',
        ['account_id', sa.text("COALESCE(fraud_type, '')")],
        unique=True,
        postgresql_where=sa.text("label_target = 'account'")
    )

    # 3. Drop check constraints
    op.drop_constraint('ck_ground_truth_incident_discovery', 'ground_truth_labels', type_='check')
    op.drop_constraint('ck_ground_truth_incident_chronology', 'ground_truth_labels', type_='check')
    op.drop_constraint('ck_ground_truth_incident_paired_boundaries', 'ground_truth_labels', type_='check')

    # 4. Drop columns
    op.drop_column('ground_truth_labels', 'incident_end_at')
    op.drop_column('ground_truth_labels', 'incident_start_at')

