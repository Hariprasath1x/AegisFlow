"""M2 Feature Store, Entity Graph & Known Fraud Schema

Revision ID: m2_0001
Revises: None
Create Date: 2026-10-31 12:15:00

"""

import json
import math
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision: str = 'm2_0001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

MANAGED_COMMENT = 'aegisflow:created_by=m2_0001'


def consolidate_entity_edge_duplicates(bind) -> None:
    """Consolidate duplicate entity edges before enforcing uniqueness.

    Aggregation rules:
    - Earliest first_seen_at is preserved.
    - Latest last_seen_at is preserved.
    - Weights are summed across duplicate occurrences (additive intensity).
    - Properties are merged without silent data loss (conflicting values stored as lists).
    - Canonical edge_id is chosen deterministically by earliest first_seen_at, then lowest edge_id.
    """
    dup_groups = bind.execute(sa.text("""
        SELECT src_type, src_id, dst_type, dst_id, edge_type, COUNT(*) as cnt
        FROM entity_edges
        GROUP BY src_type, src_id, dst_type, dst_id, edge_type
        HAVING COUNT(*) > 1
    """)).fetchall()

    if not dup_groups:
        return

    for group in dup_groups:
        st, si, dt, di, et = group[0], group[1], group[2], group[3], group[4]
        rows = bind.execute(sa.text("""
            SELECT edge_id, weight, first_seen_at, last_seen_at, properties
            FROM entity_edges
            WHERE src_type = :st AND src_id = :si AND dst_type = :dt AND dst_id = :di AND edge_type = :et
            ORDER BY first_seen_at ASC, edge_id ASC
        """), {"st": st, "si": si, "dt": dt, "di": di, "et": et}).mappings().fetchall()

        survivor_id = rows[0]["edge_id"]
        earliest_first_seen = min(r["first_seen_at"] for r in rows)
        latest_last_seen = max(r["last_seen_at"] for r in rows)
        total_weight = sum(float(r["weight"]) for r in rows)
        if not math.isfinite(total_weight) or total_weight <= 0:
            raise ValueError(
                f"Consolidation failed: non-finite or invalid total weight {total_weight} for edge "
                f"({st}:{si} -> {dt}:{di} [{et}])."
            )

        merged_properties = {}
        for r in rows:
            props = r["properties"] or {}
            if isinstance(props, str):
                props = json.loads(props)
            for k, v in props.items():
                if k not in merged_properties:
                    merged_properties[k] = v
                elif merged_properties[k] == v:
                    continue
                else:
                    prev = merged_properties[k]
                    if isinstance(prev, list):
                        if v not in prev:
                            prev.append(v)
                    else:
                        merged_properties[k] = [prev, v]

        bind.execute(sa.text("""
            UPDATE entity_edges
            SET weight = :w,
                first_seen_at = :fs,
                last_seen_at = :ls,
                properties = CAST(:props AS jsonb)
            WHERE edge_id = :survivor_id
        """), {
            "w": total_weight,
            "fs": earliest_first_seen,
            "ls": latest_last_seen,
            "props": json.dumps(merged_properties),
            "survivor_id": survivor_id,
        })

        bind.execute(sa.text("""
            DELETE FROM entity_edges
            WHERE src_type = :st AND src_id = :si AND dst_type = :dt AND dst_id = :di AND edge_type = :et
              AND edge_id != :survivor_id
        """), {"st": st, "si": si, "dt": dt, "di": di, "et": et, "survivor_id": survivor_id})


def is_table_owned(bind, table_name: str) -> bool:
    """Check if the table was created by this migration via table comment."""
    res = bind.execute(sa.text("""
        SELECT obj_description(c.oid, 'pg_class')
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relname = :tbl;
    """), {"tbl": table_name}).scalar()
    return res == MANAGED_COMMENT


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())

    # ============================================================
    # PRECONDITION CHECKS
    # ============================================================
    # If tables pre-exist, ensure they have compatible schemas or fail explicitly.
    if 'entity_edges' in existing_tables:
        cols = {c['name'] for c in inspector.get_columns('entity_edges')}
        required = {'src_type', 'src_id', 'dst_type', 'dst_id', 'edge_type', 'weight', 'first_seen_at', 'last_seen_at'}
        if not required.issubset(cols):
            raise RuntimeError(
                f"Precondition failed: table 'entity_edges' already exists with incompatible schema "
                f"(missing: {sorted(required - cols)}). Aborting migration m2_0001 to prevent data loss."
            )

    if 'graph_feature_snapshots' in existing_tables:
        cols = {c['name'] for c in inspector.get_columns('graph_feature_snapshots')}
        required = {'snapshot_id', 'account_id', 'snapshot_at', 'pagerank', 'degree'}
        if not required.issubset(cols):
            raise RuntimeError(
                f"Precondition failed: table 'graph_feature_snapshots' already exists with incompatible schema "
                f"(missing: {sorted(required - cols)}). Aborting migration m2_0001."
            )

    if 'ground_truth_labels' in existing_tables:
        cols = {c['name'] for c in inspector.get_columns('ground_truth_labels')}
        required = {'label_id', 'is_fraud', 'confidence'}
        if not required.issubset(cols):
            raise RuntimeError(
                f"Precondition failed: table 'ground_truth_labels' already exists with incompatible schema "
                f"(missing: {sorted(required - cols)}). Aborting migration m2_0001."
            )

    # ============================================================
    # 1. ENTITY EDGES
    # ============================================================
    if 'entity_edges' not in existing_tables:
        op.create_table(
            'entity_edges',
            sa.Column('edge_id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
            sa.Column('src_type', sa.String(length=32), nullable=False),
            sa.Column('src_id', sa.String(length=80), nullable=False),
            sa.Column('dst_type', sa.String(length=32), nullable=False),
            sa.Column('dst_id', sa.String(length=80), nullable=False),
            sa.Column('edge_type', sa.String(length=32), nullable=False),
            sa.Column('weight', sa.Double(), server_default='1.0', nullable=False),
            sa.Column('first_seen_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.Column('last_seen_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.Column('properties', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
            sa.PrimaryKeyConstraint('edge_id', name='pk_entity_edges'),
            sa.UniqueConstraint('src_type', 'src_id', 'dst_type', 'dst_id', 'edge_type', name='uq_entity_edges_identity'),
            sa.CheckConstraint("weight > 0 AND weight < 'Infinity'::double precision", name='ck_entity_edges_weight_positive'),
            sa.CheckConstraint('length(src_type) > 0 AND length(src_id) > 0', name='ck_entity_edges_src_not_empty'),
            sa.CheckConstraint('length(dst_type) > 0 AND length(dst_id) > 0', name='ck_entity_edges_dst_not_empty'),
            sa.CheckConstraint('length(edge_type) > 0', name='ck_entity_edges_type_not_empty'),
            sa.CheckConstraint('first_seen_at <= last_seen_at', name='ck_entity_edges_chronology'),
        )
        op.create_index('idx_entity_edges_src', 'entity_edges', ['src_type', 'src_id'], unique=False)
        op.create_index('idx_entity_edges_dst', 'entity_edges', ['dst_type', 'dst_id'], unique=False)
        op.create_index('idx_entity_edges_type', 'entity_edges', ['edge_type'], unique=False)
        op.execute(f"COMMENT ON TABLE entity_edges IS '{MANAGED_COMMENT}';")
    else:
        # Pre-existing compatible table: consolidate duplicates before applying unique constraint
        consolidate_entity_edge_duplicates(bind)
        constraints = [c['name'] for c in inspector.get_unique_constraints('entity_edges')]
        if 'uq_entity_edges_identity' not in constraints:
            op.create_unique_constraint(
                'uq_entity_edges_identity',
                'entity_edges',
                ['src_type', 'src_id', 'dst_type', 'dst_id', 'edge_type']
            )

    # ============================================================
    # 2. GRAPH FEATURE SNAPSHOTS
    # ============================================================
    if 'graph_feature_snapshots' not in existing_tables:
        op.create_table(
            'graph_feature_snapshots',
            sa.Column('snapshot_id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
            sa.Column('account_id', sa.UUID(), nullable=False),
            sa.Column('snapshot_at', sa.DateTime(timezone=True), nullable=False),
            sa.Column('pagerank', sa.Double(), server_default='0.0', nullable=False),
            sa.Column('degree', sa.Integer(), server_default='0', nullable=False),
            sa.Column('in_degree', sa.Integer(), server_default='0', nullable=False),
            sa.Column('out_degree', sa.Integer(), server_default='0', nullable=False),
            sa.Column('clustering_coefficient', sa.Double(), server_default='0.0', nullable=False),
            sa.Column('features', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.PrimaryKeyConstraint('snapshot_id', name='pk_graph_feature_snapshots'),
            sa.ForeignKeyConstraint(['account_id'], ['accounts.account_id'], name='fk_graph_snapshots_account', ondelete='CASCADE'),
            sa.UniqueConstraint('account_id', 'snapshot_at', name='uq_graph_snapshots_account_time'),
            sa.CheckConstraint("pagerank >= 0.0 AND pagerank < 'Infinity'::double precision", name='ck_graph_snapshots_pagerank_nonneg'),
            sa.CheckConstraint('degree >= 0 AND in_degree >= 0 AND out_degree >= 0', name='ck_graph_snapshots_degree_nonneg'),
            sa.CheckConstraint('clustering_coefficient >= 0.0 AND clustering_coefficient <= 1.0', name='ck_graph_snapshots_clustering_range'),
        )
        # Note: uq_graph_snapshots_account_time provides index on (account_id, snapshot_at);
        # idx_graph_snapshots_snapshot_at optimizes timeline scans across all accounts.
        op.create_index('idx_graph_snapshots_snapshot_at', 'graph_feature_snapshots', [sa.text('snapshot_at DESC')], unique=False)
        op.execute(f"COMMENT ON TABLE graph_feature_snapshots IS '{MANAGED_COMMENT}';")

    # ============================================================
    # 3. GROUND TRUTH LABELS
    # ============================================================
    if 'ground_truth_labels' not in existing_tables:
        op.create_table(
            'ground_truth_labels',
            sa.Column('label_id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
            sa.Column('txn_id', sa.UUID(), nullable=True),
            sa.Column('account_id', sa.UUID(), nullable=True),
            sa.Column('label_target', sa.String(length=20), server_default='transaction', nullable=False),
            sa.Column('is_fraud', sa.Boolean(), server_default=sa.text('true'), nullable=False),
            sa.Column('fraud_type', sa.String(length=50), nullable=True),
            sa.Column('confidence', sa.Double(), server_default='1.0', nullable=False),
            sa.Column('source', sa.String(length=50), server_default='seed', nullable=False),
            sa.Column('labeled_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.Column('notes', sa.Text(), nullable=True),
            sa.PrimaryKeyConstraint('label_id', name='pk_ground_truth_labels'),
            sa.ForeignKeyConstraint(['txn_id'], ['transactions.txn_id'], name='fk_ground_truth_txn', ondelete='CASCADE'),
            sa.ForeignKeyConstraint(['account_id'], ['accounts.account_id'], name='fk_ground_truth_account', ondelete='CASCADE'),
            sa.CheckConstraint("label_target IN ('transaction', 'account')", name='ck_ground_truth_target_valid'),
            sa.CheckConstraint(
                "(label_target = 'transaction' AND txn_id IS NOT NULL AND account_id IS NULL) "
                "OR (label_target = 'account' AND account_id IS NOT NULL AND txn_id IS NULL)",
                name='ck_ground_truth_target_presence'
            ),
            sa.CheckConstraint('confidence >= 0.0 AND confidence <= 1.0', name='ck_ground_truth_confidence_range'),
        )
        op.create_index(
            'uq_ground_truth_txn',
            'ground_truth_labels',
            ['txn_id'],
            unique=True,
            postgresql_where=sa.text('txn_id IS NOT NULL')
        )
        op.create_index(
            'uq_ground_truth_account',
            'ground_truth_labels',
            ['account_id', sa.text("COALESCE(fraud_type, '')")],
            unique=True,
            postgresql_where=sa.text("label_target = 'account'")
        )
        op.create_index('idx_ground_truth_account', 'ground_truth_labels', ['account_id'], unique=False)
        op.create_index('idx_ground_truth_labeled_at', 'ground_truth_labels', ['labeled_at'], unique=False)
        op.create_index('idx_ground_truth_fraud', 'ground_truth_labels', ['is_fraud'], unique=False)
        op.execute(f"COMMENT ON TABLE ground_truth_labels IS '{MANAGED_COMMENT}';")


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())

    # Only drop tables that were explicitly created by this migration
    if 'ground_truth_labels' in existing_tables:
        if is_table_owned(bind, 'ground_truth_labels'):
            op.drop_table('ground_truth_labels')

    if 'graph_feature_snapshots' in existing_tables:
        if is_table_owned(bind, 'graph_feature_snapshots'):
            op.drop_table('graph_feature_snapshots')

    if 'entity_edges' in existing_tables:
        if is_table_owned(bind, 'entity_edges'):
            op.drop_table('entity_edges')
        else:
            # Revert only the unique constraint if it was added to an adopted table
            constraints = [c['name'] for c in inspector.get_unique_constraints('entity_edges')]
            if 'uq_entity_edges_identity' in constraints:
                op.drop_constraint('uq_entity_edges_identity', 'entity_edges', type_='unique')
