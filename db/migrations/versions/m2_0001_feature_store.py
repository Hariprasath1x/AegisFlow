"""M2 Feature Store, Entity Graph & Known Fraud Schema

Revision ID: m2_0001
Revises: None
Create Date: 2026-10-31 11:55:00

"""

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


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = inspector.get_table_names()

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
            sa.UniqueConstraint('src_type', 'src_id', 'dst_type', 'dst_id', 'edge_type', name='uq_entity_edges_identity')
        )
        op.create_index('idx_entity_edges_src', 'entity_edges', ['src_type', 'src_id'], unique=False)
        op.create_index('idx_entity_edges_dst', 'entity_edges', ['dst_type', 'dst_id'], unique=False)
        op.create_index('idx_entity_edges_type', 'entity_edges', ['edge_type'], unique=False)
    else:
        # Table pre-exists; check and consolidate any duplicates before applying unique constraint
        op.execute("""
            DELETE FROM entity_edges a USING entity_edges b
            WHERE a.src_type = b.src_type
              AND a.src_id = b.src_id
              AND a.dst_type = b.dst_type
              AND a.dst_id = b.dst_id
              AND a.edge_type = b.edge_type
              AND a.edge_id > b.edge_id;
        """)
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
            sa.UniqueConstraint('account_id', 'snapshot_at', name='uq_graph_snapshots_account_time')
        )
        op.create_index('idx_graph_snapshots_account_time', 'graph_feature_snapshots', ['account_id', sa.text('snapshot_at DESC')], unique=False)
        op.create_index('idx_graph_snapshots_time', 'graph_feature_snapshots', [sa.text('snapshot_at DESC')], unique=False)

    # ============================================================
    # 3. GROUND TRUTH LABELS
    # ============================================================
    if 'ground_truth_labels' not in existing_tables:
        op.create_table(
            'ground_truth_labels',
            sa.Column('label_id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
            sa.Column('txn_id', sa.UUID(), nullable=True),
            sa.Column('account_id', sa.UUID(), nullable=True),
            sa.Column('is_fraud', sa.Boolean(), server_default=sa.text('true'), nullable=False),
            sa.Column('fraud_type', sa.String(length=50), nullable=True),
            sa.Column('confidence', sa.Double(), server_default='1.0', nullable=False),
            sa.Column('source', sa.String(length=50), server_default='seed', nullable=False),
            sa.Column('labeled_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.Column('notes', sa.Text(), nullable=True),
            sa.PrimaryKeyConstraint('label_id', name='pk_ground_truth_labels'),
            sa.ForeignKeyConstraint(['txn_id'], ['transactions.txn_id'], name='fk_ground_truth_txn', ondelete='CASCADE'),
            sa.ForeignKeyConstraint(['account_id'], ['accounts.account_id'], name='fk_ground_truth_account', ondelete='CASCADE'),
            sa.UniqueConstraint('txn_id', name='uq_ground_truth_txn')
        )
        op.create_index('idx_ground_truth_account', 'ground_truth_labels', ['account_id'], unique=False)
        op.create_index('idx_ground_truth_labeled_at', 'ground_truth_labels', ['labeled_at'], unique=False)
        op.create_index('idx_ground_truth_fraud', 'ground_truth_labels', ['is_fraud'], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = inspector.get_table_names()

    if 'ground_truth_labels' in existing_tables:
        op.drop_table('ground_truth_labels')

    if 'graph_feature_snapshots' in existing_tables:
        op.drop_table('graph_feature_snapshots')

    if 'entity_edges' in existing_tables:
        op.drop_table('entity_edges')

