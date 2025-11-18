"""add_ecm_residues_and_superseded_by

Revision ID: a1b2c3d4e5f6
Revises: d3798d779007
Create Date: 2025-11-17 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = 'd3798d779007'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add superseded_by column to ecm_attempts for tracking stage supersession
    op.add_column('ecm_attempts', sa.Column('superseded_by', sa.Integer(), nullable=True))
    op.create_foreign_key(
        'fk_ecm_attempts_superseded_by',
        'ecm_attempts', 'ecm_attempts',
        ['superseded_by'], ['id']
    )

    # Create ecm_residues table for decoupled two-stage ECM
    op.create_table(
        'ecm_residues',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('composite_id', sa.Integer(), nullable=False),
        sa.Column('client_id', sa.String(255), nullable=False),
        sa.Column('stage1_attempt_id', sa.Integer(), nullable=True),
        sa.Column('b1', sa.BigInteger(), nullable=False),
        sa.Column('parametrization', sa.Integer(), nullable=False),
        sa.Column('curve_count', sa.Integer(), nullable=False),
        sa.Column('storage_path', sa.String(512), nullable=False),
        sa.Column('file_size_bytes', sa.Integer(), nullable=False),
        sa.Column('checksum', sa.String(64), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='available'),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('claimed_at', sa.DateTime(), nullable=True),
        sa.Column('claimed_by', sa.String(255), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['composite_id'], ['composites.id'], ),
        sa.ForeignKeyConstraint(['stage1_attempt_id'], ['ecm_attempts.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('storage_path')
    )

    # Create indexes for ecm_residues
    op.create_index('ix_ecm_residues_id', 'ecm_residues', ['id'])
    op.create_index('ix_ecm_residues_client_id', 'ecm_residues', ['client_id'])
    op.create_index('ix_ecm_residues_status', 'ecm_residues', ['status'])
    op.create_index('ix_ecm_residues_composite_status', 'ecm_residues', ['composite_id', 'status'])
    op.create_index('ix_ecm_residues_expires_at', 'ecm_residues', ['expires_at'])


def downgrade() -> None:
    # Drop ecm_residues table and indexes
    op.drop_index('ix_ecm_residues_expires_at', table_name='ecm_residues')
    op.drop_index('ix_ecm_residues_composite_status', table_name='ecm_residues')
    op.drop_index('ix_ecm_residues_status', table_name='ecm_residues')
    op.drop_index('ix_ecm_residues_client_id', table_name='ecm_residues')
    op.drop_index('ix_ecm_residues_id', table_name='ecm_residues')
    op.drop_table('ecm_residues')

    # Remove superseded_by column from ecm_attempts
    op.drop_constraint('fk_ecm_attempts_superseded_by', 'ecm_attempts', type_='foreignkey')
    op.drop_column('ecm_attempts', 'superseded_by')
