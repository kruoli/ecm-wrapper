"""Add residue_checksum to ecm_attempts

Revision ID: c3d4e5f6a7b8
Revises: e7a08685979c
Create Date: 2026-01-13

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c3d4e5f6a7b8'
down_revision = 'e7a08685979c'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add residue_checksum column to track which residue file the attempt came from
    # This enables detection of orphaned attempts when a residue is completed by another client
    op.add_column('ecm_attempts', sa.Column('residue_checksum', sa.String(64), nullable=True))
    op.create_index('ix_ecm_attempts_residue_checksum', 'ecm_attempts', ['residue_checksum'])


def downgrade() -> None:
    op.drop_index('ix_ecm_attempts_residue_checksum', table_name='ecm_attempts')
    op.drop_column('ecm_attempts', 'residue_checksum')
