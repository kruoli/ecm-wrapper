"""Rename is_prime to is_complete

Revision ID: f080cc2efa57
Revises: 6d786bcad2e8
Create Date: 2026-01-06 06:07:12.231974

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f080cc2efa57'
down_revision = '6d786bcad2e8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Rename is_prime column to is_complete
    # This field now represents "sufficiently complete for OPN purposes"
    # instead of strictly "mathematically prime"
    op.alter_column('composites', 'is_prime', new_column_name='is_complete')


def downgrade() -> None:
    # Revert the column name back to is_prime
    op.alter_column('composites', 'is_complete', new_column_name='is_prime')