"""add_index_on_current_composite

Revision ID: e7a08685979c
Revises: f080cc2efa57
Create Date: 2026-01-07 16:31:48.166174

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e7a08685979c'
down_revision = 'f080cc2efa57'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Use hash index instead of B-tree because composite numbers can exceed
    # PostgreSQL's 8KB B-tree index entry limit. Hash indexes are ideal for
    # equality lookups (which is all we do on this column) and have no size limit.
    op.execute(
        "CREATE INDEX ix_composites_current_composite ON composites USING hash (current_composite)"
    )


def downgrade() -> None:
    op.drop_index('ix_composites_current_composite', table_name='composites')