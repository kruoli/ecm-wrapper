"""add ecm_progress generated column

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-01-24 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd4e5f6a7b8c9'
down_revision = 'c3d4e5f6a7b8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add ecm_progress as a generated (computed) column
    # This stores current_t_level / target_t_level for efficient filtering and indexing
    # NULLIF prevents division by zero - returns NULL when target_t_level is 0 or NULL
    op.execute("""
        ALTER TABLE composites
        ADD COLUMN ecm_progress FLOAT
        GENERATED ALWAYS AS (current_t_level / NULLIF(target_t_level, 0)) STORED
    """)

    # Add index for efficient filtering by progress percentage
    op.create_index('ix_composites_ecm_progress', 'composites', ['ecm_progress'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_composites_ecm_progress', table_name='composites')
    op.drop_column('composites', 'ecm_progress')
