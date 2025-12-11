"""add prior_t_level to composites

Revision ID: b7c8d9e0f1a2
Revises: fea6b9ad9060
Create Date: 2025-12-11 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b7c8d9e0f1a2'
down_revision = 'fea6b9ad9060'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add prior_t_level column - nullable since existing composites won't have this
    # This represents ECM work done before the composite was imported into this system
    op.add_column('composites', sa.Column('prior_t_level', sa.Float(), nullable=True))

    # Add index for queries that filter by prior work
    op.create_index('ix_composites_prior_t_level', 'composites', ['prior_t_level'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_composites_prior_t_level', table_name='composites')
    op.drop_column('composites', 'prior_t_level')
