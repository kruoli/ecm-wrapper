"""add_group_order_to_factors

Revision ID: 4a410e518701
Revises: 2a113ed31d3f
Create Date: 2025-10-08 21:32:51.112864

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '4a410e518701'
down_revision = '2a113ed31d3f'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add group_order and group_order_factorization columns to factors table
    op.add_column('factors', sa.Column('group_order', sa.Text(), nullable=True))
    op.add_column('factors', sa.Column('group_order_factorization', sa.Text(), nullable=True))


def downgrade() -> None:
    # Remove group_order columns
    op.drop_column('factors', 'group_order_factorization')
    op.drop_column('factors', 'group_order')