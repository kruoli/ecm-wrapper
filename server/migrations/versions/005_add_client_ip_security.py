"""Add client_ip to ecm_attempts for security logging

Revision ID: 005_add_client_ip
Revises: 004_simplify_minimal
Create Date: 2025-01-30

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '005_add_client_ip'
down_revision = '004_simplify_minimal'
branch_labels = None
depends_on = None


def upgrade():
    # Add client_ip column to ecm_attempts table
    op.add_column('ecm_attempts', sa.Column('client_ip', sa.String(length=45), nullable=True))

    # Add index on client_ip for querying attempts by IP
    op.create_index('ix_ecm_attempts_client_ip', 'ecm_attempts', ['client_ip'], unique=False)


def downgrade():
    # Remove index
    op.drop_index('ix_ecm_attempts_client_ip', table_name='ecm_attempts')

    # Remove column
    op.drop_column('ecm_attempts', 'client_ip')