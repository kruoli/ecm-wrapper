"""add index on ecm_residues.created_at

Revision ID: a9c1f2e8b3d4
Revises: d4e5f6a7b8c9
Create Date: 2026-05-05 12:00:00.000000

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = 'a9c1f2e8b3d4'
down_revision = 'd4e5f6a7b8c9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        'ix_ecm_residues_created_at',
        'ecm_residues',
        ['created_at'],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index('ix_ecm_residues_created_at', table_name='ecm_residues')
