"""change_sigma_from_bigint_to_text

Revision ID: d3798d779007
Revises: 4a410e518701
Create Date: 2025-11-02 16:41:05.108758

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd3798d779007'
down_revision = '4a410e518701'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Change sigma column from BigInteger to Text to support large parametrization 0 sigma values
    # Parametrization 0 (Brent-Suyama) can have sigma values exceeding BigInteger max (2^63-1)
    op.alter_column('factors', 'sigma',
                    existing_type=sa.BigInteger(),
                    type_=sa.Text(),
                    existing_nullable=True,
                    postgresql_using='sigma::text')


def downgrade() -> None:
    # WARNING: Downgrading from Text to BigInteger may fail if sigma values exceed BigInteger range
    # This downgrade assumes all existing sigma values fit within BigInteger range
    op.alter_column('factors', 'sigma',
                    existing_type=sa.Text(),
                    type_=sa.BigInteger(),
                    existing_nullable=True,
                    postgresql_using='sigma::bigint')