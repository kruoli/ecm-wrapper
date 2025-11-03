# Parametrization 0 Sigma Capture Fix

## Problem
GMP-ECM uses parametrization 0 (Brent-Suyama) for large numbers (>= ~200 digits), and sigma values for this parametrization can be very large (exceeding 2^63-1). Our code had two issues:

1. **Client regex patterns** only matched parametrizations 1-3, not 0
2. **Server database schema** used BigInteger for sigma (max 2^63-1 = 9,223,372,036,854,775,807)

Example problematic sigma: `0:17804659399245005261` (1.93x larger than BigInteger max)

## Client-Side Fixes

### File: `client/lib/parsing_utils.py`
Updated regex patterns to capture parametrization 0:

```python
# Before
SIGMA_COLON_FORMAT = re.compile(r'sigma=([1-3]:\d+)')
SIGMA_DASH_FORMAT = re.compile(r'-sigma (3:\d+|\d+)')
SIGMA_PARAM = re.compile(r'-sigma (3:\d+|\d+)')
SIGMA_USING_FORMAT = re.compile(r'Using.*?sigma=(?:3:)?(\d+)')

# After
SIGMA_COLON_FORMAT = re.compile(r'sigma=([0-3]:\d+)')
SIGMA_DASH_FORMAT = re.compile(r'-sigma ([0-3]:\d+|\d+)')
SIGMA_PARAM = re.compile(r'-sigma ([0-3]:\d+|\d+)')
SIGMA_USING_FORMAT = re.compile(r'Using.*?sigma=([0-3]:\d+)')
```

Also updated `extract_sigma_for_factor()` to return sigma directly (already includes parametrization prefix).

## Server-Side Fixes

### Database Migration: `server/migrations/versions/d3798d779007_change_sigma_from_bigint_to_text.py`
Changed `factors.sigma` column from `BigInteger` to `Text`:

```python
def upgrade() -> None:
    op.alter_column('factors', 'sigma',
                    existing_type=sa.BigInteger(),
                    type_=sa.Text(),
                    existing_nullable=True,
                    postgresql_using='sigma::text')
```

### File: `server/app/models/factors.py`
```python
# Before
sigma = Column(BigInteger, nullable=True)

# After
sigma = Column(Text, nullable=True)
```

### File: `server/app/services/factors.py`
Updated type hints and removed unnecessary string conversion:

```python
# Before
def add_factor(..., sigma: Optional[int] = None, ...):
    result = calculator.calculate_group_order(factor, str(sigma), parametrization)

# After
def add_factor(..., sigma: Optional[str] = None, ...):
    result = calculator.calculate_group_order(factor, sigma, parametrization)
```

### File: `server/app/api/v1/submit.py`
Changed sigma parsing to keep as string instead of converting to int:

```python
# Before
sigma = int(parts[1])  # or int(sigma_str)

# After
sigma = parts[1]  # Keep as string
```

## Testing

```bash
# Test with large number that uses parametrization 0
python3 ecm-wrapper.py --no-submit --tlevel 20 --multiprocess --workers 4 --composite <1028-digit-number>

# Output should show:
# Factor found with sigma: 0:17804659399245005261
```

## Migration Instructions

1. Apply database migration:
   ```bash
   cd server
   source venv/bin/activate
   alembic upgrade head
   ```

2. Restart API server to pick up model changes

3. Verify existing sigma values are preserved (they should automatically convert to text)

## Parametrization Reference

- **0**: Brent-Suyama (used for large numbers, large sigma values)
- **1**: Montgomery curves (CPU default for smaller numbers)
- **2**: Twisted Edwards
- **3**: Twisted Edwards (GPU default)
