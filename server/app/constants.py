"""
Shared constants for the ECM coordination server.

This module centralizes configuration constants used across multiple modules
to avoid duplication and ensure consistency.
"""

from typing import List, Tuple

# ECM parameter table based on GMP-ECM documentation and best practices
# Format: (max_digits, b1, b2, typical_curves)
# - max_digits: Maximum factor size (in digits) this B1 level is suitable for
# - b1: Stage 1 bound
# - b2: Stage 2 bound
# - typical_curves: Number of curves typically needed to find a factor of this size
ECM_BOUNDS: List[Tuple[int, int, int, int]] = [
    (30, 2000, 147000, 25),
    (35, 11000, 1900000, 90),
    (40, 50000, 12500000, 300),
    (45, 250000, 128000000, 700),
    (50, 1000000, 1000000000, 1800),
    (55, 3000000, 5000000000, 5100),
    (60, 11000000, 35000000000, 10600),
    (65, 43000000, 240000000000, 19300),
    (70, 110000000, 873000000000, 49000),
    (75, 260000000, 2600000000000, 124000),
    (80, 850000000, 11700000000000, 210000),
    (85, 2900000000, 55300000000000, 340000),
]
