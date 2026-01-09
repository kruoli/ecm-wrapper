"""
Shared constants for the ECM coordination server.

This module centralizes configuration constants used across multiple modules
to avoid duplication and ensure consistency.
"""

from typing import List, Tuple

# ECM parameter table based on Paul Zimmerman's GMP-ECM 7 recommendations
# Source: https://www.rieselprime.de/ziki/Elliptic_curve_method
# Format: (max_digits, b1, b2, typical_curves)
# - max_digits: Maximum factor size (in digits) this B1 level is suitable for
# - b1: Stage 1 bound (from Zimmerman table)
# - b2: Stage 2 bound (approximated as 100*B1, GMP-ECM calculates optimal internally)
# - typical_curves: Expected curves for GMP-ECM 7 default parameters
#
# Note: These values are currently used for internal bookkeeping only.
# The client calculates actual parameters via the t-level binary.
ECM_BOUNDS: List[Tuple[int, int, int, int]] = [
    (20, 11000, 1100000, 107),
    (25, 50000, 5000000, 261),
    (30, 250000, 25000000, 513),
    (35, 1000000, 100000000, 1071),
    (40, 3000000, 300000000, 2753),
    (45, 11000000, 1100000000, 5208),
    (50, 43000000, 4300000000, 8704),
    (55, 110000000, 11000000000, 20479),
    (60, 260000000, 26000000000, 47888),
    (65, 850000000, 85000000000, 78923),
    (70, 2900000000, 290000000000, 115153),
    (75, 7600000000, 760000000000, 211681),
    (80, 25000000000, 2500000000000, 296479),
]
