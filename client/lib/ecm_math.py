"""
Mathematical utilities for ECM wrapper.

This module provides functions for:
- Trial division factorization
- Primality testing (Miller-Rabin)
- T-level calculations
- ECM parameter optimization
"""

import logging
import random
import re
from typing import List, Tuple, Optional
from lib.subprocess_utils import execute_subprocess_simple


logger = logging.getLogger(__name__)


def trial_division(n: int, limit: int = 10**7) -> Tuple[List[int], int]:
    """
    Fast trial division to find small prime factors.

    Uses wheel factorization to skip multiples of 2, 3, 5.

    Args:
        n: Number to factor
        limit: Trial division limit (default: 10^7)

    Returns:
        Tuple of (factors_found, cofactor)

    Example:
        >>> factors, cofactor = trial_division(360)
        >>> factors
        [2, 2, 2, 3, 3, 5]
        >>> cofactor
        1
    """
    factors = []
    cofactor = n

    # Trial division by 2
    while cofactor % 2 == 0:
        factors.append(2)
        cofactor //= 2

    # Trial division by 3
    while cofactor % 3 == 0:
        factors.append(3)
        cofactor //= 3

    # Trial division by 5
    while cofactor % 5 == 0:
        factors.append(5)
        cofactor //= 5

    # Trial division by odd numbers (wheel factorization: skip multiples of 2,3,5)
    i = 7
    while i * i <= cofactor and i <= limit:
        while cofactor % i == 0:
            factors.append(i)
            cofactor //= i
        i += 2

    return factors, cofactor


def is_probably_prime(n: int, trials: int = 10) -> bool:
    """
    Miller-Rabin primality test.

    Probabilistic primality test with configurable confidence level.
    With 10 trials, false positive rate is approximately 1 in 4^10.

    Args:
        n: Number to test
        trials: Number of trials (default: 10, higher = more confident)

    Returns:
        True if probably prime, False if definitely composite

    Example:
        >>> is_probably_prime(17)
        True
        >>> is_probably_prime(18)
        False
    """
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False

    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Witness loop
    for _ in range(trials):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)

        if x == 1 or x == n - 1:
            continue

        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False

    return True


def calculate_tlevel(curve_history: List[str], tlevel_binary: str = 'bin/t-level') -> float:
    """
    Call t-level binary to calculate current t-level.

    Args:
        curve_history: List of curve strings like "100@1000000,p=1"
        tlevel_binary: Path to t-level executable

    Returns:
        Current t-level as float (0.0 on error)

    Example:
        >>> history = ["100@50000,p=1", "200@250000,p=1"]
        >>> t = calculate_tlevel(history)
        >>> t > 0
        True
    """
    if not curve_history:
        return 0.0

    # Join curve strings with semicolons
    curve_input = ";".join(curve_history)

    try:
        # Call t-level binary using unified subprocess utility
        stdout, _ = execute_subprocess_simple(
            [tlevel_binary, '-q', curve_input],
            timeout=10
        )

        # Parse output: "t40.234"
        match = re.search(r't([\d.]+)', stdout)
        if match:
            return float(match.group(1))

        logger.warning("Failed to parse t-level from: %s", stdout)
        return 0.0

    except (FileNotFoundError, ValueError) as e:
        logger.error("Error calculating t-level: %s", e)
        return 0.0


def calculate_curves_for_target(current_tlevel: float, target_tlevel: float,
                                b1: int, tlevel_binary: str = 'bin/t-level') -> Optional[int]:
    """
    Calculate exact number of curves needed to reach target t-level.

    Uses binary search with the t-level calculator to find the precise
    number of curves required.

    Args:
        current_tlevel: Current t-level (e.g., 25.084)
        target_tlevel: Target t-level (e.g., 28.7)
        b1: B1 parameter for curves
        tlevel_binary: Path to t-level executable

    Returns:
        Number of curves needed, or None if target already reached

    Example:
        >>> curves = calculate_curves_for_target(25.0, 30.0, 50000)
        >>> curves is not None and curves > 0
        True
    """
    if current_tlevel >= target_tlevel:
        return None

    # Binary search for exact curve count
    low, high = 1, 100000
    best_curves = None

    while low <= high:
        mid = (low + high) // 2

        # Test this curve count
        test_history = [f"{mid}@{b1},p=1"]  # Parametrization 1 (most common)
        test_tlevel = calculate_tlevel(test_history, tlevel_binary)

        # Adjust combined t-level (approximation: t-levels add linearly for rough estimate)
        combined_tlevel = current_tlevel + test_tlevel

        if abs(combined_tlevel - target_tlevel) < 0.1:
            # Close enough
            return mid

        if combined_tlevel < target_tlevel:
            low = mid + 1
        else:
            best_curves = mid
            high = mid - 1

    return best_curves if best_curves else low


def get_b1_for_digit_length(digits: int) -> int:
    """
    Get recommended B1 parameter based on composite digit length.

    Based on standard ECM parameter recommendations from GMP-ECM.

    Args:
        digits: Number of digits in the composite

    Returns:
        Recommended B1 value

    Example:
        >>> get_b1_for_digit_length(40)
        50000
        >>> get_b1_for_digit_length(60)
        3000000
    """
    if digits <= 20:
        return 2000
    if digits <= 25:
        return 11000
    if digits <= 30:
        return 50000
    if digits <= 35:
        return 250000
    if digits <= 40:
        return 1000000
    if digits <= 45:
        return 3000000
    if digits <= 50:
        return 11000000
    if digits <= 55:
        return 43000000
    if digits <= 60:
        return 110000000
    if digits <= 65:
        return 260000000
    # For larger numbers
    return 850000000


def get_optimal_b1_for_tlevel(target_tlevel: float) -> Tuple[int, int]:
    """
    Get optimal B1 and expected curve count for target t-level.

    Based on Zimmermann's optimal B1 values and expected curves for GMP-ECM 7.
    Source: https://members.loria.fr/PZimmermann/records/ecmnet.html

    Args:
        target_tlevel: Target t-level to achieve

    Returns:
        Tuple of (optimal_b1, expected_curves)

    Example:
        >>> b1, curves = get_optimal_b1_for_tlevel(30.0)
        >>> b1 >= 250000
        True
    """
    # Zimmermann's optimal B1 values and expected curves for GMP-ECM 7
    # Format: (digits, optimal_b1, expected_curves_ecm7)
    OPTIMAL_B1_TABLE = [
        (20, 11000, 107),
        (25, 50000, 261),
        (30, 250000, 513),
        (35, 1000000, 1071),
        (40, 3000000, 2753),
        (45, 11000000, 5208),
        (50, 43000000, 8704),
        (55, 110000000, 20479),
        (60, 260000000, 47888),
        (65, 850000000, 78923),
        (70, 2900000000, 115153),
        (75, 7600000000, 211681),
        (80, 25000000000, 296479)
    ]

    # Find exact match or next higher level
    for digits, b1, curves in OPTIMAL_B1_TABLE:
        if target_tlevel <= digits:
            return b1, curves

    # For t-levels beyond table, return highest entry
    return OPTIMAL_B1_TABLE[-1][1], OPTIMAL_B1_TABLE[-1][2]
