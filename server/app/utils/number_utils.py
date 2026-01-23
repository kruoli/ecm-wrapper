import math
import re
import random

def validate_integer(number_str: str) -> bool:
    """Validate that string represents a positive integer."""
    if not isinstance(number_str, str):
        return False

    # Check if string contains only digits
    if not re.match(r'^\d+$', number_str):
        return False

    # Check for leading zeros (except single "0")
    if len(number_str) > 1 and number_str[0] == '0':
        return False

    return True

def calculate_bit_length(number_str: str) -> int:
    """Calculate bit length of a number given as string."""
    if not validate_integer(number_str):
        raise ValueError(f"Invalid number format: {number_str}")

    number = int(number_str)
    if number == 0:
        return 1

    return number.bit_length()

def calculate_digit_length(number_str: str) -> int:
    """Calculate decimal digit length of a number given as string."""
    if not validate_integer(number_str):
        raise ValueError(f"Invalid number format: {number_str}")

    return len(number_str)

def gcd(a: int, b: int) -> int:
    """Calculate greatest common divisor using Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return a

def is_trivial_factor(factor: str, composite: str) -> bool:
    """Check if factor is trivial (1 or the number itself)."""
    return factor == "1" or factor == composite

def verify_factor_divides(factor: str, composite: str) -> bool:
    """
    Verify that a factor actually divides the composite.

    Args:
        factor: Factor to verify (as string)
        composite: The composite number (as string)

    Returns:
        True if factor divides composite evenly, False otherwise
    """
    if not validate_integer(factor) or not validate_integer(composite):
        return False

    # Check for trivial cases
    if factor == "1":
        return True  # 1 divides everything
    if factor == composite:
        return True  # Number divides itself

    try:
        factor_int = int(factor)
        composite_int = int(composite)

        # Check if factor is greater than composite
        if factor_int > composite_int:
            return False

        # Check if composite is divisible by factor
        return composite_int % factor_int == 0

    except (ValueError, OverflowError):
        return False

def verify_complete_factorization(composite: str, factors: list[str]) -> bool:
    """
    Verify that the product of factors equals the composite.
    Extracted from FactorService for better separation of concerns.
    """
    if not factors:
        return False

    try:
        # Calculate product of all factors
        product = 1
        for factor in factors:
            if not validate_integer(factor):
                return False
            product *= int(factor)

        return str(product) == composite
    except (ValueError, OverflowError):
        return False


def divide_factor(composite: str, factor: str) -> str:
    """
    Divide a factor out of a composite and return the cofactor.

    Args:
        composite: The composite number (as string)
        factor: The factor to divide out (as string)

    Returns:
        The cofactor as a string

    Raises:
        ValueError: If factor doesn't divide composite or inputs are invalid
    """
    if not validate_integer(composite) or not validate_integer(factor):
        raise ValueError("Invalid number format")

    if not verify_factor_divides(factor, composite):
        raise ValueError(f"Factor {factor} does not divide composite {composite}")

    composite_int = int(composite)
    factor_int = int(factor)

    cofactor = composite_int // factor_int
    return str(cofactor)


def is_probably_prime(n: str, trials: int = 10) -> bool:
    """
    Miller-Rabin primality test.

    Args:
        n: Number to test (as string)
        trials: Number of trials (default: 10, gives error probability < 2^-20)

    Returns:
        True if probably prime, False if definitely composite
    """
    if not validate_integer(n):
        return False

    n_int = int(n)

    # Handle small cases
    if n_int < 2:
        return False
    if n_int == 2 or n_int == 3:
        return True
    if n_int % 2 == 0:
        return False

    # Write n-1 as 2^r * d
    r, d = 0, n_int - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Witness loop
    for _ in range(trials):
        a = random.randrange(2, n_int - 1)
        x = pow(a, d, n_int)

        if x == 1 or x == n_int - 1:
            continue

        for _ in range(r - 1):
            x = pow(x, 2, n_int)
            if x == n_int - 1:
                break
        else:
            return False  # Definitely composite

    return True  # Probably prime


from typing import Tuple, Optional


def parse_sigma_with_parametrization(
    sigma_str: Optional[str],
    default_parametrization: int = 3
) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse a sigma string that may include parametrization prefix.

    ECM sigma values can be formatted as:
    - "123456" (plain sigma value)
    - "3:123456" (parametrization:sigma format)

    Args:
        sigma_str: Sigma string to parse, or None
        default_parametrization: Parametrization to use if not specified in sigma string (default: 3)

    Returns:
        Tuple of (sigma, parametrization) where:
        - sigma: The sigma value as a string (supports large numbers), or None if input was None
        - parametrization: The parametrization (0-3), or None if input was None

    Raises:
        ValueError: If parametrization is not in valid range (0-3)

    Examples:
        >>> parse_sigma_with_parametrization("3:123456")
        ('123456', 3)
        >>> parse_sigma_with_parametrization("987654")
        ('987654', 3)
        >>> parse_sigma_with_parametrization("1:42")
        ('42', 1)
        >>> parse_sigma_with_parametrization(None)
        (None, None)
    """
    if sigma_str is None:
        return None, None

    sigma_str = str(sigma_str)

    if ':' in sigma_str:
        parts = sigma_str.split(':', 1)
        parametrization = int(parts[0])
        sigma = parts[1]
    else:
        sigma = sigma_str
        parametrization = default_parametrization

    # Validate parametrization
    if parametrization not in [0, 1, 2, 3]:
        raise ValueError(
            f"Invalid parametrization {parametrization}. Must be 0, 1, 2, or 3."
        )

    return sigma, parametrization