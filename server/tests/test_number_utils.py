"""
Unit tests for number utility functions.

Tests cover:
- Factor division
- Primality testing (Miller-Rabin)
- Factor verification
"""
import pytest
from app.utils.number_utils import (
    divide_factor,
    is_probably_prime,
    verify_factor_divides,
    validate_integer,
    verify_complete_factorization
)


class TestDivideFactor:
    """Tests for divide_factor function."""

    def test_simple_division(self):
        """Test dividing a simple composite."""
        composite = "123456789012345"
        factor = "3"
        cofactor = divide_factor(composite, factor)
        assert cofactor == "41152263004115"
        # Verify: 3 * 41152263004115 = 123456789012345
        assert int(factor) * int(cofactor) == int(composite)

    def test_large_factor(self):
        """Test dividing with a large factor."""
        composite = "1000000000000000000000000000000"  # 10^30
        factor = "1000000000000000"  # 10^15
        cofactor = divide_factor(composite, factor)
        assert cofactor == "1000000000000000"  # 10^15
        assert int(factor) * int(cofactor) == int(composite)

    def test_prime_factorization(self):
        """Test dividing a number to get a prime cofactor."""
        composite = "35"
        factor = "5"
        cofactor = divide_factor(composite, factor)
        assert cofactor == "7"
        assert is_probably_prime(cofactor)

    def test_invalid_factor_raises_error(self):
        """Test that dividing by a non-factor raises ValueError."""
        composite = "100"
        factor = "7"  # Doesn't divide 100
        with pytest.raises(ValueError, match="does not divide"):
            divide_factor(composite, factor)

    def test_invalid_format_raises_error(self):
        """Test that invalid number formats raise ValueError."""
        with pytest.raises(ValueError, match="Invalid number format"):
            divide_factor("abc", "3")
        with pytest.raises(ValueError, match="Invalid number format"):
            divide_factor("100", "3.5")


class TestIsProbablyPrime:
    """Tests for Miller-Rabin primality test."""

    def test_small_primes(self):
        """Test small known primes."""
        primes = ["2", "3", "5", "7", "11", "13", "17", "19", "23", "29", "31"]
        for p in primes:
            assert is_probably_prime(p), f"{p} should be prime"

    def test_small_composites(self):
        """Test small known composites."""
        composites = ["4", "6", "8", "9", "10", "12", "14", "15", "16", "18", "20"]
        for c in composites:
            assert not is_probably_prime(c), f"{c} should be composite"

    def test_edge_cases(self):
        """Test edge cases."""
        assert not is_probably_prime("0")
        assert not is_probably_prime("1")
        assert is_probably_prime("2")

    def test_large_primes(self):
        """Test larger known primes."""
        # Known Mersenne primes
        assert is_probably_prime("2147483647")  # 2^31 - 1
        assert is_probably_prime("8191")  # 2^13 - 1

    def test_large_composites(self):
        """Test larger known composites."""
        assert not is_probably_prime("1000000000000")
        assert not is_probably_prime("123456789012345")  # Divisible by 3

    def test_carmichael_numbers(self):
        """Test Carmichael numbers (pseudoprimes that fool some tests)."""
        # 561 is the smallest Carmichael number
        assert not is_probably_prime("561")
        # Other Carmichael numbers
        assert not is_probably_prime("1105")
        assert not is_probably_prime("1729")

    def test_invalid_input(self):
        """Test that invalid inputs return False."""
        assert not is_probably_prime("abc")
        assert not is_probably_prime("3.14")
        assert not is_probably_prime("")


class TestVerifyFactorDivides:
    """Tests for factor verification."""

    def test_valid_factors(self):
        """Test valid factor-composite pairs."""
        assert verify_factor_divides("2", "100")
        assert verify_factor_divides("5", "100")
        assert verify_factor_divides("10", "100")
        assert verify_factor_divides("25", "100")
        assert verify_factor_divides("100", "100")  # Number divides itself

    def test_invalid_factors(self):
        """Test invalid factor-composite pairs."""
        assert not verify_factor_divides("3", "100")
        assert not verify_factor_divides("7", "100")
        assert not verify_factor_divides("101", "100")  # Factor > composite

    def test_trivial_cases(self):
        """Test trivial cases."""
        assert verify_factor_divides("1", "100")  # 1 divides everything
        assert verify_factor_divides("1", "12345678901234567890")

    def test_large_numbers(self):
        """Test with large numbers."""
        composite = "123456789012345678901234567890"
        factor = "10"
        assert verify_factor_divides(factor, composite)

    def test_invalid_inputs(self):
        """Test invalid inputs."""
        assert not verify_factor_divides("abc", "100")
        assert not verify_factor_divides("3", "xyz")
        assert not verify_factor_divides("3.5", "100")


class TestValidateInteger:
    """Tests for integer validation."""

    def test_valid_integers(self):
        """Test valid integer strings."""
        assert validate_integer("0")
        assert validate_integer("1")
        assert validate_integer("123")
        assert validate_integer("999999999999999999999999999999")

    def test_invalid_integers(self):
        """Test invalid integer strings."""
        assert not validate_integer("abc")
        assert not validate_integer("3.14")
        assert not validate_integer("1e10")
        assert not validate_integer("-5")
        assert not validate_integer("")
        assert not validate_integer("01")  # Leading zeros
        assert not validate_integer("00123")  # Leading zeros

    def test_non_string_input(self):
        """Test non-string inputs."""
        assert not validate_integer(123)
        assert not validate_integer(None)
        assert not validate_integer([])


class TestVerifyCompleteFactorization:
    """Tests for complete factorization verification."""

    def test_simple_factorization(self):
        """Test simple complete factorizations."""
        assert verify_complete_factorization("6", ["2", "3"])
        assert verify_complete_factorization("100", ["2", "2", "5", "5"])
        assert verify_complete_factorization("30", ["2", "3", "5"])

    def test_prime_factorization(self):
        """Test factorization of a prime."""
        assert verify_complete_factorization("17", ["17"])

    def test_large_factorization(self):
        """Test large number factorization."""
        composite = "1000000000000"  # 10^12 = 2^12 * 5^12
        factors = ["2"] * 12 + ["5"] * 12
        assert verify_complete_factorization(composite, factors)

    def test_incomplete_factorization(self):
        """Test incomplete factorizations."""
        assert not verify_complete_factorization("100", ["2", "5"])  # Missing factors (2*5=10, not 100)
        assert not verify_complete_factorization("100", ["2", "2", "2", "5"])  # Wrong factorization (8*5=40, not 100)
        # Note: The function verifies product correctness, not primality of factors
        assert verify_complete_factorization("100", ["10", "10"])  # 10*10=100 (correct product)

    def test_empty_factors(self):
        """Test empty factor list."""
        assert not verify_complete_factorization("100", [])

    def test_invalid_factors(self):
        """Test invalid factors."""
        assert not verify_complete_factorization("100", ["2", "abc"])
        assert not verify_complete_factorization("100", ["2", "3.5"])


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_factor_divide_verify_loop(self):
        """Test dividing out factors iteratively."""
        composite = "1000"
        factors_to_divide = ["2", "2", "2", "5", "5", "5"]

        current = composite
        for factor in factors_to_divide:
            assert verify_factor_divides(factor, current)
            current = divide_factor(current, factor)

        # After dividing out all factors, should be left with 1
        assert current == "1"

    def test_factor_until_prime(self):
        """Test dividing factors until a prime cofactor remains."""
        composite = "35"
        factor = "5"

        cofactor = divide_factor(composite, factor)
        assert cofactor == "7"
        assert is_probably_prime(cofactor)

    def test_complete_factorization_workflow(self):
        """Test complete factorization workflow."""
        composite = "210"  # 2 * 3 * 5 * 7
        factors = ["2", "3", "5", "7"]

        # Verify complete factorization
        assert verify_complete_factorization(composite, factors)

        # Divide out each factor
        current = composite
        for factor in factors[:-1]:  # All but the last
            current = divide_factor(current, factor)

        # Last cofactor should be the remaining prime
        assert current == factors[-1]
        assert is_probably_prime(current)
