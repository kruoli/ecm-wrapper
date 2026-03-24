#!/usr/bin/env python3
"""
Unit tests for pure functions in ecm_math.py.

Tests only the deterministic, I/O-free functions. Functions that call
the t-level binary (calculate_tlevel, calculate_curves_to_target_direct, etc.)
are excluded - those require integration tests with the actual binary.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from lib.ecm_math import (
    trial_division,
    is_probably_prime,
    get_b1_for_digit_length,
    get_optimal_b1_for_tlevel,
    get_b1_above_tlevel,
    get_max_tlevel_for_workers,
    calculate_target_tlevel,
    OPTIMAL_B1_TABLE,
)


# ============================================================
# get_b1_for_digit_length
# ============================================================

class TestGetB1ForDigitLength:
    """Test the digit-length to B1 lookup chain."""

    @pytest.mark.parametrize("digits,expected_b1", [
        (10, 2000),     # <= 20
        (20, 2000),     # boundary: exactly 20
        (21, 11000),    # 21-25 range
        (25, 11000),    # boundary: exactly 25
        (26, 50000),    # 26-30
        (30, 50000),
        (31, 250000),   # 31-35
        (35, 250000),
        (36, 1000000),  # 36-40
        (40, 1000000),
        (41, 3000000),  # 41-45
        (45, 3000000),
        (46, 11000000),  # 46-50
        (50, 11000000),
        (51, 43000000),  # 51-55
        (55, 43000000),
        (56, 110000000),  # 56-60
        (60, 110000000),
        (61, 260000000),  # 61-65
        (65, 260000000),
        (66, 850000000),  # > 65
        (100, 850000000),
        (200, 850000000),
    ])
    def test_digit_boundaries(self, digits, expected_b1):
        assert get_b1_for_digit_length(digits) == expected_b1

    def test_single_digit(self):
        assert get_b1_for_digit_length(1) == 2000

    def test_monotonically_increasing(self):
        """B1 should never decrease as digits increase."""
        prev = 0
        for d in range(1, 120):
            b1 = get_b1_for_digit_length(d)
            assert b1 >= prev, f"B1 decreased at {d} digits: {prev} -> {b1}"
            prev = b1


# ============================================================
# get_optimal_b1_for_tlevel
# ============================================================

class TestGetOptimalB1ForTlevel:
    """Test optimal B1 lookup from Zimmermann's table."""

    def test_exact_table_entries(self):
        """Each table entry's t-level should return that entry's B1 and curves."""
        for digits, b1, curves in OPTIMAL_B1_TABLE:
            result_b1, result_curves = get_optimal_b1_for_tlevel(float(digits))
            assert result_b1 == b1
            assert result_curves == curves

    def test_below_table(self):
        """T-level below lowest table entry returns first entry."""
        b1, curves = get_optimal_b1_for_tlevel(5.0)
        assert b1 == OPTIMAL_B1_TABLE[0][1]
        assert curves == OPTIMAL_B1_TABLE[0][2]

    def test_between_entries(self):
        """T-level between entries should return the next higher entry."""
        b1, curves = get_optimal_b1_for_tlevel(27.5)  # Between 25 and 30
        assert b1 == 250000   # t30 entry
        assert curves == 513

    def test_above_table(self):
        """T-level above highest entry returns last entry."""
        b1, curves = get_optimal_b1_for_tlevel(100.0)
        assert b1 == OPTIMAL_B1_TABLE[-1][1]
        assert curves == OPTIMAL_B1_TABLE[-1][2]

    def test_zero_tlevel(self):
        b1, curves = get_optimal_b1_for_tlevel(0.0)
        assert b1 == OPTIMAL_B1_TABLE[0][1]

    def test_just_above_entry(self):
        """t-level 20.01 should return t25 entry."""
        b1, _ = get_optimal_b1_for_tlevel(20.01)
        assert b1 == 50000  # t25

    def test_just_below_entry(self):
        """t-level 19.99 should still return t20 entry."""
        b1, _ = get_optimal_b1_for_tlevel(19.99)
        assert b1 == 11000  # t20


# ============================================================
# get_b1_above_tlevel
# ============================================================

class TestGetB1AboveTlevel:
    """Test 'one step above' B1 lookup for PM1/PP1 sweeps."""

    def test_one_step_above(self):
        """t20 -> returns t25's B1."""
        assert get_b1_above_tlevel(20.0) == 50000  # t25 B1

    def test_between_entries_gets_step_above_next(self):
        """t22 -> next >= 22 is t25, one above is t30."""
        assert get_b1_above_tlevel(22.0) == 250000  # t30 B1

    def test_exact_entry_48(self):
        """t48 -> next >= 48 is t50, one above is t55."""
        assert get_b1_above_tlevel(48.0) == 110000000  # t55

    def test_exact_entry_55(self):
        """t55 -> next >= 55 is t55, one above is t60."""
        assert get_b1_above_tlevel(55.0) == 260000000  # t60

    def test_highest_entry_returns_itself(self):
        """At the last table entry, there's no 'next' - returns the entry itself."""
        last_digits, last_b1, _ = OPTIMAL_B1_TABLE[-1]
        assert get_b1_above_tlevel(float(last_digits)) == last_b1

    def test_above_table_returns_last(self):
        """T-level above all entries returns last entry's B1."""
        assert get_b1_above_tlevel(100.0) == OPTIMAL_B1_TABLE[-1][1]

    def test_second_to_last_entry(self):
        """Penultimate entry should return last entry's B1."""
        second_last_digits = OPTIMAL_B1_TABLE[-2][0]
        last_b1 = OPTIMAL_B1_TABLE[-1][1]
        assert get_b1_above_tlevel(float(second_last_digits)) == last_b1

    def test_low_tlevel(self):
        """Very low t-level: next >= 5 is t20, one above is t25."""
        assert get_b1_above_tlevel(5.0) == 50000  # t25

    def test_zero_tlevel(self):
        """t0 -> next >= 0 is t20, one above is t25."""
        assert get_b1_above_tlevel(0.0) == 50000  # t25

    @pytest.mark.parametrize("tlevel", [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
    def test_all_exact_entries_return_next(self, tlevel):
        """Every exact table entry (except last) should return the next entry's B1."""
        result = get_b1_above_tlevel(float(tlevel))
        # Find this entry's index
        for i, (digits, b1, _) in enumerate(OPTIMAL_B1_TABLE):
            if digits == tlevel:
                if i + 1 < len(OPTIMAL_B1_TABLE):
                    assert result == OPTIMAL_B1_TABLE[i + 1][1]
                else:
                    assert result == b1  # Last entry returns itself
                break


# ============================================================
# get_max_tlevel_for_workers
# ============================================================

class TestGetMaxTlevelForWorkers:
    """Test adaptive CPU mode t-level caps."""

    def test_16_plus_workers(self):
        assert get_max_tlevel_for_workers(16) == 55.0
        assert get_max_tlevel_for_workers(32) == 55.0
        assert get_max_tlevel_for_workers(64) == 55.0

    def test_5_to_15_workers(self):
        assert get_max_tlevel_for_workers(5) == 50.0
        assert get_max_tlevel_for_workers(8) == 50.0
        assert get_max_tlevel_for_workers(15) == 50.0

    def test_3_to_4_workers(self):
        assert get_max_tlevel_for_workers(3) == 45.0
        assert get_max_tlevel_for_workers(4) == 45.0

    def test_1_to_2_workers(self):
        assert get_max_tlevel_for_workers(1) == 40.0
        assert get_max_tlevel_for_workers(2) == 40.0

    def test_boundary_values(self):
        """Test exact boundary transitions."""
        assert get_max_tlevel_for_workers(2) == 40.0
        assert get_max_tlevel_for_workers(3) == 45.0   # jumps at 3
        assert get_max_tlevel_for_workers(4) == 45.0
        assert get_max_tlevel_for_workers(5) == 50.0   # jumps at 5
        assert get_max_tlevel_for_workers(15) == 50.0
        assert get_max_tlevel_for_workers(16) == 55.0  # jumps at 16

    def test_monotonically_increasing(self):
        """More workers should never reduce the max t-level."""
        prev = 0.0
        for w in range(1, 100):
            t = get_max_tlevel_for_workers(w)
            assert t >= prev
            prev = t


# ============================================================
# calculate_target_tlevel
# ============================================================

class TestCalculateTargetTlevel:
    """Test the 4/13 rule for target t-level calculation."""

    def test_65_digits(self):
        assert calculate_target_tlevel(65) == pytest.approx(20.0, abs=0.01)

    def test_100_digits(self):
        assert calculate_target_tlevel(100) == pytest.approx(30.769, abs=0.01)

    def test_130_digits(self):
        """130 * 4/13 = 40.0"""
        assert calculate_target_tlevel(130) == pytest.approx(40.0, abs=0.01)

    def test_small_composite(self):
        assert calculate_target_tlevel(20) == pytest.approx(6.15, abs=0.01)

    def test_zero_digits(self):
        assert calculate_target_tlevel(0) == 0.0

    def test_linearity(self):
        """4/13 rule is linear - doubling digits doubles target."""
        t50 = calculate_target_tlevel(50)
        t100 = calculate_target_tlevel(100)
        assert t100 == pytest.approx(t50 * 2, abs=0.001)


# ============================================================
# trial_division - additional edge cases
# ============================================================

class TestTrialDivisionExtended:
    """Extended tests beyond what test_ecm_executor_unit.py covers."""

    def test_input_of_one(self):
        factors, cofactor = trial_division(1)
        assert factors == []
        assert cofactor == 1

    def test_power_of_two(self):
        factors, cofactor = trial_division(1024)  # 2^10
        assert factors == [2] * 10
        assert cofactor == 1

    def test_large_prime_as_cofactor(self):
        """A prime larger than sqrt(limit) should remain as cofactor."""
        # 7919 is prime, limit=100 means we check up to 100
        factors, cofactor = trial_division(7919, limit=100)
        assert factors == []
        assert cofactor == 7919

    def test_semiprime(self):
        """Product of two primes - smaller found by trial, larger stays as cofactor."""
        # 143 = 11 * 13; after dividing by 11, cofactor=13 and 13^2 > 13 stops loop
        factors, cofactor = trial_division(143, limit=100)
        assert factors == [11]
        assert cofactor == 13

    def test_wheel_factorization_coverage(self):
        """Test that wheel correctly finds factors beyond 5."""
        # 7 * 11 * 13 = 1001; after 7 and 11, cofactor=13 stays
        factors, cofactor = trial_division(1001, limit=100)
        assert sorted(factors) == [7, 11]
        assert cofactor == 13

    def test_perfect_square(self):
        factors, cofactor = trial_division(49)  # 7^2
        assert factors == [7, 7]
        assert cofactor == 1

    def test_limit_stops_early(self):
        """With low limit, large factors stay in cofactor."""
        # 11 * 97 = 1067
        factors, cofactor = trial_division(1067, limit=10)
        assert factors == []
        assert cofactor == 1067  # Neither factor < limit

    def test_limit_boundary(self):
        """Factor at exactly the limit should be found (i <= limit check)."""
        # 7 * 11 = 77; limit=7 should find 7
        factors, cofactor = trial_division(77, limit=7)
        assert 7 in factors
        assert cofactor == 11

    def test_smooth_number(self):
        """Highly composite number - 7 stays as cofactor since 7^2 > 7."""
        # 2^4 * 3^2 * 5 * 7 = 5040; after 2,3,5 divides out, cofactor=7
        factors, cofactor = trial_division(5040, limit=100)
        assert factors.count(2) == 4
        assert factors.count(3) == 2
        assert factors.count(5) == 1
        assert cofactor == 7

    def test_all_factors_found_when_repeated(self):
        """When factors repeat enough, i*i <= cofactor stays true."""
        # 2^10 = 1024; 2 divides cleanly to cofactor=1
        factors, cofactor = trial_division(1024, limit=100)
        assert factors == [2] * 10
        assert cofactor == 1

    def test_large_semiprime_beyond_limit(self):
        """Two primes both larger than sqrt(default limit)."""
        p1, p2 = 10000019, 10000079  # Both > 10^7
        n = p1 * p2
        factors, cofactor = trial_division(n)
        # Neither prime should be found (both > limit)
        assert factors == []
        assert cofactor == n


# ============================================================
# is_probably_prime - additional edge cases
# ============================================================

class TestIsProbablyPrimeExtended:
    """Extended primality tests beyond what test_ecm_executor_unit.py covers."""

    def test_zero(self):
        assert is_probably_prime(0) is False

    def test_negative(self):
        assert is_probably_prime(-7) is False

    def test_two(self):
        assert is_probably_prime(2) is True

    def test_three(self):
        assert is_probably_prime(3) is True

    def test_four(self):
        assert is_probably_prime(4) is False

    def test_even_numbers(self):
        for n in [6, 8, 10, 100, 1000]:
            assert is_probably_prime(n) is False

    @pytest.mark.parametrize("p", [
        5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
        53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    ])
    def test_primes_under_100(self, p):
        assert is_probably_prime(p) is True

    @pytest.mark.parametrize("n", [
        561,    # 3 * 11 * 17
        1105,   # 5 * 13 * 17
        1729,   # 7 * 13 * 19 (Hardy-Ramanujan / taxicab number)
        2465,   # 5 * 17 * 29
        2821,   # 7 * 13 * 31
        6601,   # 7 * 23 * 41
        8911,   # 7 * 19 * 67
    ])
    def test_carmichael_numbers(self, n):
        """Miller-Rabin with 10 trials should catch all small Carmichael numbers."""
        assert is_probably_prime(n) is False

    def test_large_known_prime(self):
        """Test a known large prime."""
        # Mersenne prime 2^61 - 1
        assert is_probably_prime(2305843009213693951) is True

    def test_large_known_composite(self):
        """Product of two large primes."""
        assert is_probably_prime(7919 * 7927) is False

    def test_mersenne_composite(self):
        """2^67 - 1 is not prime (factors: 193707721 * 761838257287)."""
        assert is_probably_prime(2**67 - 1) is False


# ============================================================
# OPTIMAL_B1_TABLE consistency
# ============================================================

class TestOptimalB1Table:
    """Verify the table itself is consistent."""

    def test_sorted_by_digits(self):
        digits = [entry[0] for entry in OPTIMAL_B1_TABLE]
        assert digits == sorted(digits)

    def test_b1_monotonically_increasing(self):
        b1s = [entry[1] for entry in OPTIMAL_B1_TABLE]
        for i in range(1, len(b1s)):
            assert b1s[i] > b1s[i - 1]

    def test_curves_monotonically_increasing(self):
        curvess = [entry[2] for entry in OPTIMAL_B1_TABLE]
        for i in range(1, len(curvess)):
            assert curvess[i] > curvess[i - 1]

    def test_five_digit_increments(self):
        """Table entries should be in 5-digit increments."""
        digits = [entry[0] for entry in OPTIMAL_B1_TABLE]
        for i in range(1, len(digits)):
            assert digits[i] - digits[i - 1] == 5
