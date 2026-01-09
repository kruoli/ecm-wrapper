#!/usr/bin/env python3
"""
Unit tests for ecm_executor module.

Tests core logic without requiring actual ECM binaries:
- Multiprocess result processing (process_result function logic)
- Queue draining fix (H-4 - ensures no results are lost when processes exit)
- Repeated factor division fix (H-5 - handles factors like p^2)
- Factor deduplication across workers
"""
import sys
import queue
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


class TestMultiprocessResultProcessing:
    """Tests for multiprocess result processing logic.

    These tests verify the logic used in run_multiprocess_v2's inner
    process_result function without actually running subprocesses.
    """

    def test_result_processing_accumulates_curves(self):
        """Test that curve counts from multiple workers accumulate correctly."""
        total_curves_completed = 0
        all_factors = []

        def process_result(result: dict):
            nonlocal total_curves_completed
            total_curves_completed += result['curves_completed']

            if result['factor_found'] and result['factor_found'] not in all_factors:
                all_factors.append(result['factor_found'])

        # Simulate 4 workers each completing 25 curves
        for worker_id in range(1, 5):
            process_result({
                'worker_id': worker_id,
                'curves_completed': 25,
                'factor_found': None
            })

        assert total_curves_completed == 100
        assert all_factors == []

    def test_result_processing_deduplicates_factors(self):
        """Test that duplicate factors from different workers are deduplicated."""
        all_factors = []
        all_sigmas = []

        def process_result(result: dict):
            if result['factor_found']:
                factor = result['factor_found']
                if factor not in all_factors:
                    all_factors.append(factor)
                    all_sigmas.append(result.get('sigma_found'))

        # Two workers find the same factor
        process_result({
            'worker_id': 1,
            'curves_completed': 10,
            'factor_found': '12345',
            'sigma_found': '3:111'
        })

        process_result({
            'worker_id': 2,
            'curves_completed': 15,
            'factor_found': '12345',  # Same factor
            'sigma_found': '3:222'
        })

        # Should only have one entry
        assert all_factors == ['12345']
        assert all_sigmas == ['3:111']  # First sigma wins

    def test_result_processing_handles_multiple_different_factors(self):
        """Test that different factors from workers are all recorded."""
        all_factors = []

        def process_result(result: dict):
            if result['factor_found'] and result['factor_found'] not in all_factors:
                all_factors.append(result['factor_found'])

        process_result({'worker_id': 1, 'curves_completed': 10, 'factor_found': '3'})
        process_result({'worker_id': 2, 'curves_completed': 10, 'factor_found': '7'})
        process_result({'worker_id': 3, 'curves_completed': 10, 'factor_found': '11'})

        assert sorted(all_factors) == ['11', '3', '7']


class TestQueueDrainingLogic:
    """Tests for H-4 fix: Queue draining when processes exit.

    The fix ensures that when all worker processes have exited,
    we drain any remaining results from the queue before breaking
    out of the result collection loop.
    """

    def test_queue_drain_collects_remaining_results(self):
        """Test that queue draining doesn't lose results."""
        # Simulate a queue with results that arrived after process exit check
        result_queue = queue.Queue()
        result_queue.put({'worker_id': 1, 'curves_completed': 25, 'factor_found': None})
        result_queue.put({'worker_id': 2, 'curves_completed': 25, 'factor_found': None})

        # Simulate the draining logic from run_multiprocess_v2
        drained_results = []
        while True:
            try:
                result = result_queue.get_nowait()
                drained_results.append(result)
            except queue.Empty:
                break

        assert len(drained_results) == 2
        assert sum(r['curves_completed'] for r in drained_results) == 50

    def test_empty_queue_drain_handles_gracefully(self):
        """Test that draining an empty queue doesn't raise."""
        result_queue = queue.Queue()

        drained_results = []
        while True:
            try:
                result = result_queue.get_nowait()
                drained_results.append(result)
            except queue.Empty:
                break

        assert drained_results == []

    def test_queue_drain_preserves_factor_info(self):
        """Test that factor information is preserved during queue drain."""
        result_queue = queue.Queue()
        result_queue.put({
            'worker_id': 1,
            'curves_completed': 10,
            'factor_found': '7919',
            'sigma_found': '3:12345'
        })

        drained_results = []
        while True:
            try:
                result = result_queue.get_nowait()
                drained_results.append(result)
            except queue.Empty:
                break

        assert len(drained_results) == 1
        assert drained_results[0]['factor_found'] == '7919'
        assert drained_results[0]['sigma_found'] == '3:12345'


class TestRepeatedFactorDivision:
    """Tests for H-5 fix: Safe division with repeated factors.

    The fix changes:
        current_cofactor //= int(prime)
    to:
        while current_cofactor % int(prime) == 0:
            current_cofactor //= int(prime)

    This ensures that if a factor appears multiple times (e.g., p^2),
    it is fully divided out.
    """

    def test_single_factor_division(self):
        """Test division of a factor that appears once."""
        composite = 21  # 3 * 7
        factor = 3

        cofactor = composite
        while cofactor % factor == 0:
            cofactor //= factor

        assert cofactor == 7

    def test_repeated_factor_division(self):
        """Test division of a factor that appears multiple times."""
        composite = 27  # 3^3
        factor = 3

        cofactor = composite
        while cofactor % factor == 0:
            cofactor //= factor

        assert cofactor == 1  # 3 fully divided out

    def test_repeated_factor_with_other_factors(self):
        """Test division when composite has repeated factor plus others."""
        composite = 72  # 2^3 * 3^2 = 8 * 9
        factor = 2

        cofactor = composite
        while cofactor % factor == 0:
            cofactor //= factor

        assert cofactor == 9  # Only 3^2 remains

    def test_factor_not_present(self):
        """Test division when factor doesn't divide composite."""
        composite = 35  # 5 * 7
        factor = 3

        cofactor = composite
        while cofactor % factor == 0:
            cofactor //= factor

        assert cofactor == 35  # Unchanged

    def test_multiple_factors_division(self):
        """Test dividing out multiple factors from a composite."""
        composite = 360  # 2^3 * 3^2 * 5 = 8 * 9 * 5
        factors = [2, 3, 5]

        cofactor = composite
        for factor in factors:
            while cofactor % factor == 0:
                cofactor //= factor

        assert cofactor == 1  # Fully factored

    def test_partial_factorization(self):
        """Test partial factorization leaves unfactored portion."""
        composite = 2310  # 2 * 3 * 5 * 7 * 11
        factors = [2, 3]

        cofactor = composite
        for factor in factors:
            while cofactor % factor == 0:
                cofactor //= factor

        assert cofactor == 385  # 5 * 7 * 11

    def test_string_factor_conversion(self):
        """Test that string factors are properly converted to int."""
        composite = 100  # 2^2 * 5^2
        sub_primes = ['2', '5']  # Factors as strings

        current_cofactor = composite
        for prime in sub_primes:
            while current_cofactor % int(prime) == 0:
                current_cofactor //= int(prime)

        assert current_cofactor == 1


class TestFactorResultBuilding:
    """Tests for FactorResult building from multiprocess results."""

    def test_factor_result_from_primitive(self):
        """Test FactorResult.from_primitive_result works correctly."""
        from lib.ecm_config import FactorResult

        # from_primitive_result expects keys from _execute_ecm_primitive output format
        prim_result = {
            'success': True,
            'factors': ['12345'],
            'sigmas': ['3:111'],
            'curves_completed': 50,
            'raw_output': 'Test output',
            'parametrization': 3
        }

        result = FactorResult.from_primitive_result(prim_result, execution_time=10.5)

        assert '12345' in result.factors
        assert result.curves_run == 50
        assert result.execution_time == 10.5
        assert result.success is True

    def test_factor_result_no_factor(self):
        """Test FactorResult when no factor is found."""
        from lib.ecm_config import FactorResult

        prim_result = {
            'success': False,
            'factors': [],
            'sigmas': [],
            'curves_completed': 100,
            'raw_output': 'No factor found'
        }

        result = FactorResult.from_primitive_result(prim_result, execution_time=30.0)

        assert result.factors == []
        assert result.curves_run == 100
        assert result.execution_time == 30.0

    def test_factor_result_add_factor(self):
        """Test adding factors to FactorResult."""
        from lib.ecm_config import FactorResult

        result = FactorResult()
        result.add_factor('3', '3:111')
        result.add_factor('7', '3:222')

        assert '3' in result.factors
        assert '7' in result.factors
        assert len(result.factors) == 2


class TestTrialDivision:
    """Tests for trial division helper function."""

    def test_trial_division_small_primes(self):
        """Test trial division finds small prime factors."""
        from lib.ecm_math import trial_division

        primes, cofactor = trial_division(30, limit=100)  # 2 * 3 * 5

        assert sorted(primes) == [2, 3, 5]
        assert cofactor == 1

    def test_trial_division_with_cofactor(self):
        """Test trial division returns cofactor when not fully factored."""
        from lib.ecm_math import trial_division

        # 1001 = 7 * 11 * 13, but with limit=10 we only find 7
        primes, cofactor = trial_division(1001, limit=10)

        assert 7 in primes
        assert cofactor == 143  # 11 * 13

    def test_trial_division_prime_input(self):
        """Test trial division with prime input."""
        from lib.ecm_math import trial_division

        primes, cofactor = trial_division(97, limit=100)  # 97 is prime

        assert primes == []
        assert cofactor == 97

    def test_trial_division_repeated_factors(self):
        """Test trial division handles repeated factors."""
        from lib.ecm_math import trial_division

        primes, cofactor = trial_division(72, limit=100)  # 2^3 * 3^2

        # Should find all instances
        assert primes.count(2) == 3
        assert primes.count(3) == 2
        assert cofactor == 1


class TestPrimalityCheck:
    """Tests for is_probably_prime function."""

    def test_small_primes(self):
        """Test small primes are correctly identified."""
        from lib.ecm_math import is_probably_prime

        assert is_probably_prime(2) is True
        assert is_probably_prime(3) is True
        assert is_probably_prime(5) is True
        assert is_probably_prime(7) is True
        assert is_probably_prime(97) is True

    def test_small_composites(self):
        """Test small composites are correctly identified."""
        from lib.ecm_math import is_probably_prime

        assert is_probably_prime(4) is False
        assert is_probably_prime(9) is False
        assert is_probably_prime(15) is False
        assert is_probably_prime(21) is False

    def test_one_is_not_prime(self):
        """Test that 1 is not considered prime."""
        from lib.ecm_math import is_probably_prime

        assert is_probably_prime(1) is False

    def test_large_prime(self):
        """Test a larger prime number."""
        from lib.ecm_math import is_probably_prime

        # 7919 is prime
        assert is_probably_prime(7919) is True

    def test_carmichael_number(self):
        """Test a Carmichael number (composite that fools some tests)."""
        from lib.ecm_math import is_probably_prime

        # 561 = 3 * 11 * 17 is the smallest Carmichael number
        # Miller-Rabin should correctly identify it as composite
        assert is_probably_prime(561) is False


def main():
    """Run all tests."""
    print("Running ecm_executor unit tests...\n")
    pytest.main([__file__, '-v'])


if __name__ == '__main__':
    main()
