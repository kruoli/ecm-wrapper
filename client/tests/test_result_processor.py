#!/usr/bin/env python3
"""
Unit tests for result_processor.py - factor deduplication and processing.

Tests deduplicate_factors() as a pure function and fully_factor_and_store()
with mocked wrapper dependencies.
"""
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from lib.result_processor import ResultProcessor


def make_processor(composite="1234567890123456789", **kwargs):
    """Create a ResultProcessor with a mocked wrapper."""
    wrapper = Mock()
    wrapper.logger = Mock()
    wrapper.log_factor_found = Mock()
    wrapper._fully_factor_found_result = Mock(side_effect=lambda f, quiet=False: [f])
    return ResultProcessor(
        wrapper=wrapper,
        composite=composite,
        method=kwargs.get("method", "ecm"),
        b1=kwargs.get("b1", 50000),
        b2=kwargs.get("b2", None),
        curves=kwargs.get("curves", 100),
        program=kwargs.get("program", "gmp-ecm"),
    )


# ============================================================
# deduplicate_factors
# ============================================================

class TestDeduplicateFactors:

    def test_no_duplicates(self):
        proc = make_processor()
        result = proc.deduplicate_factors([
            ("111", "3:100"),
            ("222", "3:200"),
        ])
        assert result == {"111": "3:100", "222": "3:200"}

    def test_with_duplicates(self):
        proc = make_processor()
        result = proc.deduplicate_factors([
            ("111", "3:100"),
            ("111", "3:200"),
            ("222", "3:300"),
        ])
        # First sigma wins for duplicates
        assert result == {"111": "3:100", "222": "3:300"}

    def test_empty_list(self):
        proc = make_processor()
        assert proc.deduplicate_factors([]) == {}

    def test_single_factor(self):
        proc = make_processor()
        result = proc.deduplicate_factors([("42", None)])
        assert result == {"42": None}

    def test_preserves_none_sigma(self):
        proc = make_processor()
        result = proc.deduplicate_factors([
            ("111", None),
            ("222", "3:200"),
        ])
        assert result["111"] is None
        assert result["222"] == "3:200"

    def test_first_sigma_wins_even_if_none(self):
        """If first occurrence has None sigma, keep None even if later has sigma."""
        proc = make_processor()
        result = proc.deduplicate_factors([
            ("111", None),
            ("111", "3:100"),
        ])
        assert result["111"] is None

    def test_many_duplicates(self):
        """Same factor from many workers."""
        proc = make_processor()
        factors = [("999", f"3:{i}") for i in range(10)]
        result = proc.deduplicate_factors(factors)
        assert len(result) == 1
        assert result["999"] == "3:0"  # First sigma


# ============================================================
# log_and_store_factors
# ============================================================

class TestLogAndStoreFactors:

    def test_basic_store(self):
        proc = make_processor()
        results = {}
        main = proc.log_and_store_factors([("111", "3:100")], results)

        assert main == "111"
        assert results['factor_found'] == "111"
        assert results['sigma'] == "3:100"
        assert results['factors_found'] == ["111"]
        assert results['factor_sigmas'] == {"111": "3:100"}

    def test_multiple_factors(self):
        proc = make_processor()
        results = {}
        main = proc.log_and_store_factors([
            ("111", "3:100"),
            ("222", "3:200"),
        ], results)

        assert main == "111"  # First factor
        assert "111" in results['factors_found']
        assert "222" in results['factors_found']
        assert results['factor_sigmas'] == {"111": "3:100", "222": "3:200"}

    def test_deduplicates_before_storing(self):
        proc = make_processor()
        results = {}
        proc.log_and_store_factors([
            ("111", "3:100"),
            ("111", "3:200"),
        ], results)

        assert results['factors_found'] == ["111"]

    def test_empty_factors(self):
        proc = make_processor()
        results = {}
        main = proc.log_and_store_factors([], results)
        assert main is None
        assert 'factor_found' not in results

    def test_calls_log_factor_found(self):
        proc = make_processor()
        results = {}
        proc.log_and_store_factors([("111", "3:100")], results)

        proc.wrapper.log_factor_found.assert_called_once_with(
            proc.composite, "111", proc.b1, proc.b2, proc.curves,
            method="ecm", sigma="3:100", program="gmp-ecm"
        )

    def test_quiet_skips_logging(self):
        proc = make_processor()
        results = {}
        proc.log_and_store_factors([("111", "3:100")], results, quiet=True)

        proc.wrapper.log_factor_found.assert_not_called()
        # But still stores
        assert results['factor_found'] == "111"

    def test_accumulates_into_existing_results(self):
        """If results already has factors, new ones are appended."""
        proc = make_processor()
        results = {
            'factors_found': ["000"],
            'factor_sigmas': {"000": None},
        }
        proc.log_and_store_factors([("111", "3:100")], results)

        assert "000" in results['factors_found']
        assert "111" in results['factors_found']


# ============================================================
# fully_factor_and_store
# ============================================================

class TestFullyFactorAndStore:

    def test_single_prime_factor(self):
        """Factor is already prime, cofactor is also prime."""
        # composite = 7 * 11 = 77
        proc = make_processor(composite="77")
        proc.wrapper._fully_factor_found_result = Mock(return_value=["7"])

        results = {}
        with patch('lib.result_processor.is_probably_prime', return_value=True):
            primes = proc.fully_factor_and_store(["7"], results)

        # 77 / 7 = 11 (prime cofactor)
        assert "7" in primes
        assert "11" in primes
        assert results['factors_found'] == primes
        assert results['ecm_found_factors'] == ["7"]
        assert results['cofactor_primes'] == ["11"]

    def test_empty_factors(self):
        proc = make_processor()
        results = {}
        primes = proc.fully_factor_and_store([], results)
        assert primes == []

    def test_deduplicates_input(self):
        """Duplicate factors in input should be deduplicated."""
        proc = make_processor(composite="49")  # 7^2
        proc.wrapper._fully_factor_found_result = Mock(return_value=["7"])

        results = {}
        with patch('lib.result_processor.is_probably_prime', return_value=True):
            primes = proc.fully_factor_and_store(["7", "7"], results)

        # Should only process "7" once (deduplicated)
        proc.wrapper._fully_factor_found_result.assert_called_once_with("7", quiet=True)

    def test_composite_cofactor_small_auto_factors(self):
        """Small composite cofactor (< 60 digits) gets auto-factored."""
        # composite = 3 * 5 * 7 = 105
        proc = make_processor(composite="105")
        proc.wrapper._fully_factor_found_result = Mock(
            side_effect=lambda f, quiet=False: [f] if f in ["3"] else ["5", "7"]
        )

        results = {}
        with patch('lib.result_processor.is_probably_prime', return_value=False):
            primes = proc.fully_factor_and_store(["3"], results)

        # 105 / 3 = 35 (composite, < 60 digits -> auto-factor)
        assert "3" in primes
        # The cofactor 35 should have been auto-factored
        assert results['cofactor_primes'] == ["5", "7"]

    def test_composite_cofactor_large_skipped(self):
        """Large composite cofactor (>= 60 digits) is not auto-factored."""
        large_composite = "3" * 60  # 60-digit number
        composite_str = str(int(large_composite) * 7)
        proc = make_processor(composite=composite_str)
        proc.wrapper._fully_factor_found_result = Mock(return_value=["7"])

        results = {}
        with patch('lib.result_processor.is_probably_prime', return_value=False):
            primes = proc.fully_factor_and_store(["7"], results)

        # Only the ECM-found factor should be in results
        assert "7" in primes
        assert results['cofactor_primes'] == []

    def test_cofactor_is_one(self):
        """When factor equals composite, cofactor is 1 - no further processing."""
        proc = make_processor(composite="7")
        proc.wrapper._fully_factor_found_result = Mock(return_value=["7"])

        results = {}
        primes = proc.fully_factor_and_store(["7"], results)

        assert primes == ["7"]
        assert results['cofactor_primes'] == []

    def test_logging_not_quiet(self):
        """Non-quiet mode should log ECM-found factors."""
        proc = make_processor(composite="77")
        proc.wrapper._fully_factor_found_result = Mock(return_value=["7"])

        results = {'sigma': '3:42'}
        with patch('lib.result_processor.is_probably_prime', return_value=True):
            proc.fully_factor_and_store(["7"], results, quiet=False)

        proc.wrapper.log_factor_found.assert_called()

    def test_logging_quiet(self):
        """Quiet mode should not log anything."""
        proc = make_processor(composite="77")
        proc.wrapper._fully_factor_found_result = Mock(return_value=["7"])

        results = {}
        with patch('lib.result_processor.is_probably_prime', return_value=True):
            proc.fully_factor_and_store(["7"], results, quiet=True)

        proc.wrapper.log_factor_found.assert_not_called()

    def test_multiple_ecm_factors(self):
        """Multiple factors found by ECM, all prime."""
        # composite = 3 * 5 * 7 = 105
        proc = make_processor(composite="105")
        proc.wrapper._fully_factor_found_result = Mock(
            side_effect=lambda f, quiet=False: [f]
        )

        results = {}
        with patch('lib.result_processor.is_probably_prime', return_value=True):
            primes = proc.fully_factor_and_store(["3", "5"], results)

        # 105 / 3 / 5 = 7 (prime cofactor)
        assert "3" in primes
        assert "5" in primes
        assert "7" in primes
        assert results['ecm_found_factors'] == ["3", "5"]
        assert results['cofactor_primes'] == ["7"]

    def test_sets_factor_found_to_first(self):
        """results['factor_found'] should be first prime in list."""
        proc = make_processor(composite="77")
        proc.wrapper._fully_factor_found_result = Mock(return_value=["7"])

        results = {}
        with patch('lib.result_processor.is_probably_prime', return_value=True):
            proc.fully_factor_and_store(["7"], results)

        assert results['factor_found'] == "7"
