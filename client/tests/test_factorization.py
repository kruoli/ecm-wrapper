#!/usr/bin/env python3
"""
Unit tests for ECM and YAFU factorization parsing.

Tests that we correctly parse factors and handle duplicates properly.
"""
import unittest
from collections import Counter
from typing import List, Tuple
import sys
import importlib.util
from pathlib import Path

# Get the parent directory (client/) to import wrappers
client_dir = Path(__file__).parent.parent

# Import ecm-wrapper.py
ecm_wrapper_path = client_dir / "ecm-wrapper.py"
spec = importlib.util.spec_from_file_location("ecm_wrapper", str(ecm_wrapper_path))
if spec is None or spec.loader is None:
    raise ImportError(f"Failed to load ecm-wrapper.py from {ecm_wrapper_path}")
ecm_module = importlib.util.module_from_spec(spec)
sys.modules["ecm_wrapper"] = ecm_module
spec.loader.exec_module(ecm_module)
ECMWrapper = ecm_module.ECMWrapper

# Import yafu-wrapper.py
yafu_wrapper_path = client_dir / "yafu-wrapper.py"
spec = importlib.util.spec_from_file_location("yafu_wrapper", str(yafu_wrapper_path))
if spec is None or spec.loader is None:
    raise ImportError(f"Failed to load yafu-wrapper.py from {yafu_wrapper_path}")
yafu_module = importlib.util.module_from_spec(spec)
sys.modules["yafu_wrapper"] = yafu_module
spec.loader.exec_module(yafu_module)
YAFUWrapper = yafu_module.YAFUWrapper

# Add parent directory to path for imports
sys.path.insert(0, str(client_dir))

from lib.parsing_utils import parse_yafu_ecm_output, parse_yafu_auto_factors
from lib.ecm_config import ECMConfig


class TestFactorizationParsing(unittest.TestCase):
    """Test factorization parsing for ECM and YAFU."""

    @classmethod
    def setUpClass(cls):
        """Initialize wrappers once for all tests."""
        # Use absolute path to config file (one level up from tests/)
        config_path = client_dir / 'client.yaml'
        cls.ecm = ECMWrapper(str(config_path))
        cls.yafu = YAFUWrapper(str(config_path))

        # Test composite with known factorization
        cls.test_composite = "595481287174180414621815111912500384480425142428743680084508806931548176009641041122357703"

        # Expected factorization: 874392113604259^3 * 856395168938929 * 1040100281593968479843247348533
        cls.expected_factors = {
            "874392113604259": 3,
            "856395168938929": 1,
            "1040100281593968479843247348533": 1
        }

    def verify_factorization(self, factors_found: List[str], test_name: str):
        """
        Verify that found factors match expected factorization.

        Args:
            factors_found: List of factor strings (may contain duplicates)
            test_name: Name of the test for error reporting
        """
        # Count occurrences of each factor
        factor_counts = Counter(factors_found)

        # Convert to dict with strings as keys
        found_dict = {str(f): count for f, count in factor_counts.items()}

        # Check that we found all expected factors with correct multiplicities
        self.assertEqual(found_dict, self.expected_factors,
                        f"{test_name}: Factor counts don't match expected factorization")

        # Verify product equals original composite
        product = 1
        for factor_str in factors_found:
            product *= int(factor_str)

        self.assertEqual(str(product), self.test_composite,
                        f"{test_name}: Product of factors doesn't equal original composite")

        print(f"✓ {test_name}: Found {len(factor_counts)} unique factors")
        print(f"  Factorization: {' × '.join(f'{f}^{e}' if e > 1 else f for f, e in sorted(found_dict.items()))}")

    def test_ecm_factorization(self):
        """Test ECM factorization and parsing."""
        print("\n" + "="*80)
        print("Testing GMP-ECM factorization...")
        print("="*80)

        # Run ECM with moderate B1 (should find all factors) - using v2 API
        config = ECMConfig(
            composite=self.test_composite,
            b1=1000000,  # 1M should be enough
            curves=100,
            verbose=False
        )
        result = self.ecm.run_ecm_v2(config)

        # Get factors from FactorResult
        factors_found = result.factors

        self.assertGreater(len(factors_found), 0, "ECM should find at least one factor")

        # ECM should return factors with multiplicities preserved
        # (it factors composite cofactors and returns all prime factors)
        factor_counts = Counter(factors_found)

        # Verify all returned factors are valid
        for factor in factor_counts.keys():
            self.assertIn(factor, self.expected_factors.keys(),
                         f"ECM returned unexpected factor: {factor}")

        # If ECM fully factored the number, verify complete factorization
        if len(factors_found) == 5:  # All 5 factors including multiplicities
            self.verify_factorization(factors_found, "GMP-ECM Complete")
        else:
            print(f"✓ ECM found {len(factor_counts)} unique factor(s) with multiplicities preserved")
            print(f"  Factors: {', '.join(f'{f}^{e}' if e > 1 else f for f, e in sorted(factor_counts.items()))}")

    def test_yafu_ecm_factorization(self):
        """Test YAFU ECM factorization and parsing."""
        print("\n" + "="*80)
        print("Testing YAFU ECM factorization...")
        print("="*80)

        # Run YAFU ECM with pretest (should find all factors)
        results = self.yafu.run_yafu_ecm(
            composite=self.test_composite,
            method='ecm',
            verbose=False
        )

        # Get factors
        factors_found = results.get('factors_found', [])

        # Debug output for YAFU ECM
        print(f"\nDEBUG YAFU ECM: Command: {' '.join(results.get('cmd', []))}")
        print(f"DEBUG YAFU ECM: Results keys: {results.keys()}")
        print(f"DEBUG YAFU ECM: Factors found: {factors_found}")
        print(f"DEBUG YAFU ECM: Raw output length: {len(results.get('raw_output', ''))}")
        print(f"DEBUG YAFU ECM: Raw output:\n{results.get('raw_output', '')}")

        self.assertGreater(len(factors_found), 0, "YAFU ECM should find at least one factor")

        # YAFU ECM should return factors with multiplicities preserved
        factor_counts = Counter(factors_found)

        # Verify all returned factors are valid
        for factor in factor_counts.keys():
            self.assertIn(factor, self.expected_factors.keys(),
                         f"YAFU ECM returned unexpected factor: {factor}")

        # YAFU ECM with pretest should fully factor this number
        self.verify_factorization(factors_found, "YAFU ECM")

    def test_yafu_auto_factorization(self):
        """Test YAFU automatic factorization and parsing."""
        print("\n" + "="*80)
        print("Testing YAFU automatic factorization...")
        print("="*80)

        # Run YAFU in auto mode (should fully factor)
        results = self.yafu.run_yafu_auto(
            composite=self.test_composite,
            method=None,  # Auto mode
            verbose=False
        )

        # Get factors
        factors_found = results.get('factors_found', [])

        # Debug output
        print(f"\nDEBUG: Results keys: {results.keys()}")
        print(f"DEBUG: Factors found: {factors_found}")
        print(f"DEBUG: Raw output length: {len(results.get('raw_output', ''))}")
        if len(results.get('raw_output', '')) < 500:
            print(f"DEBUG: Raw output: {results.get('raw_output', '')}")

        self.assertGreater(len(factors_found), 0, "YAFU auto should find factors")

        # YAFU auto mode should find ALL factors and complete factorization
        # Verify we get the complete factorization
        self.verify_factorization(factors_found, "YAFU Auto")

    def test_yafu_parsing_multiplicity(self):
        """Test that YAFU parsing correctly preserves factor multiplicities."""
        print("\n" + "="*80)
        print("Testing YAFU parsing multiplicity preservation...")
        print("="*80)

        # Simulated YAFU output with repeated factors (YAFU indicates multiplicity by repetition)
        simulated_output = """
***factors found***
P15 = 874392113604259
P15 = 874392113604259
P15 = 874392113604259
P15 = 856395168938929
P31 = 1040100281593968479843247348533
"""

        # Parse using YAFU auto parser
        parsed_factors = parse_yafu_auto_factors(simulated_output)

        # Extract just the factor values
        factors_found = [f[0] for f in parsed_factors]

        # Verify we preserved multiplicities (should have 5 total factors)
        self.assertEqual(len(factors_found), 5,
                        "YAFU parser should preserve all factor occurrences (multiplicity)")

        # Count occurrences
        factor_counts = Counter(factors_found)

        # Verify correct multiplicities
        self.assertEqual(factor_counts["874392113604259"], 3,
                        "874392113604259 should appear 3 times")
        self.assertEqual(factor_counts["856395168938929"], 1,
                        "856395168938929 should appear 1 time")
        self.assertEqual(factor_counts["1040100281593968479843247348533"], 1,
                        "1040100281593968479843247348533 should appear 1 time")

        # Verify factorization
        self.verify_factorization(factors_found, "YAFU Parser Multiplicity Test")

        print(f"✓ YAFU parser correctly preserved multiplicities: {len(factors_found)} total factors")

    def test_pm1_method(self):
        """Test that --method pm1 actually runs P-1 (not ECM)."""
        print("\n" + "="*80)
        print("Testing P-1 method selection...")
        print("="*80)

        # Small composite that P-1 can factor: 12345678901 = 3 × 857 × 14405693
        # 857 - 1 = 856 = 2^3 × 107 (smooth, good for P-1)
        pm1_composite = "12345678901"

        # Run with method='pm1' using v2 API
        config = ECMConfig(
            composite=pm1_composite,
            b1=50,
            b2=500,
            curves=1,
            method='pm1',  # This is the key - should run P-1 not ECM
            verbose=False
        )
        result = self.ecm.run_ecm_v2(config)

        # Check that we got a result
        self.assertIsNotNone(result, "P-1 should return a result")

        # Check raw output contains [P-1] not [ECM]
        # This verifies GMP-ECM actually ran in P-1 mode
        self.assertIn('[P-1]', result.raw_output,
                     "Output should contain [P-1] marker, indicating P-1 mode was used")
        self.assertNotIn('[ECM]', result.raw_output,
                        "Output should NOT contain [ECM] marker when running P-1")

        # P-1 should find the factor 857 (or possibly all factors)
        if result.factors:
            print(f"✓ P-1 found factor(s): {result.factors}")
            # Verify at least one valid factor was found
            for factor in result.factors:
                factor_int = int(factor)
                self.assertEqual(int(pm1_composite) % factor_int, 0,
                               f"Factor {factor} should divide {pm1_composite}")
        else:
            # P-1 might not always find factors with these small bounds
            print("✓ P-1 ran successfully (no factors found with these bounds)")

        print(f"✓ Confirmed: method='pm1' runs P-1, not ECM")


def main():
    """Run tests with verbose output."""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFactorizationParsing)

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
