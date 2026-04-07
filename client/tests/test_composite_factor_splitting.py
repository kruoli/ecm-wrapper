#!/usr/bin/env python3
"""
Integration test for composite factor splitting.

Verifies that _fully_factor_composite() correctly breaks down composite
factors into their prime components. Uses a crafted composite where ECM
at high B1 will find a C30 (product of two small primes) as a factor,
then verifies the recursive factoring splits it.

Requires: GMP-ECM binary installed and accessible.

Test composite (generated via openssl prime -generate):
  P16  = 1005883299031357
  P15  = 848647905930343
  P100 = 6809176950498902954726360594124819403572482951246044828034216367423425089828925965113417567338162803
  C130 = P16 * P15 * P100

At B1=11M, ECM will almost always find both small primes simultaneously
(their group orders are B1-smooth), returning C30 = P16 * P15. The test
then verifies _fully_factor_composite splits C30 into P16 and P15.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from lib.ecm_executor import ECMWrapper
from lib.ecm_math import is_probably_prime

# Skip entire module if ecm binary not available
pytestmark = pytest.mark.skipif(
    shutil.which('ecm') is None,
    reason='GMP-ECM binary not found'
)

# Test primes (generated via openssl prime -generate)
P16 = '1005883299031357'
P15 = '848647905930343'
P100 = '6809176950498902954726360594124819403572482951246044828034216367423425089828925965113417567338162803'
C30 = str(int(P16) * int(P15))  # 853640755333266133427814765451
COMPOSITE = str(int(P16) * int(P15) * int(P100))

# Path to client.yaml
CONFIG_PATH = str(Path(__file__).parent.parent / 'client.yaml')


class TestCompositeFactorSplitting:
    """Integration tests for _fully_factor_composite with real ECM binary."""

    @pytest.fixture
    def wrapper(self):
        """Create an ECMWrapper instance using standard config."""
        return ECMWrapper(CONFIG_PATH)

    def test_fully_factor_composite_splits_c30(self, wrapper):
        """Test that _fully_factor_composite splits C30 = P16 * P15 into primes."""
        result = wrapper._fully_factor_composite(C30)

        # Should return exactly two prime factors
        assert len(result) == 2, f"Expected 2 prime factors, got {len(result)}: {result}"

        # Both should be prime
        for f in result:
            assert is_probably_prime(int(f)), f"Factor {f} is not prime"

        # Should be our two known primes (order may vary)
        assert set(result) == {P16, P15}, f"Expected {{{P16}, {P15}}}, got {set(result)}"

    def test_ecm_finds_composite_factor_and_splits(self, wrapper):
        """
        End-to-end: run ECM on C130 = P16 * P15 * P100.

        At B1=11M, ECM almost always finds both small primes simultaneously
        (C30), not individually. If a run happens to find only one prime,
        retry until we get a composite factor to test splitting.
        """
        max_retries = 10
        composite_factor_found = False

        for _ in range(max_retries):
            prim_result = wrapper._execute_ecm_primitive(
                composite=COMPOSITE,
                b1=11_000_000,
                curves=1,
                verbose=False,
            )

            factors = prim_result.get('factors', [])
            if not factors:
                continue

            factor = factors[0]
            if not is_probably_prime(int(factor)):
                # Got a composite factor - this is the case we want to test
                composite_factor_found = True

                # Now verify _fully_factor_composite splits it
                primes = wrapper._fully_factor_composite(factor)

                # All results should be prime
                for p in primes:
                    assert is_probably_prime(int(p)), f"Factor {p} is not prime"

                # The primes should divide the original composite factor
                product = 1
                for p in primes:
                    product *= int(p)
                assert product == int(factor), (
                    f"Prime product {product} != composite factor {factor}"
                )
                break

        assert composite_factor_found, (
            f"ECM never returned a composite factor in {max_retries} attempts "
            f"(always found individual primes). This is unlikely but not a bug."
        )

    def test_fully_factor_composite_handles_prime(self, wrapper):
        """Test that _fully_factor_composite returns a prime factor as-is."""
        result = wrapper._fully_factor_composite(P16)
        assert result == [P16]

    def test_fully_factor_composite_handles_small_composites(self, wrapper):
        """Test splitting a small composite that trial division can handle."""
        # 2 * 3 * 5 * 7 = 210
        result = wrapper._fully_factor_composite('210')
        primes = sorted(int(p) for p in result)
        assert primes == [2, 3, 5, 7]
