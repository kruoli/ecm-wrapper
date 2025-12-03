#!/usr/bin/env python3
"""
ResultProcessor - Unified factor processing for all ECM wrapper execution modes.

This class eliminates ~80 lines of duplicated factor processing logic across:
- ECMWrapper._log_and_store_factors()
- ECMWrapper.run_ecm() (composite factor handling)
- ECMWrapper.run_ecm_multiprocess() (composite factor handling)
"""
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING
from lib.ecm_math import is_probably_prime

if TYPE_CHECKING:
    from ecm_wrapper import ECMWrapper


class ResultProcessor:
    """Handles factor deduplication, logging, and recursive factorization."""

    def __init__(self, wrapper: 'ECMWrapper', composite: str, method: str,
                 b1: int, b2: Optional[int], curves: int, program: str):
        """
        Initialize result processor.

        Args:
            wrapper: ECMWrapper instance (for logging and factorization)
            composite: Original composite number being factored
            method: Method name (ecm, pm1, pp1)
            b1, b2, curves: ECM parameters
            program: Program name for logging
        """
        self.wrapper = wrapper
        self.composite = composite
        self.method = method
        self.b1 = b1
        self.b2 = b2
        self.curves = curves
        self.program = program
        self.logger = wrapper.logger

    def deduplicate_factors(self, all_factors: List[Tuple[str, Optional[str]]]) -> Dict[str, Optional[str]]:
        """
        Deduplicate factors - same factor can be found by multiple curves.

        Args:
            all_factors: List of (factor, sigma) tuples

        Returns:
            Dict mapping factor -> sigma (keeping first sigma found)
        """
        unique_factors = {}
        for factor, sigma in all_factors:
            if factor not in unique_factors:
                unique_factors[factor] = sigma  # Keep first sigma found
        return unique_factors

    def log_and_store_factors(self, all_factors: List[Tuple[str, Optional[str]]],
                              results: Dict[str, Any], quiet: bool = False) -> Optional[str]:
        """
        Deduplicate factors, log them, and store in results dictionary.

        This replaces ECMWrapper._log_and_store_factors() and similar code patterns.

        Args:
            all_factors: List of (factor, sigma) tuples
            results: Results dictionary to update
            quiet: If True, skip logging (but still store factors)

        Returns:
            First factor (for compatibility)
        """
        if not all_factors:
            return None

        # Deduplicate factors
        unique_factors = self.deduplicate_factors(all_factors)

        # Log each unique factor once
        if not quiet:
            for factor, sigma in unique_factors.items():
                self.wrapper.log_factor_found(
                    self.composite, factor, self.b1, self.b2, self.curves,
                    method=self.method, sigma=sigma, program=self.program
                )

        # Store all unique factors for API submission
        if 'factors_found' not in results:
            results['factors_found'] = []
        results['factors_found'].extend(unique_factors.keys())

        # Store factor-to-sigma mapping for multiple factor submissions
        if 'factor_sigmas' not in results:
            results['factor_sigmas'] = {}
        results['factor_sigmas'].update(unique_factors)

        # Set the main factor for compatibility (use first factor found)
        main_factor = list(unique_factors.keys())[0]
        results['factor_found'] = main_factor

        # Store sigma for the main factor (for API submission)
        results['sigma'] = unique_factors[main_factor]

        self.logger.info(f"Factors found: {list(unique_factors.keys())}")

        return main_factor

    def fully_factor_and_store(self, factors_found: List[str], results: Dict[str, Any],
                               quiet: bool = False) -> List[str]:
        """
        Fully factor any composite factors found, calculate cofactor, and update results.

        This replaces the duplicated logic in:
        - run_ecm() lines 317-356
        - run_ecm_multiprocess() lines 1061-1096

        Args:
            factors_found: List of factors to process (may contain composites)
            results: Results dictionary to update
            quiet: If True, skip logging individual primes

        Returns:
            List of all prime factors
        """
        if not factors_found:
            return []

        # Deduplicate factors (preserve order)
        unique_factors = list(dict.fromkeys(factors_found))

        # Track which factors were actually found by ECM vs cofactor primes
        ecm_found_factors = []  # Actually found by ECM (have sigma)
        all_prime_factors = []  # All primes including cofactor primes

        # Fully factor any composite factors found
        self.logger.info(f"Checking {len(unique_factors)} factor(s) for composites...")

        for factor in unique_factors:
            prime_factors = self.wrapper._fully_factor_found_result(factor, quiet=True)
            self.logger.info(f"Prime factorization of {factor}: {prime_factors}")
            all_prime_factors.extend(prime_factors)
            # These prime factors came from the ECM-found factor, so they count as ECM-found
            ecm_found_factors.extend(prime_factors)

        # Calculate remaining cofactor after dividing out all found primes
        cofactor = int(self.composite)
        for prime in all_prime_factors:
            cofactor //= int(prime)

        # Check if there's a remaining cofactor
        cofactor_primes = []  # Track primes from cofactor separately
        if cofactor > 1:
            cofactor_digits = len(str(cofactor))

            # Test if cofactor is prime
            if is_probably_prime(cofactor):
                self.logger.info(f"Remaining cofactor {cofactor} is prime")
                all_prime_factors.append(str(cofactor))
                cofactor_primes.append(str(cofactor))
            else:
                # Cofactor is composite - only auto-factor if small enough (ECM is fastest for <60 digits)
                if cofactor_digits < 60:
                    self.logger.info(f"Remaining cofactor {cofactor} is composite - factoring...")
                    cofactor_primes = self.wrapper._fully_factor_found_result(str(cofactor), quiet=True)
                    all_prime_factors.extend(cofactor_primes)
                else:
                    self.logger.info(f"Remaining cofactor {cofactor} is composite (not auto-factoring)")

        # Replace with fully factored results
        results['factors_found'] = all_prime_factors  # All factors (for aliquot-wrapper compatibility)
        results['ecm_found_factors'] = ecm_found_factors  # Only ECM-found factors (for API submission)
        results['cofactor_primes'] = cofactor_primes  # Cofactor primes (not submitted to API)
        if all_prime_factors:
            results['factor_found'] = all_prime_factors[0]

        # Log only ECM-found factors (with sigma) to factors_found.txt
        if not quiet:
            for prime in ecm_found_factors:
                self.wrapper.log_factor_found(
                    self.composite, prime, self.b1, self.b2, self.curves,
                    method=self.method, sigma=results.get('sigma'), program=self.program
                )
            # Log cofactor primes to console but not to factors file
            for prime in cofactor_primes:
                self.logger.info(f"Cofactor prime found (not logged to file): {prime}")

        return all_prime_factors
