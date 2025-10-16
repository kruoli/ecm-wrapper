#!/usr/bin/env python3
"""
Aliquot Sequence Calculator using YAFU for factorization.

An aliquot sequence starting with n is defined as:
- a(0) = n
- a(k+1) = s(a(k)) where s(n) = σ(n) - n (sum of proper divisors)

The sequence terminates at 1, or may enter a cycle (sociable chain).
"""
import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

from base_wrapper import BaseWrapper
import importlib.util

# Import yafu-wrapper.py (Python can't import modules with hyphens normally)
spec = importlib.util.spec_from_file_location("yafu_wrapper", "yafu-wrapper.py")
yafu_module = importlib.util.module_from_spec(spec)
sys.modules["yafu_wrapper"] = yafu_module
spec.loader.exec_module(yafu_module)
YAFUWrapper = yafu_module.YAFUWrapper

# Import cado-wrapper.py
spec = importlib.util.spec_from_file_location("cado_wrapper", "cado-wrapper.py")
cado_module = importlib.util.module_from_spec(spec)
sys.modules["cado_wrapper"] = cado_module
spec.loader.exec_module(cado_module)
CADOWrapper = cado_module.CADOWrapper

# Import ecm-wrapper.py
spec = importlib.util.spec_from_file_location("ecm_wrapper", "ecm-wrapper.py")
ecm_module = importlib.util.module_from_spec(spec)
sys.modules["ecm_wrapper"] = ecm_module
spec.loader.exec_module(ecm_module)
ECMWrapper = ecm_module.ECMWrapper


class AliquotSequence:
    """Represents an aliquot sequence with tracking and cycle detection."""

    def __init__(self, start: int):
        self.start = start
        self.sequence = [start]
        self.factorizations = {}
        self.terminated = False
        self.cycle_start = None
        self.cycle_length = None

    def add_term(self, term: int, factorization: Dict[int, int]):
        """Add a term to the sequence with its factorization."""
        self.sequence.append(term)
        self.factorizations[term] = factorization

        # Check for cycles (excluding the first term)
        if term in self.sequence[:-1]:
            cycle_idx = self.sequence.index(term)
            self.cycle_start = cycle_idx
            self.cycle_length = len(self.sequence) - cycle_idx - 1
            self.terminated = True

    def check_termination(self) -> Tuple[bool, str]:
        """Check if sequence has terminated and return reason."""
        current = self.sequence[-1]

        if current == 1:
            return True, "terminated (reached 1)"

        if self.cycle_start is not None:
            cycle_terms = self.sequence[self.cycle_start:-1]
            return True, f"cyclic (period {self.cycle_length}): {' → '.join(map(str, cycle_terms))}"

        # Check if current term is prime (factorization has only one factor with exponent 1)
        if current in self.factorizations:
            factors = self.factorizations[current]
            if len(factors) == 1 and list(factors.values())[0] == 1:
                return True, f"terminated (prime: {current})"

        return False, ""


class AliquotWrapper(BaseWrapper):
    """Wrapper for computing aliquot sequences using YAFU or CADO-NFS."""

    def __init__(self, config_path: str, factorizer: str = 'yafu', hybrid_threshold: int = 100, threads: Optional[int] = None, verbose: bool = False):
        """Initialize aliquot wrapper with specified factorization engine.

        Args:
            config_path: Path to configuration file
            factorizer: Either 'yafu', 'cado', or 'hybrid' (default: 'yafu')
            hybrid_threshold: Digit length threshold for switching to ECM+CADO (default: 100)
            threads: Optional thread/worker count for parallel execution
            verbose: Enable verbose output from factorization programs
        """
        super().__init__(config_path)
        self.factorizer_name = factorizer
        self.hybrid_threshold = hybrid_threshold
        self.threads = threads
        self.verbose = verbose

        # Initialize all factorizers for hybrid mode
        self.yafu = YAFUWrapper(config_path)
        self.cado = CADOWrapper(config_path)
        self.ecm = ECMWrapper(config_path)

        # Set primary factorizer
        if factorizer == 'cado':
            self.factorizer = self.cado
        elif factorizer == 'hybrid':
            self.factorizer = None  # Will be selected dynamically
        else:
            self.factorizer = self.yafu

    def parse_factorization(self, factors_found: List[str]) -> Dict[int, int]:
        """
        Parse list of prime factors into a dictionary of {prime: exponent}.

        Args:
            factors_found: List of prime factors (may contain duplicates)

        Returns:
            Dictionary mapping prime to its exponent
        """
        if not factors_found:
            return {}

        # Count occurrences of each prime
        factor_counts = Counter(int(f) for f in factors_found)
        return dict(factor_counts)

    def calculate_divisor_sum(self, factorization: Dict[int, int]) -> int:
        """
        Calculate σ(n) - sum of all divisors including n.

        For n = p₁^a₁ × p₂^a₂ × ... × pₖ^aₖ:
        σ(n) = σ(p₁^a₁) × σ(p₂^a₂) × ... × σ(pₖ^aₖ)
        where σ(p^a) = (p^(a+1) - 1) / (p - 1)

        Args:
            factorization: Dictionary of {prime: exponent}

        Returns:
            Sum of all divisors
        """
        if not factorization:
            return 0

        sigma = 1
        for prime, exponent in factorization.items():
            # σ(p^a) = (p^(a+1) - 1) / (p - 1)
            sigma *= (prime**(exponent + 1) - 1) // (prime - 1)

        return sigma

    def calculate_next_term(self, n: int, factorization: Dict[int, int]) -> int:
        """
        Calculate next term in aliquot sequence: s(n) = σ(n) - n.

        Args:
            n: Current term
            factorization: Prime factorization of n

        Returns:
            Sum of proper divisors (next term in sequence)
        """
        sigma = self.calculate_divisor_sum(factorization)
        return sigma - n

    def factor_number(self, n: int) -> Tuple[bool, Dict[int, int], Dict]:
        """
        Factor a number completely using the selected factorization strategy.

        Strategy:
        - Numbers < hybrid_threshold digits: Use YAFU
        - Numbers >= hybrid_threshold digits (hybrid mode):
          1. Run ECM to 4/13 * digit_length t-level
          2. If fully factored, done
          3. If cofactor remains, use CADO-NFS on cofactor

        Args:
            n: Number to factor

        Returns:
            Tuple of (success, factorization_dict, raw_results)
        """
        digit_length = len(str(n))
        self.logger.info(f"Factoring {n} ({digit_length} digits)...")

        # Determine strategy
        if self.factorizer_name == 'hybrid' and digit_length >= self.hybrid_threshold:
            # Hybrid strategy: ECM pre-factorization + CADO-NFS
            return self._factor_hybrid(n, digit_length)
        elif self.factorizer_name == 'cado':
            # Pure CADO-NFS
            results = self.cado.run_cado_nfs(composite=str(n), threads=self.threads, verbose=self.verbose)
        else:
            # Pure YAFU
            results = self.yafu.run_yafu_auto(composite=str(n), threads=self.threads)

        if not results.get('success'):
            self.logger.error(f"Factorization failed for {n}")
            return False, {}, results

        factors = results.get('factors_found', [])
        if not factors:
            self.logger.warning(f"No factors found for {n}")
            return False, {}, results

        factorization = self.parse_factorization(factors)
        self.logger.info(f"Factorization: {self.format_factorization(factorization)}")

        return True, factorization, results

    def _trial_division(self, n: int, limit: int = 10**7) -> Tuple[List[int], int]:
        """
        Fast trial division to find small prime factors.

        Args:
            n: Number to factor
            limit: Trial division limit (default: 10^7)

        Returns:
            Tuple of (factors_found, cofactor)
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
        # This is faster than checking every number
        i = 7
        while i * i <= cofactor and i <= limit:
            while cofactor % i == 0:
                factors.append(i)
                cofactor //= i
            i += 2
            # Skip multiples of 3 and 5
            if i % 3 == 0 or i % 5 == 0:
                continue

        return factors, cofactor

    def _factor_hybrid(self, n: int, digit_length: int) -> Tuple[bool, Dict[int, int], Dict]:
        """
        Hybrid factorization: Trial division + ECM pre-factorization + YAFU + CADO-NFS.

        Strategy:
        1. Trial division up to 10^7 (very fast, catches small factors)
        2. ECM to 4/13 * digit_length t-level (finds medium factors)
        3. YAFU auto (handles remaining small-medium composites + primality)
        4. CADO-NFS if large composite remains (90+ digits)

        Args:
            n: Number to factor
            digit_length: Number of digits in n

        Returns:
            Tuple of (success, factorization_dict, raw_results)
        """
        all_factors = []
        current_composite = n

        # Step 0: Trial division with small primes (very fast)
        self.logger.info(f"Running trial division up to 10^7...")
        trial_factors, current_composite = self._trial_division(current_composite)
        if trial_factors:
            self.logger.info(f"Trial division found {len(trial_factors)} small factor(s)")
            all_factors.extend([str(f) for f in trial_factors])

        if current_composite == 1:
            self.logger.info("Fully factored by trial division")
            factorization = self.parse_factorization(all_factors)
            return True, factorization, {'success': True, 'method': 'trial_division'}

        cofactor_digits = len(str(current_composite))
        self.logger.info(f"Cofactor after trial division: {current_composite} ({cofactor_digits} digits)")

        # If cofactor dropped below hybrid threshold, skip ECM and use YAFU
        if cofactor_digits < self.hybrid_threshold:
            self.logger.info(f"Cofactor is now < {self.hybrid_threshold} digits, using YAFU directly")
            yafu_results = self.yafu.run_yafu_auto(composite=str(current_composite), threads=self.threads)
            yafu_factors = yafu_results.get('factors_found', [])
            if yafu_factors:
                all_factors.extend(yafu_factors)

            factorization = self.parse_factorization(all_factors)

            # Verify factorization
            product = 1
            for factor_str in all_factors:
                product *= int(factor_str)

            if product != n:
                self.logger.error(f"Factorization verification failed: {product} != {n}")
                return False, {}, yafu_results

            return True, factorization, yafu_results

        # Step 1: Run ECM to 4/13 * digit_length t-level
        target_t_level = int((4.0 / 13.0) * digit_length)
        self.logger.info(f"Running ECM pre-factorization to t{target_t_level}...")

        # Get ECM B1 from t-level (approximate)
        # This is a simplified lookup - you might want a more precise table
        b1_table = {
            20: 11000, 25: 50000, 30: 250000, 35: 1000000,
            40: 3000000, 45: 11000000, 50: 43000000, 55: 110000000,
            60: 260000000, 65: 850000000, 70: 2900000000
        }

        # Find closest t-level
        b1 = b1_table.get(target_t_level)
        if b1 is None:
            # Interpolate or use closest
            closest_t = min(b1_table.keys(), key=lambda x: abs(x - target_t_level))
            b1 = b1_table[closest_t]

        self.logger.info(f"Using B1={b1} for t{target_t_level}")

        # Run ECM with enough curves for the target t-level
        # Rough estimate: ~100-1000 curves depending on t-level
        curves = max(100, target_t_level * 10)

        self.logger.info(f"Calculated {curves} curves for t{target_t_level}")

        # Run ECM on the cofactor (after trial division)
        # Use multiprocess mode if threads specified, otherwise regular mode
        if self.threads and self.threads > 1:
            self.logger.info(f"Using multiprocess ECM with {self.threads} workers, {curves} total curves")
            ecm_results = self.ecm.run_ecm_multiprocess(
                composite=str(current_composite),
                b1=b1,
                b2=None,  # Use ECM default B2
                curves=curves,
                workers=self.threads,
                verbose=self.verbose
            )
        else:
            ecm_results = self.ecm.run_ecm(
                composite=str(current_composite),
                b1=b1,
                b2=None,  # Use ECM default B2
                curves=curves,
                verbose=self.verbose
            )

        # Collect ECM factors (handle both singular and plural returns)
        ecm_factors = ecm_results.get('factors_found', [])
        if not ecm_factors and ecm_results.get('factor_found'):
            # Multiprocess mode may return factor_found (singular)
            ecm_factors = [ecm_results['factor_found']]

        if ecm_factors:
            self.logger.info(f"ECM found {len(ecm_factors)} factor(s): {ecm_factors}")
            all_factors.extend(ecm_factors)

            # Calculate cofactor
            for factor in ecm_factors:
                current_composite = current_composite // int(factor)

            self.logger.info(f"Cofactor after ECM: {current_composite} ({len(str(current_composite))} digits)")

        # Step 2: Check if fully factored or need further factorization
        if current_composite == 1:
            # Fully factored by ECM
            self.logger.info("Fully factored by ECM")
            factorization = self.parse_factorization(all_factors)
            return True, factorization, ecm_results

        # If cofactor is still large (90+ digits), use CADO-NFS directly
        cofactor_digits = len(str(current_composite))
        if cofactor_digits >= 90:
            self.logger.info(f"Cofactor is {cofactor_digits} digits, using CADO-NFS")
            cado_results = self.cado.run_cado_nfs(composite=str(current_composite), threads=self.threads, verbose=self.verbose)
            cado_factors = cado_results.get('factors_found', [])
            if cado_factors:
                all_factors.extend(cado_factors)
        else:
            # Small enough for YAFU to handle efficiently
            self.logger.info(f"Cofactor is {cofactor_digits} digits, using YAFU")
            yafu_results = self.yafu.run_yafu_auto(composite=str(current_composite), threads=self.threads)
            yafu_factors = yafu_results.get('factors_found', [])
            if yafu_factors:
                all_factors.extend(yafu_factors)

        if not all_factors:
            self.logger.error("Hybrid factorization failed - no factors found")
            return False, {}, ecm_results

        factorization = self.parse_factorization(all_factors)
        self.logger.info(f"Final factorization: {self.format_factorization(factorization)}")

        # Verify factorization
        product = 1
        for factor_str in all_factors:
            product *= int(factor_str)

        if product != n:
            self.logger.error(f"Factorization verification failed: {product} != {n}")
            return False, {}, ecm_results

        return True, factorization, ecm_results

    def format_factorization(self, factorization: Dict[int, int]) -> str:
        """Format factorization as string like '2^3 × 3 × 23'."""
        parts = []
        for prime in sorted(factorization.keys()):
            exp = factorization[prime]
            if exp == 1:
                parts.append(str(prime))
            else:
                parts.append(f"{prime}^{exp}")
        return " × ".join(parts)

    def compute_sequence(self, start: int, max_iterations: int = 100,
                        submit_to_factordb: bool = False) -> AliquotSequence:
        """
        Compute aliquot sequence starting from given number.

        Args:
            start: Starting number
            max_iterations: Maximum number of iterations
            submit_to_factordb: Whether to submit to FactorDB

        Returns:
            AliquotSequence object with full sequence data
        """
        seq = AliquotSequence(start)

        # Factor the starting number
        success, factorization, _ = self.factor_number(start)
        if not success:
            self.logger.error("Failed to factor starting number")
            return seq

        seq.factorizations[start] = factorization

        # Compute sequence
        current = start
        for iteration in range(max_iterations):
            # Calculate next term
            next_term = self.calculate_next_term(current, seq.factorizations[current])

            self.logger.info(f"Iteration {iteration + 1}: {current} → {next_term}")
            print(f"\nStep {iteration + 1}:")
            print(f"  Current: {current}")
            print(f"  Factorization: {self.format_factorization(seq.factorizations[current])}")
            print(f"  σ({current}) = {self.calculate_divisor_sum(seq.factorizations[current])}")
            print(f"  Next term: {next_term}")

            # Check for termination before factoring next term
            if next_term == 0:
                self.logger.info("Sequence terminated (reached 0 - perfect number)")
                seq.terminated = True
                break

            if next_term == 1:
                seq.add_term(next_term, {1: 1})
                self.logger.info("Sequence terminated (reached 1)")
                seq.terminated = True
                break

            # Factor next term
            success, factorization, results = self.factor_number(next_term)
            if not success:
                self.logger.error(f"Failed to factor {next_term}, stopping sequence")
                seq.add_term(next_term, {})
                break

            # Submit to FactorDB if requested
            if submit_to_factordb and factorization:
                self.submit_to_factordb(next_term, factorization)

            # Add to sequence
            seq.add_term(next_term, factorization)

            # Check for cycles or other termination
            terminated, reason = seq.check_termination()
            if terminated:
                self.logger.info(f"Sequence {reason}")
                print(f"\n  Status: {reason}")
                break

            current = next_term
        else:
            self.logger.warning(f"Reached maximum iterations ({max_iterations})")
            print(f"\nReached maximum iterations ({max_iterations})")

        return seq

    def fetch_factordb_last_term(self, start: int) -> Optional[Tuple[int, int]]:
        """
        Fetch the last known term from FactorDB for an aliquot sequence.

        Args:
            start: Starting number of the aliquot sequence

        Returns:
            Tuple of (iteration, composite) or None if fetch failed
        """
        import requests
        import re

        try:
            url = f"https://factordb.com/sequences.php?se=1&aq={start}&action=last&fr=0&to=100"
            self.logger.info(f"Fetching last known term from FactorDB for sequence {start}...")

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            html = response.text

            # Parse iteration number: <td bgcolor="#DDDDDD">2157</td>
            iteration_match = re.search(r'<td bgcolor="#DDDDDD">(\d+)</td>', html)
            if not iteration_match:
                self.logger.warning("Could not find iteration number in FactorDB response")
                return None

            iteration = int(iteration_match.group(1))

            # Parse the number ID to fetch full composite
            id_match = re.search(r'id=(\d+).*?<font color="#\w+">(.*?)</font>', html)
            if not id_match:
                self.logger.warning("Could not find composite ID in FactorDB response")
                return None

            composite_id = id_match.group(1)

            # Fetch full number from FactorDB
            show_url = f"https://factordb.com/index.php?showid={composite_id}"
            show_response = requests.get(show_url, timeout=30)
            show_response.raise_for_status()

            # Extract full number from show page
            number_match = re.search(r'<pre>([\d\s]+)</pre>', show_response.text)
            if number_match:
                composite_str = number_match.group(1).replace(' ', '').replace('\n', '')
                composite = int(composite_str)
                self.logger.info(f"FactorDB: Found iteration {iteration} with {len(composite_str)}-digit composite")
                return (iteration, composite)
            else:
                self.logger.warning("Could not extract full number from FactorDB")
                return None

        except requests.RequestException as e:
            self.logger.error(f"FactorDB fetch failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error parsing FactorDB response: {e}")
            return None

    def submit_to_factordb(self, n: int, factorization: Dict[int, int]) -> bool:
        """
        Submit factorization to FactorDB.

        FactorDB API documentation: http://factordb.com/api
        Submission: GET request to http://factordb.com/index.php?query=<number>
        Query factorization: GET request to http://factordb.com/api?query=<number>

        Args:
            n: Number that was factored
            factorization: Prime factorization

        Returns:
            True if submission succeeded
        """
        import requests

        # Build factor list with repetitions
        factors = []
        for prime, exp in sorted(factorization.items()):
            factors.extend([str(prime)] * exp)

        # Reconstruct number from factors to verify
        product = 1
        for f in factors:
            product *= int(f)

        if product != n:
            self.logger.error(f"Factor verification failed: {product} != {n}")
            return False

        try:
            # First query to see if number is already in database
            query_url = f"http://factordb.com/api?query={n}"
            self.logger.info(f"Querying FactorDB for {n}...")

            response = requests.get(query_url, timeout=30)
            response.raise_for_status()

            result = response.json()

            # Check current factorization status
            # Status codes: C=Composite, CF=Composite fully factored, FF=Composite factors found
            status = result.get('status', 'Unknown')

            self.logger.info(f"FactorDB status for {n}: {status}")
            print(f"  FactorDB status: {status}")

            # If not fully factored, submit our factorization by visiting the URL
            # FactorDB automatically processes factorizations from query parameters
            if status != 'FF' and status != 'P':  # Not fully factored or prime
                submit_url = f"http://factordb.com/index.php?query={n}"
                self.logger.info(f"Submitting factorization to FactorDB: {self.format_factorization(factorization)}")
                print(f"  Submitting: {self.format_factorization(factorization)}")

                submit_response = requests.get(submit_url, timeout=30)
                submit_response.raise_for_status()

                print(f"  FactorDB URL: {submit_url}")

            return True

        except requests.RequestException as e:
            self.logger.error(f"FactorDB submission failed: {e}")
            print(f"  Error: Failed to submit to FactorDB - {e}")
            return False

    def save_sequence(self, seq: AliquotSequence, output_file: Optional[Path] = None) -> Path:
        """
        Save sequence data to JSON file in data/aliquot_sequences/.

        Args:
            seq: AliquotSequence to save
            output_file: Optional output path

        Returns:
            Path where sequence was saved
        """
        if output_file is None:
            output_dir = Path("data/aliquot_sequences")
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_dir / f"aliquot_{seq.start}_{timestamp}.json"
        else:
            # Ensure it's in data/ directory
            if not str(output_file).startswith('data/'):
                output_file = Path("data/aliquot_sequences") / output_file.name
            output_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'start': seq.start,
            'sequence': seq.sequence,
            'length': len(seq.sequence),
            'factorizations': {
                str(n): {str(p): e for p, e in factors.items()}
                for n, factors in seq.factorizations.items()
            },
            'terminated': seq.terminated,
            'cycle_start': seq.cycle_start,
            'cycle_length': seq.cycle_length,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Sequence saved to {output_file}")
        return output_file

    def cleanup_temp_files(self):
        """Clean up temporary files created by YAFU and CADO-NFS."""
        import glob

        # YAFU creates these files
        yafu_temp_files = [
            'factor.log',
            'session.log',
            'siqs.dat',
            'ggnfs.log',
            '*.fb',
            '*.job'
        ]

        # CADO-NFS working directory files (if run from client/)
        cado_temp_patterns = [
            'cado-nfs.*',
            '*.poly',
            '*.roots*'
        ]

        cleaned_files = []
        for pattern in yafu_temp_files + cado_temp_patterns:
            for filepath in glob.glob(pattern):
                try:
                    Path(filepath).unlink()
                    cleaned_files.append(filepath)
                except Exception as e:
                    self.logger.debug(f"Could not remove {filepath}: {e}")

        if cleaned_files:
            self.logger.info(f"Cleaned up {len(cleaned_files)} temporary file(s): {', '.join(cleaned_files)}")

        return cleaned_files

    def print_summary(self, seq: AliquotSequence):
        """Print summary of the sequence."""
        print("\n" + "="*80)
        print("ALIQUOT SEQUENCE SUMMARY")
        print("="*80)
        print(f"Starting number: {seq.start}")
        print(f"Sequence length: {len(seq.sequence)}")
        print(f"Sequence: {' → '.join(map(str, seq.sequence[:10]))}")
        if len(seq.sequence) > 10:
            print(f"          ... ({len(seq.sequence) - 10} more terms)")

        terminated, reason = seq.check_termination()
        if terminated:
            print(f"Status: {reason.capitalize()}")
        else:
            print(f"Status: Open (not yet terminated)")

        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Calculate aliquot sequences using YAFU or CADO-NFS for factorization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate aliquot sequence starting from 276
  python3 aliquot-wrapper.py --start 276

  # Use CADO-NFS for faster GNFS factorization
  python3 aliquot-wrapper.py --start 276 --factorizer cado

  # Calculate with more iterations
  python3 aliquot-wrapper.py --start 1248 --max-iterations 50

  # Submit results to FactorDB
  python3 aliquot-wrapper.py --start 138 --factordb

  # Quiet mode (no factor spam)
  python3 aliquot-wrapper.py --start 276 --quiet-factors --factorizer cado

  # Resume from FactorDB (fetches last known term automatically)
  python3 aliquot-wrapper.py --start 276 --resume-factordb --quiet-factors

  # Manual resume from specific iteration
  python3 aliquot-wrapper.py --start 276 --resume-iteration 2157 --resume-composite 175258998...

  # Use 8 threads/workers for parallel execution
  python3 aliquot-wrapper.py --start 276 --threads 8 --quiet-factors

  # Verbose mode (show detailed output from ECM and CADO-NFS)
  python3 aliquot-wrapper.py --start 276 -v --threads 8

Common test sequences:
  276 → 396 → 696 → 1104 → 1872 → 3770 → ... (terminates at 1)
  220 → 284 → 220 (amicable pair, cycle of length 2)
  138 → long open sequence
        """
    )

    parser.add_argument('--start', type=int, required=True,
                       help='Starting number for the aliquot sequence')
    parser.add_argument('--max-iterations', type=int, default=100,
                       help='Maximum number of iterations (default: 100)')
    parser.add_argument('--config', type=str, default='client.yaml',
                       help='Configuration file path (default: client.yaml)')
    parser.add_argument('--factordb', action='store_true',
                       help='Submit factorizations to FactorDB')
    parser.add_argument('--output', type=str,
                       help='Output JSON file for sequence data')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save sequence to file')
    parser.add_argument('--quiet-factors', action='store_true',
                       help='Disable factor logging to factors_found.txt (reduces spam for aliquot sequences)')
    parser.add_argument('--factorizer', type=str, choices=['yafu', 'cado', 'hybrid'], default='hybrid',
                       help='Factorization strategy: yafu, cado, or hybrid (default: hybrid - uses ECM+CADO for large numbers)')
    parser.add_argument('--hybrid-threshold', type=int, default=100,
                       help='Digit length threshold for hybrid ECM+CADO strategy (default: 100)')
    parser.add_argument('--resume-factordb', action='store_true',
                       help='Resume from last known term in FactorDB')
    parser.add_argument('--resume-iteration', type=int,
                       help='Resume from specific iteration with composite given via --resume-composite')
    parser.add_argument('--resume-composite', type=str,
                       help='Composite number to resume from (use with --resume-iteration)')
    parser.add_argument('--threads', type=int,
                       help='Number of threads/workers for parallel execution (ECM: multiprocess workers, YAFU/CADO: threads)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output from factorization programs (ECM, CADO-NFS)')

    args = parser.parse_args()

    # Initialize wrapper with selected factorizer
    wrapper = AliquotWrapper(args.config, factorizer=args.factorizer, hybrid_threshold=args.hybrid_threshold, threads=args.threads, verbose=args.verbose)

    # Override factor logging config if requested
    if args.quiet_factors:
        wrapper.config['logging']['log_factors_found'] = False
        wrapper.yafu.config['logging']['log_factors_found'] = False
        wrapper.cado.config['logging']['log_factors_found'] = False
        wrapper.ecm.config['logging']['log_factors_found'] = False

    print(f"\nComputing aliquot sequence starting from {args.start}")
    print("="*80)

    # Handle resume options
    resume_iteration = None
    resume_composite = None

    if args.resume_factordb:
        # Fetch last known term from FactorDB
        result = wrapper.fetch_factordb_last_term(args.start)
        if result:
            resume_iteration, resume_composite = result
            print(f"Resuming from FactorDB: iteration {resume_iteration}, {len(str(resume_composite))}-digit composite")
        else:
            print("Failed to fetch from FactorDB, starting from scratch")
    elif args.resume_iteration is not None and args.resume_composite:
        # Manual resume
        resume_iteration = args.resume_iteration
        resume_composite = int(args.resume_composite)
        print(f"Resuming from manual input: iteration {resume_iteration}, {len(str(resume_composite))}-digit composite")

    # Initialize sequence appropriately
    if resume_iteration is not None and resume_composite is not None:
        # Create sequence starting at resume point
        seq = AliquotSequence(args.start)
        # Mark iterations up to resume point as already done
        for i in range(resume_iteration):
            seq.sequence.append(None)  # Placeholder for unknown intermediates
        seq.sequence.append(resume_composite)

        # Factor the resume composite
        success, factorization, _ = wrapper.factor_number(resume_composite)
        if success:
            seq.factorizations[resume_composite] = factorization

            # Continue from this point
            current = resume_composite
            for iteration in range(args.max_iterations):
                next_term = wrapper.calculate_next_term(current, seq.factorizations[current])

                wrapper.logger.info(f"Iteration {resume_iteration + iteration + 1}: {current} → {next_term}")
                print(f"\nStep {resume_iteration + iteration + 1}:")
                print(f"  Current: {current}")
                print(f"  Factorization: {wrapper.format_factorization(seq.factorizations[current])}")
                print(f"  σ({current}) = {wrapper.calculate_divisor_sum(seq.factorizations[current])}")
                print(f"  Next term: {next_term}")

                if next_term == 0 or next_term == 1:
                    seq.add_term(next_term, {1: 1} if next_term == 1 else {})
                    seq.terminated = True
                    break

                # Factor next term
                success, factorization, results = wrapper.factor_number(next_term)
                if not success:
                    wrapper.logger.error(f"Failed to factor {next_term}, stopping")
                    seq.add_term(next_term, {})
                    break

                if args.factordb and factorization:
                    wrapper.submit_to_factordb(next_term, factorization)

                seq.add_term(next_term, factorization)

                terminated, reason = seq.check_termination()
                if terminated:
                    wrapper.logger.info(f"Sequence {reason}")
                    print(f"\n  Status: {reason}")
                    break

                current = next_term
        else:
            print("Failed to factor resume composite")
            wrapper.cleanup_temp_files()
            sys.exit(1)
    else:
        # Normal computation from start
        seq = wrapper.compute_sequence(
            start=args.start,
            max_iterations=args.max_iterations,
            submit_to_factordb=args.factordb
        )

    # Print summary
    wrapper.print_summary(seq)

    # Save sequence unless disabled
    if not args.no_save:
        output_path = Path(args.output) if args.output else None
        saved_path = wrapper.save_sequence(seq, output_path)
        print(f"\nSequence saved to: {saved_path}")

    # Clean up temporary files created by YAFU/CADO
    wrapper.cleanup_temp_files()

    # Exit with success if sequence completed
    sys.exit(0 if seq.terminated else 1)


if __name__ == '__main__':
    main()
