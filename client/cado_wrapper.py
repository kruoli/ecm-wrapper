#!/usr/bin/env python3
import os
import subprocess
import sys
from typing import Optional, Dict, Any, List
from lib.base_wrapper import BaseWrapper
from lib.parsing_utils import Timeouts
from lib.user_output import UserOutput
from lib.results_builder import ResultsBuilder

class CADOWrapper(BaseWrapper):
    def __init__(self, config_path: str):
        """Initialize CADO-NFS wrapper with shared base functionality"""
        super().__init__(config_path)

    def _build_cado_cmd(self, composite: str, threads: Optional[int] = None) -> List[str]:
        """Build CADO-NFS command.

        Args:
            composite: Number to factor
            threads: Number of threads (uses config default if not specified)

        Returns:
            Command list for execution
        """
        cado_path = os.path.expanduser(self.config['programs']['cado_nfs']['path'])
        cmd = ['python3', cado_path, composite]

        # Add threading parameter
        if threads is None:
            threads = self.config['programs']['cado_nfs'].get('threads', 4)
        cmd.extend(['-t', str(threads)])

        return cmd

    def parse_cado_output(self, output: str) -> List[tuple[str, Optional[str]]]:
        """Parse CADO-NFS output for factors.

        CADO-NFS outputs factors in two places:
        1. "Square Root: Factors: factor1 factor2 ..." in the log
        2. Space-separated factors on the last non-empty line

        Args:
            output: CADO-NFS stdout/stderr

        Returns:
            List of (factor, sigma) tuples (sigma is always None for NFS)
        """
        factors: List[tuple[str, Optional[str]]] = []

        # Strategy 1: Look for "Square Root: Factors:" line
        for line in output.split('\n'):
            if 'Square Root: Factors:' in line:
                # Extract factors after "Factors:"
                parts = line.split('Factors:')
                if len(parts) > 1:
                    factor_strs = parts[1].strip().split()
                    for factor in factor_strs:
                        # Remove ANSI color codes if present
                        clean_factor = factor.strip()
                        if clean_factor.isdigit():
                            factors.append((clean_factor, None))
                            self.logger.debug(f"Found factor via Square Root line: {clean_factor}")

        # Strategy 2: Check last line for space-separated factors
        if not factors:
            lines = [l.strip() for l in output.split('\n') if l.strip()]
            if lines:
                last_line = lines[-1]
                # Last line should be space-separated factors
                potential_factors = last_line.split()
                # Verify all are digits
                if all(p.isdigit() for p in potential_factors):
                    for factor in potential_factors:
                        factors.append((factor, None))
                        self.logger.debug(f"Found factor from final line: {factor}")

        if factors:
            self.logger.info(f"Parsed {len(factors)} factors from CADO-NFS output")

        return factors

    def run_cado_nfs(self, composite: str, threads: Optional[int] = None, verbose: bool = False) -> Dict[str, Any]:
        """Run CADO-NFS factorization.

        Args:
            composite: Number to factor
            threads: Number of threads (optional, uses config default)
            verbose: If True, stream CADO-NFS output to stdout in real-time

        Returns:
            Results dictionary with factors_found, execution_time, etc.
        """
        import time

        # Build command
        cmd = self._build_cado_cmd(composite, threads)

        # Create results dictionary using ResultsBuilder
        builder = ResultsBuilder(composite, 'nfs')
        results = builder.build_no_truncate()
        results['threads'] = threads

        start_time = time.time()
        process: Optional[subprocess.Popen[str]] = None

        try:
            self.logger.info(f"Running CADO-NFS on {len(composite)}-digit number with {threads or self.config['programs']['cado_nfs'].get('threads', 4)} threads")

            # Run CADO-NFS
            working_dir = os.path.expanduser(self.config['programs']['cado_nfs'].get('working_dir', '~'))
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered to prevent blocking
                cwd=working_dir
            )

            # Stream output in real-time if verbose
            if verbose and process.stdout:
                stdout_lines = []
                for line in process.stdout:
                    sys.stdout.write(line)  # Print to terminal in real-time
                    sys.stdout.flush()  # Ensure immediate output, no backpressure
                    stdout_lines.append(line)
                process.wait(timeout=Timeouts.CADO_NFS)
                stdout = ''.join(stdout_lines)
            else:
                stdout, _ = process.communicate(timeout=Timeouts.CADO_NFS)

            results['raw_output'] = stdout

            # Parse factors
            parsed_factors = self.parse_cado_output(stdout)
            if parsed_factors:
                results['factors_found'] = [f[0] for f in parsed_factors]
                results['factor_found'] = parsed_factors[0][0]  # First factor

            results['success'] = process.returncode == 0 and len(parsed_factors) > 0

        except subprocess.TimeoutExpired:
            self.logger.error(f"CADO-NFS timed out after {Timeouts.CADO_NFS} seconds")
            if process:
                process.kill()
            results['success'] = False
            results['timeout'] = True
        except Exception as e:
            self.logger.exception(f"CADO-NFS execution failed: {e}")
            results['success'] = False
            results['error'] = str(e)

        results['execution_time'] = time.time() - start_time

        # Log found factors
        if results.get('factors_found'):
            for factor in results['factors_found']:
                self.log_factor_found(composite, factor, None, None, None,
                                    method='nfs', program="CADO-NFS")

        # Save raw output if configured
        if self.config['execution']['save_raw_output']:
            self.save_raw_output(results, 'cado-nfs')

        return results

    def get_program_version(self, program: str) -> str:
        """Override base class method to get CADO-NFS version."""
        return self.get_cado_version()

    def get_cado_version(self) -> str:
        """Get CADO-NFS version."""
        from lib.parsing_utils import get_binary_version
        return get_binary_version(
            self.config['programs']['cado_nfs']['path'],
            'cado',
            help_flag='--help',
            use_python=True
        )


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description='CADO-NFS wrapper for integer factorization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Factor a composite number
  python3 cado_wrapper.py --composite 1191913975959397481605242916777

  # Use specific number of workers
  python3 cado_wrapper.py --composite 1191913975959397481605242916777 --workers 8

  # Show CADO-NFS output in real-time
  python3 cado_wrapper.py --composite 1191913975959397481605242916777 -v

  # Test without API submission
  python3 cado_wrapper.py --composite 1191913975959397481605242916777 --no-submit
        """
    )

    parser.add_argument('--composite', type=str, required=True,
                       help='Composite number to factor')
    parser.add_argument('--workers', type=int,
                       help='Number of parallel workers (default: from config)')
    parser.add_argument('--config', type=str, default='client.yaml',
                       help='Configuration file path (default: client.yaml)')
    parser.add_argument('--project', type=str,
                       help='Project identifier for API submission')
    parser.add_argument('--no-submit', action='store_true',
                       help='Do not submit results to API')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show CADO-NFS output in real-time')

    args = parser.parse_args()

    wrapper = CADOWrapper(args.config)

    # Run CADO-NFS
    results = wrapper.run_cado_nfs(
        composite=args.composite,
        threads=args.workers,
        verbose=args.verbose
    )

    # Print results
    output = UserOutput()
    if results.get('factors_found'):
        output.section("Factorization successful!")
        output.item("Composite", args.composite)
        output.item("Factors", ' × '.join(results['factors_found']))
        output.item("Time", f"{results['execution_time']:.2f}s")
    else:
        output.error("Factorization failed")
        sys.exit(1)

    # Submit results unless disabled
    if not args.no_submit:
        program_name = 'cado-nfs'
        success = wrapper.submit_result(results, args.project, program_name)
        sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
