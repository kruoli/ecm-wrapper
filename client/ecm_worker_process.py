#!/usr/bin/env python3
"""
ECMWorkerProcess - Encapsulates single-process ECM execution with factor detection.

This module eliminates ~110 lines of duplicated subprocess logic in the global
_run_worker_ecm_process() function by providing a reusable, testable class.

Design note: multiprocessing requires pickleable functions, so we provide both:
- A class-based implementation (testable, maintainable)
- A thin global function wrapper (for multiprocessing compatibility)
"""
import subprocess
from typing import Optional, Dict, Any
from parsing_utils import parse_ecm_output, ECMPatterns


class ECMWorkerProcess:
    """Encapsulates single-process ECM execution for multiprocessing pools."""

    def __init__(self, worker_id: int, composite: str, b1: int, b2: Optional[int],
                 curves: int, verbose: bool, method: str, ecm_path: str):
        """
        Initialize ECM worker process.

        Args:
            worker_id: Worker identifier for logging
            composite: Composite number to factor
            b1: B1 parameter
            b2: B2 parameter (None to use GMP-ECM default)
            curves: Number of curves to run
            verbose: Enable verbose output
            method: Method name (ecm, pm1, pp1)
            ecm_path: Path to GMP-ECM binary
        """
        self.worker_id = worker_id
        self.composite = composite
        self.b1 = b1
        self.b2 = b2
        self.curves = curves
        self.verbose = verbose
        self.method = method
        self.ecm_path = ecm_path

    def execute(self, stop_event=None) -> Dict[str, Any]:
        """
        Execute ECM and return results.

        Args:
            stop_event: Optional multiprocessing.Event for early termination

        Returns:
            Dict with keys:
            - worker_id: Worker identifier
            - factor_found: Factor string or None
            - sigma_found: Sigma value that found the factor
            - sigma_values: List of all sigma values used
            - curves_completed: Number of curves completed
            - raw_output: Full program output
        """
        # Build command for this worker
        cmd = [self.ecm_path]

        # Add method-specific parameters
        if self.method == "pm1":
            cmd.append('-pm1')
        elif self.method == "pp1":
            cmd.append('-pp1')

        if self.verbose:
            cmd.append('-v')

        # Run specified number of curves
        cmd.extend(['-c', str(self.curves), str(self.b1)])
        if self.b2 is not None:
            cmd.append(str(self.b2))

        try:
            print(f"Worker {self.worker_id} starting {self.curves} curves")
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            # Send composite number
            if process.stdin:
                process.stdin.write(self.composite)
                process.stdin.close()

            # Stream output and collect results
            output_lines = []
            curves_completed = 0
            factor_found = None
            sigma_found = None
            sigma_values = []  # Collect all sigma values used

            while True:
                # Check stop event
                if stop_event and stop_event.is_set():
                    process.terminate()
                    break

                if not process.stdout:
                    break
                line = process.stdout.readline()
                if not line:
                    break
                line = line.rstrip()
                if line:
                    print(f"Worker {self.worker_id}: {line}")
                    output_lines.append(line)

                    # Track progress and check for factors
                    if "Step 1 took" in line:
                        curves_completed += 1

                    # Collect sigma values from curve output
                    # Match both formats: "sigma=1:xxxx" and "-sigma 3:xxxx"
                    sigma_match = ECMPatterns.SIGMA_COLON_FORMAT.search(line) or \
                                 ECMPatterns.SIGMA_DASH_FORMAT.search(line)
                    if sigma_match:
                        sigma_val = sigma_match.group(1)
                        if sigma_val not in sigma_values:
                            sigma_values.append(sigma_val)

                    # Check for factor
                    if not factor_found:
                        factor, sigma = parse_ecm_output(line)
                        if factor:
                            factor_found = factor
                            sigma_found = sigma
                            break  # Stop immediately when factor found

            process.wait()

            # If no factor found during streaming, check full output
            if not factor_found and (not stop_event or not stop_event.is_set()):
                full_output = '\n'.join(output_lines)
                factor_found, sigma_found = parse_ecm_output(full_output)
                curves_completed = self.curves  # All curves completed

            return {
                'worker_id': self.worker_id,
                'factor_found': factor_found,
                'sigma_found': sigma_found,  # Sigma of the curve that found the factor
                'sigma_values': sigma_values,  # All sigma values used by this worker
                'curves_completed': curves_completed if not stop_event or not stop_event.is_set() else curves_completed,
                'raw_output': '\n'.join(output_lines)
            }

        except Exception as e:
            print(f"Worker {self.worker_id} failed: {e}")
            return {
                'worker_id': self.worker_id,
                'factor_found': None,
                'sigma_found': None,
                'sigma_values': [],
                'curves_completed': 0,
                'raw_output': f"Worker failed: {e}"
            }


def run_worker_ecm_process(worker_id: int, composite: str, b1: int, b2: Optional[int],
                           curves: int, verbose: bool, method: str, ecm_path: str,
                           result_queue, stop_event) -> None:
    """
    Global wrapper function for multiprocessing compatibility.

    This thin wrapper allows ECMWorkerProcess to be used with multiprocessing
    by providing a pickleable top-level function.
    """
    worker = ECMWorkerProcess(worker_id, composite, b1, b2, curves, verbose, method, ecm_path)
    result = worker.execute(stop_event)
    result_queue.put(result)
