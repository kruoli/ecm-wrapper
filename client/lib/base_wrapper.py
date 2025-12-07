#!/usr/bin/env python3
"""
Base wrapper class containing shared functionality for ECM and YAFU wrappers.
"""
import datetime
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, TYPE_CHECKING

from .config_manager import ConfigManager
from .api_client import APIClient
from .file_utils import save_json, load_json

if TYPE_CHECKING:
    from .typed_config import AppConfig


class BaseWrapper:
    """Base class for factorization wrappers with common functionality."""

    def __init__(self, config_path: str):
        """Initialize wrapper with configuration."""
        self._validate_working_directory()

        # Store config path for lazy typed config loading
        self._config_path = config_path
        self._typed_config: Optional['AppConfig'] = None

        # Load configuration using ConfigManager
        config_manager = ConfigManager()
        self.config = config_manager.load_config(config_path)

        self.setup_logging()
        # Construct client_id from username and cpu_name
        username = self.config['client']['username']
        cpu_name = self.config['client']['cpu_name']
        self.client_id = f"{username}-{cpu_name}"

        # Defer API client initialization until first use (lazy loading)
        self.api_clients = None
        self.api_client = None

    @property
    def typed_config(self) -> 'AppConfig':
        """
        Get typed configuration object (lazy-loaded).

        Provides type-safe access to configuration values with IDE
        autocompletion. Use this instead of self.config for new code.

        Returns:
            Typed AppConfig instance

        Example:
            path = self.typed_config.programs.gmp_ecm.path
            timeout = self.typed_config.api.timeout
        """
        if self._typed_config is None:
            from .typed_config import TypedConfigLoader
            loader = TypedConfigLoader()
            self._typed_config = loader.load(self._config_path)
        assert self._typed_config is not None  # For type checker
        return self._typed_config

    def _ensure_api_clients(self):
        """Initialize API clients on first use (lazy loading)."""
        if self.api_clients is not None:
            return  # Already initialized

        self.api_clients = []
        api_config = self.config['api']

        # Check if multiple endpoints are configured
        if 'endpoints' in api_config and api_config['endpoints']:
            # Multiple endpoints mode
            for endpoint_config in api_config['endpoints']:
                client = APIClient(
                    api_endpoint=endpoint_config['url'],
                    timeout=api_config['timeout'],
                    retry_attempts=api_config['retry_attempts']
                )
                self.api_clients.append({
                    'client': client,
                    'name': endpoint_config.get('name', endpoint_config['url']),
                    'url': endpoint_config['url']
                })
            self.logger.info(f"Configured {len(self.api_clients)} API endpoints: {', '.join([c['name'] for c in self.api_clients])}")
        else:
            # Single endpoint mode (backward compatibility)
            self.api_endpoint = api_config['endpoint']
            client = APIClient(
                api_endpoint=self.api_endpoint,
                timeout=api_config['timeout'],
                retry_attempts=api_config['retry_attempts']
            )
            self.api_clients.append({
                'client': client,
                'name': 'default',
                'url': self.api_endpoint
            })

        # Keep backward compatibility reference to first client
        self.api_client = self.api_clients[0]['client']

    def _get_api_client(self) -> 'APIClient':
        """
        Get the API client, ensuring it's initialized.

        Returns:
            The primary APIClient instance

        Raises:
            RuntimeError: If API client cannot be initialized
        """
        self._ensure_api_clients()
        if self.api_client is None:
            raise RuntimeError("API client failed to initialize")
        return self.api_client

    def _validate_working_directory(self):
        """Validate that we're running from the correct directory."""
        current_dir = Path.cwd()

        # Check if we're in the client directory by looking for key files
        expected_files = ['ecm_wrapper.py', 'ecm_client.py', 'yafu_wrapper.py', 'client.yaml', 'lib']
        missing_files = [f for f in expected_files if not (current_dir / f).exists()]

        if missing_files:
            print("🚨 WARNING: You appear to be running from the wrong directory!")
            print(f"   Current directory: {current_dir}")
            print("   Expected to be in: .../ecm-wrapper/client/")
            print(f"   Missing files: {', '.join(missing_files)}")
            print("   This may cause issues with file paths and data organization.")
            print("   Please run from the client/ directory for proper operation.\n")

            # Also check if we're one level up (in ecm-wrapper root)
            if (current_dir / 'client').exists():
                print("💡 TIP: Try running: cd client && python3 ecm_wrapper.py [args]")
            print()

    def setup_logging(self):
        """Set up logging configuration."""
        # Get logging config with defaults
        logging_config = self.config.get('logging', {})
        log_file_path = logging_config.get('file', 'data/logs/ecm_client.log')
        log_level = logging_config.get('level', 'INFO')

        log_file = Path(log_file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log_factor_found(self, composite: str, factor: str, b1: Optional[int],
                        b2: Optional[int], curves: Optional[int],
                        method: str = "ecm", sigma: Optional[str] = None,
                        program: str = "unknown", quiet: bool = False):
        """Log found factors to a dedicated factors file.

        Args:
            quiet: If True, skip console output (still logs to file)
        """
        # Check if factor logging is enabled in config
        log_factors = self.config.get('logging', {}).get('log_factors_found', True)
        if not log_factors:
            return

        factors_file = Path("data/factors_found.txt")
        factors_file.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Write to text file
        with open(factors_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"FACTOR FOUND: {timestamp}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Composite ({len(composite)} digits): {composite}\n")
            f.write(f"Factor: {factor}\n")
            if b1 is not None:
                f.write(f"Parameters: B1={b1}, B2={b2}, Curves={curves}")
                if sigma is not None:
                    f.write(f", Sigma={sigma}")
                f.write("\n")
            f.write(f"Program: {program} ({method.upper()} mode)\n")
            f.write(f"{'='*80}\n\n")

        # Also write to JSON file
        factors_json_file = Path("data/factors.json")

        # Load existing entries
        factors_data = load_json(factors_json_file, default=[])

        # Create new entry
        entry = {
            "timestamp": timestamp,
            "composite": composite,
            "composite_digits": len(composite),
            "factor": factor,
            "method": method.upper(),
            "program": program
        }

        if b1 is not None:
            entry["b1"] = b1
        if b2 is not None:
            entry["b2"] = b2
        if curves is not None:
            entry["curves"] = curves
        if sigma is not None:
            entry["sigma"] = sigma

        factors_data.append(entry)

        # Write back to JSON file
        save_json(factors_json_file, factors_data)

        # Also log to console with highlight (unless quiet mode)
        if not quiet:
            print(f"\n🎉 FACTOR FOUND: {factor}")
            print(f"📋 Logged to: {factors_file}")
            print(f"📋 JSON: {factors_json_file}")

    def submit_payload_to_endpoints(self, payload: Dict[str, Any],
                                    save_on_failure: bool = True,
                                    results_context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Submit a pre-built payload to all configured API endpoints.

        Args:
            payload: Pre-built API submission payload
            save_on_failure: Whether to save failed submissions to disk
            results_context: Optional full results dict for failure persistence

        Returns:
            Response dictionary from first successful submission (contains attempt_id, composite_id, etc.)
            Returns None if all submissions failed
        """
        # Ensure API clients are initialized
        if not self.api_clients:
            self._ensure_api_clients()
        assert self.api_clients is not None  # For type checker

        submission_results = []
        first_success_response = None

        for api_client_info in self.api_clients:
            api_client = api_client_info['client']
            endpoint_name = api_client_info['name']

            try:
                self.logger.info(f"Submitting to {endpoint_name} ({api_client_info['url']})")
                response = api_client.submit_result(
                    payload=payload,
                    save_on_failure=save_on_failure,
                    results_context=results_context
                )

                if response:
                    self.logger.info(f"✓ Successfully submitted to {endpoint_name}")
                    submission_results.append(True)
                    # Keep track of first successful response (contains attempt_id)
                    if first_success_response is None:
                        first_success_response = response
                else:
                    self.logger.warning(f"✗ Failed to submit to {endpoint_name}")
                    submission_results.append(False)

            except Exception as e:
                self.logger.error(f"✗ Error submitting to {endpoint_name}: {str(e)}")
                submission_results.append(False)

        # Log summary if multiple endpoints
        if len(self.api_clients) > 1:
            success_count = sum(submission_results)
            total_count = len(submission_results)
            self.logger.info(f"Submission summary: {success_count}/{total_count} endpoints succeeded")

        # Return first successful response (or None if all failed)
        return first_success_response

    def submit_result(self, results: Dict[str, Any], project: Optional[str] = None,
                     program: str = "unknown") -> Optional[Dict[str, Any]]:
        """
        Submit results to API endpoint(s) with retry logic.

        If multiple endpoints are configured, submits to all of them.
        Returns response from first successful submission (contains attempt_id).
        """
        self._ensure_api_clients()  # Lazy load API clients on first use
        assert self.api_clients is not None  # For type checker

        # Build payload once (same for all endpoints)
        payload = self.api_clients[0]['client'].build_submission_payload(
            composite=results['composite'],
            client_id=self.client_id,
            method=results.get('method', 'ecm'),
            program=program,
            program_version=self.get_program_version(program),
            results=results,
            project=project
        )

        # Submit to all configured endpoints
        return self.submit_payload_to_endpoints(
            payload=payload,
            save_on_failure=True,
            results_context=results
        )

    def abandon_work(self, work_id: str, reason: str = "client_terminated") -> bool:
        """
        Abandon work assignment (convenience wrapper that automatically passes client_id).

        Args:
            work_id: Work assignment ID to abandon
            reason: Reason for abandoning (optional)

        Returns:
            True if successfully abandoned, False otherwise
        """
        self._ensure_api_clients()  # Lazy load API clients on first use
        assert self.api_client is not None  # For type checker
        return self.api_client.abandon_work(work_id, reason=reason, client_id=self.client_id)

    def run_subprocess_with_parsing(self, cmd: List[str],
                                  composite: str, method: str,
                                  parse_function: Callable,
                                  **kwargs) -> Dict[str, Any]:
        """
        Unified subprocess execution with parsing for factorization programs.

        This method provides comprehensive error handling for subprocess execution,
        including I/O errors and parsing errors. Subclasses should use this method
        instead of reimplementing subprocess execution and error handling.

        Note: No timeout is enforced - ECM factorization can run for extended periods.

        The method handles:
        - Subprocess execution errors (subprocess.SubprocessError)
        - I/O errors (OSError, IOError)
        - Invalid parameters (ValueError)
        - Unexpected errors (general Exception with traceback logging)

        Args:
            cmd: Command list to execute
            composite: Number being factored
            method: Method name (ecm, pm1, pp1, etc.)
            parse_function: Function to parse output (factor, sigma)
            **kwargs: Additional parameters for results

        Returns:
            Dictionary with standardized results including success status, factors,
            execution time, and error details if applicable

        Note:
            Subclasses should NOT add redundant try/except blocks around calls to
            this method, as all error cases are already handled internally.
        """
        from lib.subprocess_utils import execute_subprocess
        from lib.results_builder import ResultsBuilder

        start_time = time.time()

        # Build results using ResultsBuilder
        builder = ResultsBuilder(composite, method)
        if 'b1' in kwargs:
            builder.with_b1(kwargs['b1'])
        if 'b2' in kwargs:
            builder.with_b2(kwargs['b2'])
        if 'curves' in kwargs:
            builder.with_curves(kwargs['curves'])
        if 'sigma' in kwargs:
            builder.with_sigma(kwargs['sigma'])
        results = builder.build_no_truncate()

        # Add any extra kwargs to results
        for key, value in kwargs.items():
            if key not in ['b1', 'b2', 'curves', 'sigma', 'verbose', 'input']:
                results[key] = value

        verbose = kwargs.get('verbose', False)

        try:
            self.logger.info(f"Running {method.upper()} on {len(composite)}-digit number with {' '.join(cmd[1:3])}")

            # Get program input if specified
            program_input = kwargs.get('input')

            # Execute subprocess using unified utility
            # Check explicitly for None to allow empty string override
            result = execute_subprocess(
                cmd=cmd,
                composite=program_input if program_input is not None else composite,
                verbose=verbose
            )

            stdout = result['stdout']
            results['raw_output'] = stdout

            # Parse for factors using provided function
            parse_result = parse_function(stdout)
            if isinstance(parse_result, tuple) and len(parse_result) == 2:
                # Single factor result (factor, sigma) - from ECM parsing
                factor, sigma = parse_result
                if factor:
                    results['factor_found'] = factor
                    results['factors_found'] = [factor]
                    results['sigma'] = sigma
            elif isinstance(parse_result, list):
                # Multiple factors result [(factor, sigma), ...] - from YAFU parsing
                if parse_result:
                    results['factors_found'] = [f[0] for f in parse_result]
                    results['factor_found'] = parse_result[0][0]  # First factor
                    results['sigma'] = parse_result[0][1] if parse_result[0][1] else None

            # Parse curves completed for YAFU methods
            if kwargs.get('track_curves', False):
                from .parsing_utils import YAFUPatterns
                curves_match = YAFUPatterns.CURVES_COMPLETED.search(stdout)
                if curves_match:
                    results['curves_completed'] = int(curves_match.group(1))
                else:
                    # Try alternative format
                    progress_match = YAFUPatterns.CURVE_PROGRESS.search(stdout)
                    if progress_match:
                        results['curves_completed'] = int(progress_match.group(1))

            results['success'] = result['returncode'] == 0

        except subprocess.SubprocessError as e:
            self.logger.error(f"{method.upper()} subprocess execution failed: {e}")
            results['success'] = False
            results['error'] = str(e)
        except (OSError, IOError) as e:
            self.logger.error(f"{method.upper()} I/O error: {e}")
            results['success'] = False
            results['error'] = f"I/O error: {str(e)}"
        except ValueError as e:
            self.logger.error(f"{method.upper()} invalid parameter: {e}")
            results['success'] = False
            results['error'] = f"Invalid parameter: {str(e)}"
        except Exception as e:
            # Catch-all for unexpected errors - log with full traceback for debugging
            self.logger.exception(f"{method.upper()} unexpected error: {e}")
            results['success'] = False
            results['error'] = f"Unexpected error: {type(e).__name__}"

        results['execution_time'] = time.time() - start_time
        return results

    def _stream_subprocess_output(self, cmd: List[str], composite: Optional[str],
                                   log_prefix: str, line_callback: Optional[Callable] = None) -> tuple[subprocess.Popen, List[str]]:
        """
        Stream subprocess output with live logging and optional per-line processing.

        Args:
            cmd: Command to execute
            composite: Optional composite number to send to stdin
            log_prefix: Prefix for log messages (e.g., "ECM", "Stage1")
            line_callback: Optional function called for each line, receives (line, output_lines_so_far)

        Returns:
            Tuple of (process, output_lines)
        """
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Send composite to stdin if provided
        if composite and process.stdin:
            process.stdin.write(composite)
        if process.stdin:
            process.stdin.close()

        # Stream output
        output_lines: List[str] = []
        if not process.stdout:
            return process, output_lines

        try:
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                line = line.rstrip()
                if line:
                    self.logger.info(f"{log_prefix}: {line}")
                    output_lines.append(line)

                    # Call optional per-line callback
                    if line_callback:
                        line_callback(line, output_lines)

            process.wait()
        except KeyboardInterrupt:
            self.logger.info(f"{log_prefix}: Interrupted by user, terminating subprocess...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            # Set interrupted flag if available (ECMWrapper has this, BaseWrapper may not)
            if hasattr(self, 'interrupted'):
                self.interrupted = True
            raise  # Re-raise to let caller know about the interrupt

        return process, output_lines

    def get_program_version(self, program: str) -> str:
        """Get program version - to be implemented by subclasses."""
        return "unknown"

    def save_raw_output(self, results: Dict[str, Any], program: str = "unknown") -> None:
        """Save raw output to file for debugging."""
        output_dir = Path(self.config['execution']['output_dir'])
        output_dir.mkdir(exist_ok=True, parents=True)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        method = results.get('method', 'unknown')
        curves = results.get('curves_completed', 0)
        filename = output_dir / f"{program}_{method}_{timestamp}_{curves}curves.txt"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Composite: {results['composite']}\n")
            if 'b1' in results:
                f.write(f"B1: {results['b1']}, B2: {results.get('b2')}\n")
            f.write(f"Method: {method}\n")
            f.write(f"Program: {program}\n")

            # Handle both single factor and multiple factors
            if 'factor_found' in results:
                f.write(f"Factor found: {results['factor_found']}\n")
            elif 'factors_found' in results:
                f.write(f"Factors found: {results['factors_found']}\n")

            f.write(f"Curves completed: {results.get('curves_completed', 0)}\n")
            f.write(f"Execution time: {results.get('execution_time', 0):.2f}s\n\n")
            f.write(results.get('raw_output', ''))
