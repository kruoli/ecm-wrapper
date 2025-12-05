#!/usr/bin/env python3
import subprocess
import time
import sys
import signal
import threading
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from lib.base_wrapper import BaseWrapper
from lib.parsing_utils import ECMPatterns
from lib.residue_manager import ResidueFileManager
from lib.result_processor import ResultProcessor
from lib.results_builder import results_for_stage1
from lib.stage2_executor import Stage2Executor
from lib.ecm_worker_process import run_worker_ecm_process
from lib.ecm_arg_helpers import parse_sigma_arg, resolve_param, resolve_stage2_workers
from lib.work_helpers import print_work_header, print_work_status, request_ecm_work
from lib.stage1_helpers import submit_stage1_complete_workflow
from lib.error_helpers import handle_work_failure, check_work_limit_reached
from lib.cleanup_helpers import handle_shutdown

# New modularized utilities
from lib.ecm_config import ECMConfig, TwoStageConfig, MultiprocessConfig, TLevelConfig, FactorResult
from lib.ecm_math import (
    trial_division, is_probably_prime, get_b1_for_digit_length
)

class ECMWrapper(BaseWrapper):
    """
    Wrapper for GMP-ECM factorization with multiple execution modes.

    Supports:
    - Standard ECM execution (run_ecm_v2)
    - Two-stage GPU/CPU pipeline (run_two_stage_v2)
    - Multiprocess parallelization (run_multiprocess_v2)
    - Progressive t-level targeting (run_tlevel_v2)
    - Server-coordinated auto-work mode

    All methods use type-safe configuration objects (ECMConfig, TwoStageConfig, etc.)
    and return FactorResult objects with discovered factors and metadata.
    """

    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.residue_manager = ResidueFileManager()
        # Initialize new executor for config-based methods
                # Graceful shutdown support
        self.stop_event = threading.Event()
        self.interrupted = False

    # ==================== NEW CONFIG-BASED METHODS ====================
    # These methods use configuration objects for cleaner interfaces

    def run_ecm_v2(self, config: ECMConfig) -> FactorResult:
        """
        Execute ECM with configuration object (simplified interface).

        This is the new, recommended method that uses ECMConfig dataclass.
        It has better type safety, validation, and testability.

        Delegates to run_multiprocess_v2 with num_processes=1, which provides:
        - Automatic stop on first factor found
        - Accurate curve counting (not just requested count)
        - Proven, well-tested implementation

        Args:
            config: ECM configuration object

        Returns:
            FactorResult with discovered factors and metadata

        Example:
            >>> config = ECMConfig(composite="123456789", b1=50000, curves=100)
            >>> result = wrapper.run_ecm_v2(config)
            >>> if result.success:
            ...     print(f"Found factors: {result.factors}")
        """
        # Delegate to multiprocess with 1 worker for proper factor detection and curve counting
        mp_config = MultiprocessConfig(
            composite=config.composite,
            b1=config.b1,
            b2=config.b2,
            total_curves=config.curves,
            curves_per_process=config.curves,
            num_processes=1,
            method=config.method,
            verbose=config.verbose
        )
        return self.run_multiprocess_v2(mp_config)

    def run_ecm_from_dict(self, params: Dict[str, Any]) -> FactorResult:
        """
        Execute ECM from dictionary parameters.

        Convenient wrapper for config-based execution when you have
        a dictionary of parameters (e.g., from JSON config file).

        Args:
            params: Dictionary with ECMConfig fields

        Returns:
            FactorResult object

        Example:
            >>> params = {"composite": "12345", "b1": 50000, "curves": 10}
            >>> result = wrapper.run_ecm_from_dict(params)
        """
        config = ECMConfig(**params)
        return self.run_ecm_v2(config)

    def run_two_stage_v2(self, config: TwoStageConfig) -> FactorResult:
        """
        Execute two-stage ECM pipeline with configuration object.

        Stage 1: GPU-accelerated residue generation
        Stage 2: CPU processing of residues

        Args:
            config: Two-stage configuration object
                   - If config.resume_file is provided, skip stage 1 and process existing residue

        Returns:
            FactorResult with discovered factors

        Example:
            >>> # Full two-stage pipeline
            >>> config = TwoStageConfig(
            ...     composite="12345...",
            ...     b1=50000,
            ...     stage1_curves=100,
            ...     stage2_curves_per_residue=1000
            ... )
            >>> result = wrapper.run_two_stage_v2(config)
            >>>
            >>> # Resume from existing residue (stage 2 only)
            >>> config = TwoStageConfig(
            ...     composite="12345...",
            ...     b1=50000,
            ...     resume_file="path/to/residue.txt"
            ... )
            >>> result = wrapper.run_two_stage_v2(config)
        """
        start_time = time.time()

        # Check if resuming from existing residue file
        if config.resume_file:
            residue_file = Path(config.resume_file)
            if not residue_file.exists():
                self.logger.error(f"Resume file not found: {residue_file}")
                result = FactorResult()
                result.success = False
                result.execution_time = time.time() - start_time
                return result

            # Parse residue file for metadata
            residue_info = self._parse_residue_file(residue_file)
            _composite = residue_info['composite']
            stage1_curves = residue_info['curve_count']
            stage1_b1 = residue_info['b1']

            # Use B1 from residue file if available
            if stage1_b1 > 0:
                if stage1_b1 != config.b1:
                    self.logger.info(f"Using B1={stage1_b1} from residue file (overriding config B1={config.b1})")
                actual_b1 = stage1_b1
            else:
                actual_b1 = config.b1

            self.logger.info(f"Resuming from residue file: {residue_file}")
            self.logger.info(f"Stage 1 completed {stage1_curves} curves at B1={actual_b1}")

            # Skip stage 1, go directly to stage 2
            stage1_result = None
        else:
            # Normal flow: Setup residue file and run stage 1
            residue_file = self._create_residue_path(config)
            actual_b1 = config.b1
            stage1_curves = 0

            # Stage 1: GPU execution
            stage1_result = self._run_stage1_primitive(
                composite=config.composite,
                b1=config.b1,
                curves=config.stage1_curves,
                residue_file=residue_file,
                use_gpu=(config.stage1_device == "GPU"),
                param=config.stage1_parametrization,
                verbose=config.verbose,
                timeout=config.timeout_stage1
            )

            # Early return if factor found in stage 1
            if stage1_result['factors']:
                return self._build_factor_result(stage1_result, start_time)

            # Skip stage 2 if b2=0 (stage 1 only mode)
            if config.b2 == 0:
                self.logger.info("B2=0: Skipping stage 2 (stage 1 only mode)")
                return self._build_factor_result(stage1_result, start_time)

            stage1_curves = stage1_result.get('curves_run', 0)

        # Stage 2: Multi-threaded CPU processing
        # _run_stage2_multithread returns: (factor, all_factors, curves_completed, execution_time, sigma)
        _factor, all_factors, curves_completed, _stage2_time, sigma = self._run_stage2_multithread(
            residue_file=residue_file,
            b1=actual_b1,
            b2=config.b2 or (actual_b1 * 100),  # Default B2
            workers=config.threads,
            verbose=config.verbose
        )

        # Build FactorResult from stage 2 results
        result = FactorResult()
        if all_factors:
            for f in all_factors:
                result.add_factor(f, sigma)
        result.curves_run = curves_completed
        result.execution_time = time.time() - start_time
        result.success = len(all_factors) > 0

        return result

    def run_multiprocess_v2(self, config: MultiprocessConfig) -> FactorResult:
        """
        Execute multiprocess ECM with configuration object.

        Distributes curves across multiple CPU cores for parallel execution.

        Args:
            config: Multiprocess configuration object

        Returns:
            FactorResult with discovered factors

        Example:
            >>> config = MultiprocessConfig(
            ...     composite="12345...",
            ...     b1=50000,
            ...     total_curves=1000,
            ...     curves_per_process=100
            ... )
            >>> result = wrapper.run_multiprocess_v2(config)
        """
        import multiprocessing as mp

        # Ensure num_processes is set (should be auto-set in __post_init__)
        num_processes = config.num_processes or mp.cpu_count()

        self.logger.info(f"Running multi-process ECM: {num_processes} workers, {config.total_curves} total curves")
        start_time = time.time()

        # Distribute curves across workers
        curves_per_worker = config.total_curves // num_processes
        remaining_curves = config.total_curves % num_processes
        worker_assignments = []

        for worker_id in range(num_processes):
            worker_curves = curves_per_worker + (1 if worker_id < remaining_curves else 0)
            if worker_curves > 0:
                worker_assignments.append((worker_id + 1, worker_curves))

        # Create shared variables for worker coordination
        manager = mp.Manager()
        result_queue = manager.Queue()
        progress_queue = manager.Queue()
        stop_event = manager.Event()

        # Start worker processes
        processes = []
        for worker_id, worker_curves in worker_assignments:
            p = mp.Process(
                target=run_worker_ecm_process,
                args=(worker_id, config.composite, config.b1, config.b2, worker_curves,
                      config.verbose, config.method, self.config['programs']['gmp_ecm']['path'],
                      result_queue, stop_event, 0, progress_queue)
            )
            p.start()
            processes.append(p)

        # Collect results from workers
        all_factors = []
        all_sigmas = []
        all_raw_outputs = []
        total_curves_completed = 0
        completed_workers = 0

        while completed_workers < len(processes):
            try:
                result = result_queue.get(timeout=0.5)
                total_curves_completed += result['curves_completed']

                if result['factor_found']:
                    all_factors.append(result['factor_found'])
                    all_sigmas.append(result.get('sigma_found'))
                    self.logger.info(f"Worker {result['worker_id']} found factor: {result['factor_found']}")
                    stop_event.set()  # Signal workers to stop on factor found

                # Collect raw output from each worker
                if 'raw_output' in result:
                    all_raw_outputs.append(f"=== Worker {result['worker_id']} ===\n{result['raw_output']}")

                completed_workers += 1
            except:
                # Check if processes are still alive
                if not any(p.is_alive() for p in processes):
                    break
                continue

        # Wait for all processes to finish
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

        # Build FactorResult
        result = FactorResult()
        for factor, sigma in zip(all_factors, all_sigmas):
            result.add_factor(factor, sigma)
        result.curves_run = total_curves_completed
        result.execution_time = time.time() - start_time
        result.success = len(all_factors) > 0
        result.raw_output = '\n\n'.join(all_raw_outputs) if all_raw_outputs else None

        return result

    def run_tlevel_v2(self, config: TLevelConfig) -> FactorResult:
        """
        Execute T-level targeting with configuration object.

        Progressively runs ECM with optimized B1 values to reach target t-level.

        Args:
            config: T-level configuration object

        Returns:
            FactorResult with discovered factors

        Example:
            >>> config = TLevelConfig(
            ...     composite="12345...",
            ...     target_t_level=30.0,
            ...     b1_strategy='optimal'
            ... )
            >>> result = wrapper.run_tlevel_v2(config)
        """
        from lib.ecm_math import (get_optimal_b1_for_tlevel, calculate_tlevel,
                                  calculate_curves_to_target_direct, TLEVEL_TRANSITION_CACHE)

        self.logger.info(f"Starting progressive ECM with target t{config.target_t_level:.1f}")
        start_time = time.time()

        # Accumulated results
        all_factors = []
        all_sigmas = []
        total_curves = 0
        curve_history = []  # Track B1 values and curves for t-level calculation

        # Progressive loop: start at t20 and work up to target
        current_t_level = 0.0
        step_targets = []

        # Build step targets (every 5 t-levels from 20 to target, plus final target)
        t = 20.0
        while t < config.target_t_level:
            step_targets.append(t)
            t += 5.0
        # Always add the final target (might not be a multiple of 5)
        if config.target_t_level not in step_targets:
            step_targets.append(config.target_t_level)

        # Run ECM at each step
        for step_target in step_targets:
            # Get optimal B1 for this t-level
            b1, _ = get_optimal_b1_for_tlevel(step_target)

            # Calculate curves needed to reach this step from current position
            # First try the cached table for standard 5-digit increments
            current_rounded = round(current_t_level)
            target_rounded = round(step_target)
            cache_key = (current_rounded, target_rounded, config.parametrization)

            if cache_key in TLEVEL_TRANSITION_CACHE:
                # Use cached value for standard transition
                cached_b1, curves = TLEVEL_TRANSITION_CACHE[cache_key]
                if cached_b1 == b1:  # Only use cache if B1 matches
                    self.logger.info(f"Using cached transition: t{current_rounded} → t{target_rounded} = {curves} curves at B1={b1}, p={config.parametrization}")
                else:
                    # B1 doesn't match, call binary
                    curves = calculate_curves_to_target_direct(current_t_level, step_target, b1, config.parametrization)
            else:
                # Non-standard transition, call t-level binary directly
                curves = calculate_curves_to_target_direct(current_t_level, step_target, b1, config.parametrization)

            if curves is None or curves <= 0:
                self.logger.warning(f"Could not calculate curves for t{current_t_level:.3f} → t{step_target:.1f}, using Zimmermann estimate")
                # Fallback to Zimmermann table estimate
                _, curves = get_optimal_b1_for_tlevel(step_target)

            self.logger.info(f"Running {curves} curves at B1={b1} (targeting t{step_target:.1f}, currently at t{current_t_level:.2f})")

            # Run this batch (use multiprocess if workers > 1)
            if config.threads > 1:
                # Multiprocess mode
                self.logger.info(f"Using multiprocess mode with {config.threads} workers")
                step_result = self.run_multiprocess_v2(MultiprocessConfig(
                    composite=config.composite,
                    b1=b1,
                    total_curves=curves,
                    num_processes=config.threads,
                    parametrization=config.parametrization,
                    verbose=config.verbose,
                    timeout=config.timeout
                ))
            else:
                # Single process mode
                step_result = self.run_ecm_v2(ECMConfig(
                    composite=config.composite,
                    b1=b1,
                    curves=curves,
                    parametrization=config.parametrization,
                    threads=1,
                    verbose=config.verbose,
                    timeout=config.timeout
                ))

            # Accumulate results
            all_factors.extend(step_result.factors)
            all_sigmas.extend(step_result.sigmas)
            total_curves += step_result.curves_run

            # Track curve history for t-level calculation (format: "curves@b1,p=parametrization")
            curve_history.append(f"{step_result.curves_run}@{b1},p={config.parametrization}")

            # Submit this batch's results if enabled
            if not config.no_submit and step_result.curves_run > 0:
                step_results = {
                    'success': True,
                    'factors_found': step_result.factors,
                    'curves_completed': step_result.curves_run,
                    'execution_time': step_result.execution_time,
                    'raw_output': step_result.raw_output,
                    'composite': config.composite,
                    'method': 'ecm',
                    'b1': b1,
                    'b2': None  # T-level mode uses default B2
                }
                if config.work_id:
                    step_results['work_id'] = config.work_id

                program_name = 'gmp-ecm-ecm'
                submit_response = self.submit_result(step_results, config.project, program_name)
                if not submit_response:
                    self.logger.warning(f"Failed to submit results for B1={b1}")

            # Break if factor found
            if step_result.factors:
                self.logger.info(f"Factor found after {total_curves} curves")
                # Build and return result immediately
                result = FactorResult()
                for factor, sigma in zip(all_factors, all_sigmas):
                    result.add_factor(factor, sigma)
                result.curves_run = total_curves
                result.execution_time = time.time() - start_time
                result.success = True
                return result

            # Update current t-level after this batch
            try:
                current_t_level = calculate_tlevel(curve_history)
                self.logger.info(f"Current t-level: {current_t_level:.2f}")
            except:
                # If t-level calculation fails, approximate it
                current_t_level = step_target

            # Check if we've reached overall target
            if current_t_level >= config.target_t_level:
                self.logger.info(f"Reached target t-level: {current_t_level:.2f} >= {config.target_t_level:.1f}")
                break

        # Build final FactorResult
        result = FactorResult()
        for factor, sigma in zip(all_factors, all_sigmas):
            result.add_factor(factor, sigma)
        result.curves_run = total_curves
        result.execution_time = time.time() - start_time
        result.success = len(all_factors) > 0

        return result

    # ==================== LEGACY METHODS (BACKWARD COMPATIBLE) ====================

    def _log_and_store_factors(self, all_factors: List[Tuple[str, Optional[str]]],
                               results: Dict[str, Any], composite: str, b1: int,
                               b2: Optional[int], curves: int, method: str,
                               program: str) -> Optional[str]:
        """
        Deduplicate factors, log them, and store in results dictionary.

        This is now a thin wrapper around ResultProcessor for backward compatibility.

        Args:
            all_factors: List of (factor, sigma) tuples
            results: Results dictionary to update
            composite: Composite number being factored
            b1, b2, curves: ECM parameters
            method: Method name (ecm, pm1, pp1)
            program: Program name for logging

        Returns:
            First factor (for compatibility)
        """
        processor = ResultProcessor(self, composite, method, b1, b2, curves, program)
        return processor.log_and_store_factors(all_factors, results, quiet=False)

    # ==================== PRIMITIVE ECM EXECUTION ====================
    # Core primitive that all execution methods build upon

    def _execute_ecm_primitive(
        self,
        composite: str,
        b1: int,
        b2: Optional[int] = None,
        curves: int = 1,
        # Stage separation
        residue_save: Optional[Path] = None,
        residue_load: Optional[Path] = None,
        # ECM parameters
        sigma: Optional[Union[str, int]] = None,
        param: Optional[int] = None,
        method: str = "ecm",
        # GPU support
        use_gpu: bool = False,
        gpu_device: Optional[int] = None,
        gpu_curves: Optional[int] = None,
        # Progress and control
        progress_callback: Optional[Callable[[str, List[str]], None]] = None,
        stop_on_factor: bool = True,
        # Execution
        verbose: bool = False,
        timeout: int = 3600
    ) -> Dict[str, Any]:
        """
        Execute GMP-ECM primitive operation.

        Single source of truth for all ECM execution. Handles 3 modes:
        1. Full ECM (B1 + B2) - default
        2. Stage 1 only - when residue_save is set
        3. Stage 2 only - when residue_load is set

        Args:
            composite: Number to factor
            b1: B1 bound
            b2: B2 bound (None = GMP-ECM default, 0 = stage1 only)
            curves: Number of curves to run
            residue_save: Path to save residue file (stage 1 only)
            residue_load: Path to load residue from (stage 2 only)
            sigma: Sigma value
            param: Parametrization (0-3)
            method: Method ("ecm", "pm1", "pp1")
            use_gpu: Use GPU acceleration
            gpu_device: GPU device number
            gpu_curves: Curves per GPU batch
            progress_callback: Called for each output line with (line, all_lines)
            stop_on_factor: Stop when first factor found
            verbose: Verbose output
            timeout: Execution timeout in seconds

        Returns:
            {
                'success': bool,
                'factors': List[str],
                'sigmas': List[Optional[str]],
                'curves_completed': int,
                'raw_output': str,
                'parametrization': Optional[int],
                'exit_code': int,
                'interrupted': bool
            }
        """
        ecm_path = self.config['programs']['gmp_ecm']['path']

        # Build command
        cmd = [ecm_path]

        # Method-specific flags
        if method == "pm1":
            cmd.append('-pm1')
        elif method == "pp1":
            cmd.append('-pp1')

        # GPU flags (must come early)
        if use_gpu and method == "ecm":
            cmd.append('-gpu')
            if gpu_device is not None:
                cmd.extend(['-gpudevice', str(gpu_device)])
            if gpu_curves is not None:
                cmd.extend(['-gpucurves', str(gpu_curves)])

        # Residue operations
        if residue_save:
            cmd.extend(['-save', str(residue_save)])
        if residue_load:
            cmd.extend(['-resume', str(residue_load)])

        # Verbosity
        if verbose:
            cmd.append('-v')

        # Parametrization (ECM only)
        if param is not None and method == "ecm":
            cmd.extend(['-param', str(param)])

        # Sigma (ECM only)
        if sigma and method == "ecm":
            cmd.extend(['-sigma', str(sigma)])

        # Curves
        cmd.extend(['-c', str(curves)])

        # B1 parameter
        cmd.append(str(b1))

        # B2 parameter
        if b2 is not None:
            cmd.append(str(b2))

        # Execute subprocess with streaming
        try:
            process, output_lines = self._stream_subprocess_output(
                cmd=cmd,
                composite=composite if not residue_load else None,
                log_prefix=method.upper(),
                line_callback=progress_callback
            )

            exit_code = process.returncode
            raw_output = '\n'.join(output_lines)

            # Check for interruption
            interrupted = self.interrupted or exit_code == -15  # SIGTERM

            # Parse factors
            from lib.parsing_utils import parse_ecm_output_multiple
            factors_with_sigmas = parse_ecm_output_multiple(raw_output)

            # Separate factors and sigmas
            factors = [f[0] for f in factors_with_sigmas]
            sigmas = [f[1] for f in factors_with_sigmas]

            # Extract parametrization from output
            parametrization_extracted = None
            if sigmas and sigmas[0]:
                first_sigma = sigmas[0]
                if ':' in first_sigma:
                    parametrization_extracted = int(first_sigma.split(':')[0])
            if parametrization_extracted is None:
                parametrization_extracted = param if param is not None else 3

            # Determine success
            success = exit_code in [0, 8, 14] and not interrupted

            # Count curves completed (extract from GPU output, fallback to requested)
            curves_completed = curves  # Default fallback
            if use_gpu:
                # GPU mode: extract actual curve count from output
                curve_match = ECMPatterns.CURVE_COUNT.search(raw_output)
                if curve_match:
                    curves_completed = int(curve_match.group(1))
                    self.logger.debug(f"GPU completed {curves_completed} curves (requested {curves})")

            return {
                'success': success,
                'factors': factors,
                'sigmas': sigmas,
                'curves_completed': curves_completed,
                'raw_output': raw_output,
                'parametrization': parametrization_extracted,
                'exit_code': exit_code,
                'interrupted': interrupted
            }

        except subprocess.TimeoutExpired:
            self.logger.error(f"{method.upper()} execution timed out after {timeout}s")
            return {
                'success': False,
                'factors': [],
                'sigmas': [],
                'curves_completed': 0,
                'raw_output': '',
                'parametrization': param if param is not None else 3,
                'exit_code': -1,
                'interrupted': False
            }
        except Exception as e:
            self.logger.error(f"{method.upper()} execution failed: {e}")
            return {
                'success': False,
                'factors': [],
                'sigmas': [],
                'curves_completed': 0,
                'raw_output': str(e),
                'parametrization': param if param is not None else 3,
                'exit_code': -1,
                'interrupted': False
            }

    def _run_stage1_primitive(
        self,
        composite: str,
        b1: int,
        curves: int,
        residue_file: Path,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run stage 1 only, save residue to file.

        Helper wrapper around _execute_ecm_primitive() for stage 1 execution.

        Args:
            composite: Number to factor
            b1: B1 bound
            curves: Number of curves
            residue_file: Path to save residue file
            **kwargs: Pass through sigma, param, use_gpu, verbose, etc.

        Returns:
            Result dict from primitive
        """
        return self._execute_ecm_primitive(
            composite=composite,
            b1=b1,
            b2=0,  # Stage 1 only
            curves=curves,
            residue_save=residue_file,
            **kwargs
        )

    def _run_stage2_primitive(
        self,
        residue_file: Path,
        b1: int,
        b2: Optional[int],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run stage 2 from residue file.

        Helper wrapper around _execute_ecm_primitive() for stage 2 execution.
        Automatically parses residue metadata to extract composite and curve count.

        Args:
            residue_file: Path to residue file
            b1: B1 bound (must match stage1)
            b2: B2 bound for stage 2
            **kwargs: Pass through verbose, etc.

        Returns:
            Result dict from primitive
        """
        # Parse residue metadata to get composite and curve count
        metadata = self.residue_manager.parse_metadata(str(residue_file))
        if not metadata:
            raise ValueError(f"Could not parse residue file: {residue_file}")

        composite, _, curve_count = metadata

        return self._execute_ecm_primitive(
            composite=composite,
            b1=b1,
            b2=b2,
            curves=curve_count,
            residue_load=residue_file,
            **kwargs
        )

    def _build_factor_result(self, prim_result: Dict[str, Any], start_time: float) -> FactorResult:
        """
        Convert primitive result dict to FactorResult object.

        Recursively factors any composite factors found to ensure
        all returned factors are prime (or small composites).

        Args:
            prim_result: Result dictionary from _execute_ecm_primitive()
            start_time: Timestamp when execution started (from time.time())

        Returns:
            FactorResult object with populated fields
        """
        result = FactorResult()

        # Recursively factor each discovered factor to get all primes
        for factor, sigma in zip(prim_result['factors'], prim_result['sigmas']):
            # Fully factor this composite to get all prime factors
            prime_factors = self._fully_factor_found_result(factor, quiet=False)

            # Add each prime factor with the original sigma
            # (the sigma that found the composite factor)
            for prime in prime_factors:
                result.add_factor(prime, sigma)

        # Populate metadata fields
        result.curves_run = prim_result['curves_completed']
        result.execution_time = time.time() - start_time
        result.success = prim_result['success']
        result.raw_output = prim_result['raw_output']

        return result

    def _create_residue_path(self, config: TwoStageConfig) -> Path:
        """
        Create residue file path with auto-mkdir.

        If config specifies save_residues path, uses that. Otherwise creates
        an auto-generated path in the residue directory.

        Args:
            config: TwoStageConfig with optional save_residues path

        Returns:
            Path object for residue file
        """
        if config.save_residues:
            return Path(config.save_residues)

        # Auto-generate residue path
        residue_dir = Path(self.config['execution']['residue_dir'])
        residue_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        composite_hash = hashlib.md5(config.composite.encode()).hexdigest()[:8]
        return residue_dir / f"stage1_{composite_hash}_{timestamp}.txt"


    def _preserve_failed_upload(self, residue_file: Path) -> None:
        """
        Preserve a copy of a residue file that failed to upload.

        Saves the file to failed_uploads_dir with timestamp for manual retry later.

        Args:
            residue_file: Path to the residue file to preserve
        """
        import shutil
        import time

        try:
            failed_dir = Path(self.config['execution'].get('failed_uploads_dir', 'data/failed_uploads'))
            failed_dir.mkdir(parents=True, exist_ok=True)

            # Create filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            preserved_name = f"{residue_file.stem}_failed_{timestamp}{residue_file.suffix}"
            preserved_path = failed_dir / preserved_name

            # Copy the file
            shutil.copy2(residue_file, preserved_path)

            self.logger.info(f"Preserved failed upload: {preserved_path}")
            print(f"Residue file preserved for manual retry: {preserved_path}")

        except Exception as e:
            self.logger.error(f"Failed to preserve residue file: {e}")

    def _upload_residue_if_needed(
        self,
        residue_file: Path,
        stage1_attempt_id: Optional[str],
        factor_found: Optional[str],
        client_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Upload residue file to server only if no factor was found.

        When a factor is found in stage 1, there's no point in running stage 2,
        so we skip the residue upload.

        Args:
            residue_file: Path to the residue file
            stage1_attempt_id: Attempt ID from stage1 submission (None if submission failed)
            factor_found: Factor found in stage 1 (None if no factor)
            client_id: Client identifier for upload

        Returns:
            Upload result dict if uploaded, None if skipped or failed
        """
        if not factor_found and stage1_attempt_id:
            # No factor found and we have attempt ID - upload residue for potential stage 2
            print(f"Uploading residue file ({residue_file.stat().st_size} bytes)...")
            upload_result = self.api_client.upload_residue(
                client_id=client_id,
                residue_file_path=str(residue_file),
                stage1_attempt_id=stage1_attempt_id,
                expiry_days=7
            )

            if upload_result:
                print(f"Residue uploaded: ID {upload_result['residue_id']}, "
                      f"{upload_result['curve_count']} curves")
                return upload_result
            else:
                self.logger.error("Failed to upload residue file")

                # Preserve failed upload if configured
                if self.config['execution'].get('preserve_failed_uploads', False):
                    self._preserve_failed_upload(residue_file)

                return None
        elif factor_found:
            # Factor found - no need for stage 2
            print("Factor found - skipping residue upload")
            return None
        else:
            # No stage1_attempt_id - can't upload
            return None

    def _run_stage1(self, composite: str, b1: int, curves: int,
                   residue_file: Path, use_gpu: bool, verbose: bool,
                   gpu_device: Optional[int] = None, gpu_curves: Optional[int] = None,
                   sigma: Optional[Union[str, int]] = None, param: Optional[int] = None) -> tuple[bool, Optional[str], int, str, List[tuple[str, Optional[str]]]]:
        """
        Run Stage 1 with GPU or CPU and save residues.

        This is a convenience wrapper around _run_stage1_primitive() that provides
        a tuple-based return format for easier unpacking in main block code.

        Returns:
            tuple: (success, factor, actual_curves, raw_output, all_factors)
        """
        # Call the primitive
        result = self._run_stage1_primitive(
            composite=composite,
            b1=b1,
            curves=curves,
            residue_file=residue_file,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            gpu_curves=gpu_curves,
            sigma=sigma,
            param=param,
            verbose=verbose
        )

        # Convert dict result to tuple format for backward compatibility
        success = result['success']
        factor = result['factors'][-1] if result['factors'] else None  # Use last factor
        actual_curves = result['curves_completed']
        output = result['raw_output']
        all_factors = list(zip(result['factors'], result['sigmas']))

        return success, factor, actual_curves, output, all_factors

    def _run_stage2_multithread(self, residue_file: Path, b1: int, b2: Optional[int],
                               workers: int, verbose: bool, early_termination: bool = True,
                               progress_interval: int = 0) -> Tuple[Optional[str], List[str], int, float, Optional[str]]:
        """
        Run Stage 2 with multiple CPU workers.

        This is now a thin wrapper around Stage2Executor.

        Returns:
            Tuple of (factor, all_factors, curves_completed, execution_time, sigma)
        """
        executor = Stage2Executor(self, residue_file, b1, b2, workers, verbose)
        return executor.execute(early_termination, progress_interval)

    def _split_residue_file(self, residue_file: Path, num_chunks: int) -> List[Path]:
        """Split residue file into chunks for parallel processing"""
        # Create unique temporary directory for chunk files to avoid conflicts between concurrent jobs
        import tempfile
        chunk_dir = tempfile.mkdtemp(prefix="ecm_chunks_")
        self.logger.debug(f"Creating chunks in temporary directory: {chunk_dir}")

        # Use ResidueFileManager to split the file
        chunk_paths = self.residue_manager.split_into_chunks(
            str(residue_file), num_chunks, chunk_dir
        )

        # Convert string paths to Path objects
        return [Path(p) for p in chunk_paths]



    def _parse_residue_file(self, residue_path: Path) -> Dict[str, Any]:
        """
        Parse ECM residue file and extract all metadata in a single pass.

        Returns:
            Dict with keys: composite, b1, curve_count
        """
        metadata = self.residue_manager.parse_metadata(str(residue_path))

        if metadata:
            composite, b1, curve_count = metadata
            return {
                'composite': composite,
                'b1': b1,
                'curve_count': curve_count
            }
        else:
            # Return default values on parse failure
            return {
                'composite': 'unknown',
                'b1': 0,
                'curve_count': 0
            }

    def _correlate_factor_to_sigma(self, factor: str, residue_path: Path) -> Optional[str]:
        """
        Try to determine which sigma value found the factor by testing each residue.
        This is a fallback when ECM output doesn't contain sigma information.
        """
        sigma = self.residue_manager.correlate_factor_to_sigma(factor, str(residue_path))

        if sigma:
            # Format as "3:sigma" (ECM format with parametrization 3)
            # If sigma already contains parametrization prefix, use as-is
            if ':' not in sigma:
                return f"3:{sigma}"
            return sigma

        return None




    def get_program_version(self, program: str) -> str:
        """Override base class method to get GMP-ECM version"""
        return self.get_ecm_version()

    def get_ecm_version(self) -> str:
        """Get GMP-ECM version"""
        try:
            result = subprocess.run(
                [self.config['programs']['gmp_ecm']['path'], '-h'],
                capture_output=True,
                text=True
            )
            from lib.parsing_utils import extract_program_version
            return extract_program_version(result.stdout, 'ecm')
        except:
            pass
        return "unknown"

    def _fully_factor_found_result(self, factor: str, max_ecm_attempts: int = 5, quiet: bool = False) -> List[str]:
        """
        Recursively factor a result from ECM until all prime factors found.
        Handles composite factors by using trial division + ECM with increasing B1.

        Args:
            factor: Factor found by ECM (may be composite)
            max_ecm_attempts: Maximum ECM attempts with increasing B1 (default: 5)

        Returns:
            List of prime factors (as strings)
        """
        factor_int = int(factor)

        # Trial division catches small factors quickly (2, 3, 5, 7, ... up to 10^7)
        small_primes, cofactor = trial_division(factor_int, limit=10**7)
        all_primes = [str(p) for p in small_primes]

        if cofactor == 1:
            return all_primes

        # Check if cofactor is prime using probabilistic test
        if is_probably_prime(cofactor):
            self.logger.info(f"Cofactor {cofactor} is prime")
            all_primes.append(str(cofactor))
            return all_primes

        # Cofactor is composite - use ECM with increasing B1
        digit_length = len(str(cofactor))
        self.logger.info(f"Cofactor remaining: C{digit_length}, using ECM to complete factorization")

        current_cofactor = cofactor
        for attempt in range(max_ecm_attempts):
            if current_cofactor == 1:
                break

            # Select B1 based on cofactor size
            cofactor_digits = len(str(current_cofactor))
            b1 = get_b1_for_digit_length(cofactor_digits)

            # Use more curves for smaller numbers (they're faster)
            curves = max(10, 50 - (cofactor_digits // 2))

            self.logger.info(f"ECM attempt {attempt+1}/{max_ecm_attempts} on C{cofactor_digits} with B1={b1}, {curves} curves")

            try:
                # Use v2 API for ECM execution
                from lib.ecm_config import ECMConfig
                config = ECMConfig(
                    composite=str(current_cofactor),
                    b1=b1,
                    curves=curves,
                    verbose=False
                )
                ecm_result = self.run_ecm_v2(config)

                # Extract factors from FactorResult
                found_factors = ecm_result.factors if ecm_result.factors else []

                if found_factors:
                    self.logger.info(f"ECM found {len(found_factors)} factor(s): {found_factors}")

                    # Recursively factor each found factor
                    for found_factor in found_factors:
                        sub_primes = self._fully_factor_found_result(found_factor, max_ecm_attempts, quiet=quiet)
                        all_primes.extend(sub_primes)

                        # Divide out from cofactor
                        for prime in sub_primes:
                            current_cofactor //= int(prime)

                    # Check if fully factored
                    if current_cofactor == 1:
                        break

                    # Check if remaining cofactor is prime
                    if is_probably_prime(current_cofactor):
                        self.logger.info(f"Remaining cofactor {current_cofactor} is prime")
                        all_primes.append(str(current_cofactor))
                        current_cofactor = 1
                        break
                else:
                    self.logger.info(f"No factor found in attempt {attempt+1}")

            except Exception as e:
                self.logger.error(f"ECM factorization error: {e}")
                break

        # If we still have a composite cofactor after all attempts, return it as-is
        if current_cofactor > 1:
            self.logger.warning(f"Could not fully factor C{len(str(current_cofactor))}: {current_cofactor}")
            all_primes.append(str(current_cofactor))

        return all_primes


if __name__ == '__main__':
    from lib.arg_parser import (
        create_ecm_parser, validate_ecm_args, print_validation_errors,
        resolve_gpu_settings, get_method_defaults,
        resolve_worker_count, get_stage2_workers_default
    )

    parser = create_ecm_parser()
    args = parser.parse_args()
    wrapper = ECMWrapper('client.yaml')

    errors = validate_ecm_args(args, wrapper.config)
    print_validation_errors(errors)

    # Set up graceful shutdown handler for Ctrl+C
    def signal_handler(signum, frame):  # noqa: ARG001
        if not wrapper.interrupted:
            wrapper.interrupted = True
            print("\n^C Interrupt received. Stopping workers and preparing to submit partial results...")
            wrapper.stop_event.set()
        else:
            # Second Ctrl+C forces immediate exit
            print("\nForced exit (second interrupt)")
            sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)

    # Auto-work mode: continuously request and process work assignments
    if hasattr(args, 'auto_work') and args.auto_work:
        work_count_limit = args.work_count if hasattr(args, 'work_count') and args.work_count else None

        # Check for decoupled two-stage modes
        is_stage1_only = hasattr(args, 'stage1_only') and args.stage1_only
        is_stage2_work = hasattr(args, 'stage2_work') and args.stage2_work

        print("=" * 60)
        if is_stage1_only:
            mode_name = "Stage 1 Producer (GPU)"
        elif is_stage2_work:
            mode_name = "Stage 2 Consumer (CPU)"
        else:
            mode_name = "Auto-work"

        if work_count_limit:
            print(f"{mode_name} mode enabled - will process {work_count_limit} assignment(s)")
        else:
            print(f"{mode_name} mode enabled - requesting work from server")
            print("Press Ctrl+C to stop")
        print("=" * 60)
        print()

        # Initialize API clients for auto-work mode
        wrapper._ensure_api_clients()

        # Get client ID from config
        client_id = wrapper.config['client']['username']
        current_work_id = None
        current_residue_id = None  # For stage2-work mode
        completed_count = 0
        consecutive_failures = 0  # Track consecutive failures to prevent infinite loops
        MAX_CONSECUTIVE_FAILURES = 3

        # Stage 1 Only Mode
        if is_stage1_only:
            try:
                while not wrapper.interrupted:
                    # Request regular ECM work
                    work = request_ecm_work(wrapper.api_client, client_id, args, wrapper.logger)

                    if not work:
                        continue

                    current_work_id = work['work_id']
                    composite = work['composite']
                    digit_length = work['digit_length']

                    # Resolve GPU settings
                    use_gpu, gpu_device, gpu_curves = resolve_gpu_settings(args, wrapper.config)

                    # Determine curves (use config default if not specified)
                    curves = args.curves if args.curves is not None else wrapper.config['programs']['gmp_ecm']['default_curves']

                    print_work_header(
                        work_id=current_work_id,
                        composite=composite,
                        digit_length=digit_length,
                        params={'B1': args.b1, 'curves': curves}
                    )

                    try:
                        # Generate residue file path
                        residue_dir = Path(wrapper.config['execution'].get('residue_dir', 'data/residues'))
                        residue_dir.mkdir(parents=True, exist_ok=True)
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        residue_file = residue_dir / f"stage1_{timestamp}_{composite[:20]}.txt"

                        # Run stage 1 only (B2=0)
                        sigma = parse_sigma_arg(args)
                        param = resolve_param(args, use_gpu)

                        print(f"Running ECM stage 1 (B1={args.b1}, curves={curves})...")
                        print(f"Saving residues to: {residue_file}")

                        success, factor, actual_curves, raw_output, all_factors = wrapper._run_stage1(
                            composite=composite,
                            b1=args.b1,
                            curves=curves,
                            residue_file=residue_file,
                            sigma=sigma,
                            param=param,
                            use_gpu=use_gpu,
                            gpu_device=gpu_device,
                            gpu_curves=gpu_curves,
                            verbose=args.verbose
                        )

                        # Check if we're interrupted (don't treat user cancellation as failure)
                        if wrapper.interrupted:
                            wrapper.logger.info("Stage 1 interrupted by user, cleaning up...")
                            # Try to abandon work (might fail if already expired)
                            try:
                                wrapper.abandon_work(current_work_id, reason="user_cancelled")
                            except Exception:
                                pass  # Ignore errors when abandoning (work might have expired)
                            current_work_id = None
                            break  # Exit the work loop

                        if not success:
                            wrapper.logger.error("Stage 1 execution failed")
                            wrapper.abandon_work(current_work_id, reason="stage1_failed")
                            current_work_id = None
                            continue

                        # Build results using ResultsBuilder
                        builder = (results_for_stage1(composite, args.b1, actual_curves, param if param is not None else 3)
                            .with_curves(actual_curves, actual_curves)
                            .with_factors(all_factors)
                            .add_raw_output(raw_output)
                            .with_execution_time(0))  # Will be filled by subprocess
                        if current_work_id:
                            builder.with_work_id(current_work_id)  # For failed submission recovery
                        results = builder.build()

                        # Submit stage 1 results and handle workflow
                        stage1_attempt_id = submit_stage1_complete_workflow(
                            wrapper=wrapper,
                            results=results,
                            residue_file=residue_file,
                            work_id=current_work_id,
                            project=args.project,
                            client_id=client_id,
                            factor_found=factor,
                            cleanup_residue=True
                        )

                        # Check if submission failed
                        if not stage1_attempt_id:
                            current_work_id = None
                            continue

                        # Mark work as complete
                        wrapper.api_client.complete_work(current_work_id, client_id)
                        current_work_id = None
                        completed_count += 1
                        consecutive_failures = 0  # Reset on success

                        if print_work_status("Stage 1", completed_count, work_count_limit):
                            break

                    except Exception as e:
                        consecutive_failures += 1

                        # Handle work failure with circuit breaker
                        if handle_work_failure(
                            wrapper=wrapper,
                            current_work_id=current_work_id,
                            consecutive_failures=consecutive_failures,
                            max_failures=MAX_CONSECUTIVE_FAILURES,
                            error_msg=f"Error in stage 1 processing: {e}"
                        ):
                            break

                        current_work_id = None

                        # Check if work limit reached
                        if check_work_limit_reached(completed_count, work_count_limit):
                            break

            except KeyboardInterrupt:
                handle_shutdown(
                    wrapper=wrapper,
                    current_work_id=current_work_id,
                    current_residue_id=None,
                    mode_name="Stage 1 Producer mode",
                    completed_count=completed_count
                )

            # Exit after stage1-only mode completes to avoid falling through to standard mode
            sys.exit(0)

        # Stage 2 Work Mode
        elif is_stage2_work:
            # Initialize variables for KeyboardInterrupt handler
            local_residue_file = None

            try:
                while not wrapper.interrupted:
                    # Request residue work from server
                    residue_work = wrapper.api_client.get_residue_work(
                        client_id=client_id,
                        min_digits=args.min_digits if hasattr(args, 'min_digits') else None,
                        max_digits=args.max_digits if hasattr(args, 'max_digits') else None,
                        min_priority=args.priority if hasattr(args, 'priority') else None,
                        claim_timeout_hours=24
                    )

                    if not residue_work:
                        wrapper.logger.info("No residue work available, waiting 30 seconds before retry...")
                        time.sleep(30)
                        continue

                    current_residue_id = residue_work['residue_id']
                    composite = residue_work['composite']
                    digit_length = residue_work['digit_length']
                    b1 = residue_work['b1']
                    curve_count = residue_work['curve_count']
                    stage1_attempt_id = residue_work.get('stage1_attempt_id')
                    suggested_b2 = residue_work.get('suggested_b2', b1 * 100)

                    # Determine B2 (priority: explicit --b2 > --b2-multiplier > server suggestion)
                    if args.b2 is not None:
                        # Explicit B2 specified (0 means GMP-ECM default)
                        b2 = args.b2
                    elif hasattr(args, 'b2_multiplier') and args.b2_multiplier is not None:
                        # Dynamic calculation based on B1
                        b2 = int(b1 * args.b2_multiplier)
                        print(f"Using dynamic B2 = B1 * {args.b2_multiplier} = {b2}")
                    else:
                        # Use server suggestion (default)
                        b2 = suggested_b2

                    b2_display = "GMP-ECM default" if b2 == -1 else str(b2)
                    print_work_header(
                        work_id=current_residue_id,
                        composite=composite,
                        digit_length=digit_length,
                        params={
                            'B1': b1,
                            'B2': b2_display,
                            'curves': curve_count,
                            'Stage 1 attempt ID': stage1_attempt_id
                        }
                    )

                    try:
                        # Download residue file
                        residue_dir = Path(wrapper.config['execution'].get('residue_dir', 'data/residues'))
                        residue_dir.mkdir(parents=True, exist_ok=True)
                        local_residue_file = residue_dir / f"s2_residue_{current_residue_id}.txt"

                        print(f"Downloading residue file...")
                        download_success = wrapper.api_client.download_residue(
                            client_id=client_id,
                            residue_id=current_residue_id,
                            output_path=str(local_residue_file)
                        )

                        if not download_success:
                            wrapper.logger.error("Failed to download residue file")
                            wrapper.api_client.abandon_residue(client_id, current_residue_id)
                            current_residue_id = None
                            continue

                        print(f"Downloaded {local_residue_file.stat().st_size} bytes")

                        # Get stage2 workers
                        stage2_workers = args.stage2_workers if hasattr(args, 'stage2_workers') else get_stage2_workers_default(wrapper.config)

                        # Run stage 2 on residue file
                        print(f"Running stage 2 with {stage2_workers} workers...")
                        stage2_executor = Stage2Executor(
                            wrapper, local_residue_file, b1, b2, stage2_workers, args.verbose
                        )
                        stage2_factor, stage2_all_factors, stage2_curves, stage2_time, stage2_sigma = stage2_executor.execute(
                            early_termination=not (hasattr(args, 'continue_after_factor') and args.continue_after_factor),
                            progress_interval=args.progress_interval if hasattr(args, 'progress_interval') else 0
                        )

                        # Build results
                        results = {
                            'composite': composite,
                            'b1': b1,
                            'b2': None if b2 == -1 else b2,  # -1 means GMP-ECM default, submit as None
                            'curves_requested': curve_count,
                            'curves_completed': stage2_curves,
                            'factors_found': stage2_all_factors if stage2_all_factors else [],
                            'factor_found': stage2_factor,
                            'sigma': stage2_sigma,
                            'raw_output': f"Stage 2 from residue {current_residue_id}",
                            'method': 'ecm',
                            'parametrization': residue_work.get('parametrization', 3),
                            'execution_time': stage2_time,
                        }

                        # Submit stage 2 results
                        print("Submitting stage 2 results...")
                        program_name = 'gmp-ecm-ecm'
                        submit_response = wrapper.submit_result(results, args.project, program_name)

                        if not submit_response:
                            wrapper.logger.error("Failed to submit stage 2 results")
                            wrapper.api_client.abandon_residue(client_id, current_residue_id)
                            current_residue_id = None
                            if local_residue_file.exists():
                                local_residue_file.unlink()
                            continue

                        # Extract attempt_id from response
                        stage2_attempt_id = submit_response.get('attempt_id')
                        if stage2_attempt_id:
                            print(f"Stage 2 attempt ID: {stage2_attempt_id}")
                        else:
                            wrapper.logger.error("No attempt_id returned from submit")
                            wrapper.api_client.abandon_residue(client_id, current_residue_id)
                            current_residue_id = None
                            if local_residue_file.exists():
                                local_residue_file.unlink()
                            continue

                        # Complete residue (supersedes stage 1, deletes server file)
                        print("Completing residue work...")
                        complete_result = wrapper.api_client.complete_residue(
                            client_id=client_id,
                            residue_id=current_residue_id,
                            stage2_attempt_id=stage2_attempt_id
                        )

                        if complete_result:
                            new_t_level = complete_result.get('new_t_level')
                            if new_t_level is not None:
                                print(f"T-level updated to {new_t_level:.2f}")
                        else:
                            wrapper.logger.warning("Failed to complete residue on server")

                        # Clean up local residue file
                        if local_residue_file.exists():
                            local_residue_file.unlink()
                            wrapper.logger.info(f"Deleted local residue file: {local_residue_file}")

                        current_residue_id = None
                        completed_count += 1
                        consecutive_failures = 0  # Reset on success

                        if print_work_status("Stage 2", completed_count, work_count_limit):
                            break

                    except Exception as e:
                        consecutive_failures += 1

                        # Abandon residue work
                        if current_residue_id:
                            wrapper.api_client.abandon_residue(client_id, current_residue_id)
                            current_residue_id = None

                        # Clean up local residue file
                        if local_residue_file and local_residue_file.exists():
                            local_residue_file.unlink()

                        # Check circuit breaker
                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                            wrapper.logger.error(
                                f"Too many consecutive failures ({consecutive_failures}), exiting..."
                            )
                            break

                        # Check if work limit reached
                        if check_work_limit_reached(completed_count, work_count_limit):
                            break

            except KeyboardInterrupt:
                handle_shutdown(
                    wrapper=wrapper,
                    current_work_id=None,
                    current_residue_id=current_residue_id,
                    mode_name="Stage 2 Consumer mode",
                    completed_count=completed_count,
                    local_residue_file=local_residue_file
                )

            # Exit after stage2-work mode completes to avoid falling through to standard mode
            sys.exit(0)

        # Standard auto-work mode
        try:
            while not wrapper.interrupted:
                # Request work from server
                work = request_ecm_work(wrapper.api_client, client_id, args, wrapper.logger)

                if not work:
                    continue

                # Store current work ID for cleanup on interrupt
                current_work_id = work['work_id']
                composite = work['composite']
                digit_length = work['digit_length']

                print_work_header(
                    work_id=current_work_id,
                    composite=composite,
                    digit_length=digit_length,
                    params={
                        'T-level': f"{work.get('current_t_level', 0):.1f} → {work.get('target_t_level', 0):.1f}"
                    }
                )

                # Execute ECM - determine mode from parameters
                try:
                    has_b1_b2 = args.b1 is not None and args.b2 is not None
                    has_client_tlevel = hasattr(args, 'tlevel') and args.tlevel is not None

                    # Determine execution mode
                    if has_client_tlevel or (not has_b1_b2 and not has_client_tlevel):
                        # T-level mode (client-specified or server default)
                        target_tlevel = args.tlevel if has_client_tlevel else work.get('target_t_level', 35.0)

                        # Start from user-specified level, server's current level, or 0
                        if hasattr(args, 'start_tlevel') and args.start_tlevel is not None:
                            start_tlevel = args.start_tlevel
                        else:
                            start_tlevel = work.get('current_t_level', 0.0)

                        mode_desc = "client t-level" if has_client_tlevel else "server t-level"
                        print(f"Mode: {mode_desc} (start: {start_tlevel:.1f}, target: {target_tlevel:.1f})")

                        # Resolve worker count for multiprocess
                        workers = resolve_worker_count(args) if args.multiprocess else 1

                        # T-level mode (v2 API) - submits after each step internally
                        config = TLevelConfig(
                            composite=composite,
                            target_t_level=target_tlevel,
                            threads=workers,
                            verbose=args.verbose,
                            project=args.project,
                            no_submit=False,
                            work_id=current_work_id
                        )
                        result = wrapper.run_tlevel_v2(config)

                        # Note: Batches were submitted individually during execution
                        # Just need to track results for factor detection
                        results = result.to_dict(composite, args.method)

                    else:
                        # B1/B2 mode with optional two-stage or multiprocess
                        b1 = args.b1
                        b2 = args.b2
                        curves = args.curves if args.curves else (1 if args.two_stage else wrapper.config['programs']['gmp_ecm']['default_curves'])

                        # Common parameters
                        use_gpu, gpu_device, gpu_curves = resolve_gpu_settings(args, wrapper.config)
                        sigma = parse_sigma_arg(args)
                        param = resolve_param(args, use_gpu)
                        continue_after_factor = args.continue_after_factor if hasattr(args, 'continue_after_factor') else False

                        if args.two_stage and args.method == 'ecm':
                            # Two-stage mode (v2 API)
                            print(f"Mode: two-stage GPU+CPU (B1={b1}, B2={b2}, curves={curves})")
                            stage2_workers = resolve_stage2_workers(args, wrapper.config)

                            config = TwoStageConfig(
                                composite=composite,
                                b1=b1,
                                b2=b2,
                                stage1_curves=curves,
                                stage1_device="GPU" if use_gpu else "CPU",
                                stage2_device="CPU",
                                stage1_parametrization=param if param else 3,
                                threads=stage2_workers,
                                verbose=args.verbose
                            )
                            result = wrapper.run_two_stage_v2(config)

                            # Convert FactorResult to dict
                            results = result.to_dict(composite, args.method)

                        elif args.multiprocess:
                            # Multiprocess mode (v2 API)
                            workers = resolve_worker_count(args)
                            print(f"Mode: multiprocess (B1={b1}, B2={b2}, curves={curves}, workers={workers})")

                            config = MultiprocessConfig(
                                composite=composite,
                                b1=b1,
                                b2=b2,
                                total_curves=curves,
                                num_processes=workers,
                                parametrization=param if param else 3,
                                method=args.method,
                                verbose=args.verbose
                            )
                            result = wrapper.run_multiprocess_v2(config)

                            # Convert FactorResult to dict
                            results = result.to_dict(composite, args.method)

                        else:
                            # Standard mode (v2 API)
                            print(f"Mode: standard (B1={b1}, B2={b2}, curves={curves})")

                            config = ECMConfig(
                                composite=composite,
                                b1=b1,
                                b2=b2,
                                curves=curves,
                                sigma=sigma,
                                parametrization=param if param else 3,
                                method=args.method,
                                verbose=args.verbose
                            )
                            result = wrapper.run_ecm_v2(config)

                            # Convert FactorResult to dict
                            results = result.to_dict(composite, args.method)

                        # Submit results for B1/B2 modes
                        if results.get('curves_completed', 0) > 0:
                            # Include work_id for failed submission recovery
                            results['work_id'] = current_work_id
                            program_name = f'gmp-ecm-{results.get("method", "ecm")}'
                            submit_response = wrapper.submit_result(results, args.project, program_name)

                            if not submit_response:
                                wrapper.logger.error("Failed to submit results, abandoning work assignment")
                                wrapper.abandon_work(current_work_id, reason="submission_failed")
                                current_work_id = None
                                continue

                    # Mark work as complete
                    wrapper.api_client.complete_work(current_work_id, client_id)
                    current_work_id = None
                    completed_count += 1

                    # Check if we've reached the work count limit
                    if print_work_status("Work assignment completed successfully", completed_count, work_count_limit):
                        break

                except Exception as e:
                    wrapper.logger.exception(f"Error processing work assignment: {e}")
                    if current_work_id:
                        wrapper.abandon_work(current_work_id, reason="execution_error")
                        current_work_id = None

        except KeyboardInterrupt:
            handle_shutdown(
                wrapper=wrapper,
                current_work_id=current_work_id,
                current_residue_id=None,
                mode_name="Auto-work mode",
                completed_count=completed_count
            )

        # Exit after auto-work mode completes to avoid falling through to standard mode
        sys.exit(0)

    # Use shared argument processing utilities
    b1_default, b2_default = get_method_defaults(wrapper.config, args.method)
    b1 = args.b1 or b1_default
    b2 = args.b2 if args.b2 is not None else b2_default
    curves = args.curves if args.curves is not None else wrapper.config['programs']['gmp_ecm']['default_curves']

    # Resolve GPU settings using shared utility
    use_gpu, gpu_device, gpu_curves = resolve_gpu_settings(args, wrapper.config)

    # Resolve worker count
    args.workers = resolve_worker_count(args)

    # Resolve stage2 workers from config if not explicitly set
    stage2_workers = resolve_stage2_workers(args, wrapper.config)

    # Manual stage1-only mode (without auto-work)
    if hasattr(args, 'stage1_only') and args.stage1_only and not (hasattr(args, 'auto_work') and args.auto_work):
        # Use config defaults if B1/curves not specified
        if args.b1 is None:
            args.b1 = wrapper.config['programs']['gmp_ecm']['default_b1']
            print(f"Using default B1 from config: {args.b1}")
        if args.curves is None:
            args.curves = wrapper.config['programs']['gmp_ecm']['default_curves']
            print(f"Using default curves from config: {args.curves}")

        # Generate residue file path
        residue_dir = Path(wrapper.config['execution'].get('residue_dir', 'data/residues'))
        residue_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        residue_file = residue_dir / f"stage1_manual_{timestamp}.txt"

        # Parse sigma if provided
        sigma = parse_sigma_arg(args)

        # Get param if provided, default to 3 for GPU
        param = resolve_param(args, use_gpu)

        print()
        print("=" * 60)
        print("Manual Stage 1 Only Mode")
        print(f"Composite: {args.composite[:50]}... ({len(args.composite)} digits)")
        print(f"Parameters: B1={args.b1}, curves={args.curves}")
        print(f"Residue file: {residue_file}")
        print("=" * 60)
        print()

        # Run stage 1
        success, factor, actual_curves, raw_output, all_factors = wrapper._run_stage1(
            composite=args.composite,
            b1=args.b1,
            curves=args.curves,
            residue_file=residue_file,
            sigma=sigma,
            param=param,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            gpu_curves=gpu_curves,
            verbose=args.verbose
        )

        if not success:
            wrapper.logger.error("Stage 1 execution failed")
            sys.exit(1)

        # Build results using ResultsBuilder
        results = (results_for_stage1(args.composite, args.b1, actual_curves, param if param is not None else 3)
            .with_curves(actual_curves, actual_curves)
            .with_factors(all_factors)
            .add_raw_output(raw_output)
            .with_execution_time(0)
            .build())

        # Submit stage 1 results
        if not args.no_submit:
            client_id = wrapper.config['client']['username'] + '-' + wrapper.config['client']['cpu_name']
            stage1_attempt_id = submit_stage1_complete_workflow(
                wrapper=wrapper,
                results=results,
                residue_file=residue_file,
                work_id=None,  # Not auto-work mode
                project=args.project,
                client_id=client_id,
                factor_found=factor,
                cleanup_residue=True
            )

            if not stage1_attempt_id:
                sys.exit(1)
        else:
            # Clean up local residue file if not submitting
            if residue_file.exists():
                residue_file.unlink()

        print()
        print("=" * 60)
        print("Stage 1 complete")
        print("=" * 60)
        sys.exit(0)

    # Check for T-level mode first (highest priority)
    elif hasattr(args, 'tlevel') and args.tlevel:
        # T-level mode: run ECM iteratively until target t-level reached (v2 API)
        # Submits each step individually if not --no-submit
        config = TLevelConfig(
            composite=args.composite,
            target_t_level=args.tlevel,
            threads=args.workers if args.multiprocess else 1,
            verbose=args.verbose,
            project=args.project if hasattr(args, 'project') else None,
            no_submit=args.no_submit if hasattr(args, 'no_submit') else False
        )
        result = wrapper.run_tlevel_v2(config)

        # Convert FactorResult to dict for backward compatibility with result handling below
        # Note: Individual batches already submitted during execution if enabled
        results = result.to_dict(args.composite, args.method)
    # Run ECM - choose mode based on arguments (validation already done by validate_ecm_args)
    elif args.resume_residues:
        # Resume from existing residues - run stage 2 only (v2 API)
        config = TwoStageConfig(
            composite="",  # Will be extracted from residue file
            b1=b1,
            b2=b2,
            resume_file=args.resume_residues,
            threads=stage2_workers,
            continue_after_factor=args.continue_after_factor,
            verbose=args.verbose
        )
        result = wrapper.run_two_stage_v2(config)

        # Convert FactorResult to dict for backward compatibility
        results = result.to_dict(args.composite, args.method)
    elif args.stage2_only:
        # Stage 2 only mode (v2 API)
        config = TwoStageConfig(
            composite="",  # Will be extracted from residue file
            b1=b1,
            b2=b2,
            resume_file=args.stage2_only,
            threads=stage2_workers,
            continue_after_factor=args.continue_after_factor,
            verbose=args.verbose
        )
        result = wrapper.run_two_stage_v2(config)

        # Convert FactorResult to dict for backward compatibility
        results = result.to_dict(args.composite, args.method)
    elif args.multiprocess:
        # Multiprocess mode (v2 API)
        # Resolve parametrization (multiprocess is CPU-only, so use_gpu=False)
        param = resolve_param(args, use_gpu=False)

        config = MultiprocessConfig(
            composite=args.composite,
            b1=b1,
            b2=b2,
            total_curves=curves,
            num_processes=args.workers,
            parametrization=param,
            verbose=args.verbose
        )
        result = wrapper.run_multiprocess_v2(config)

        # Convert FactorResult to dict for backward compatibility
        results = result.to_dict(args.composite, args.method)
    elif args.two_stage and args.method == 'ecm':
        # Two-stage mode (v2 API)
        # Parse sigma if provided (convert "N" to integer, keep "3:N" as string)
        sigma = parse_sigma_arg(args)

        # Get param if provided (two-stage uses GPU by default)
        param = resolve_param(args, use_gpu)

        config = TwoStageConfig(
            composite=args.composite,
            b1=b1,
            b2=b2,
            stage1_curves=curves,
            stage1_device="GPU" if use_gpu else "CPU",
            stage2_device="CPU",
            stage1_parametrization=param,
            threads=stage2_workers,
            verbose=args.verbose,
            save_residues=args.save_residues
        )
        result = wrapper.run_two_stage_v2(config)

        # Convert FactorResult to dict for backward compatibility
        results = result.to_dict(args.composite, args.method)
    else:
        # Standard mode (v2 API)
        if args.two_stage:
            print("Warning: Two-stage mode only available for ECM method, falling back to standard mode")

        # Parse sigma if provided (convert "N" to integer, keep "3:N" as string)
        sigma = parse_sigma_arg(args)

        # Get param if provided
        param = resolve_param(args, use_gpu)

        config = ECMConfig(
            composite=args.composite,
            b1=b1,
            b2=b2,
            curves=curves,
            sigma=sigma,
            parametrization=param,
            method=args.method,
            verbose=args.verbose
        )
        result = wrapper.run_ecm_v2(config)

        # Convert FactorResult to dict for backward compatibility
        results = result.to_dict(args.composite, args.method)

    # Submit results unless disabled or failed
    # Skip submission for t-level mode (each step already submitted)
    if not args.no_submit and not (hasattr(args, 'tlevel') and args.tlevel):
        # Only submit if we actually completed some curves (not a failure)
        if results.get('curves_completed', 0) > 0:
            # Show detailed status if interrupted
            if wrapper.interrupted:
                curves_completed = results.get('curves_completed', 0)
                curves_requested = results.get('curves_requested', curves)
                exec_time = results.get('execution_time', 0)
                print(f"\nCompleted {curves_completed}/{curves_requested} curves in {exec_time:.1f}s before interruption")
                print("Submitting partial results...")

            program_name = f'gmp-ecm-{results.get("method", "ecm")}'
            submit_response = wrapper.submit_result(results, args.project, program_name)

            if wrapper.interrupted:
                if submit_response:
                    print("Partial results submitted successfully")
                else:
                    print("Warning: Partial result submission failed (saved to data/results/ for retry)")

            sys.exit(0 if submit_response else 1)
        else:
            wrapper.logger.warning("Skipping result submission due to failure (0 curves completed)")
            sys.exit(1)
    elif args.no_submit and wrapper.interrupted:
        # If interrupted and --no-submit, show what would have been submitted
        curves_completed = results.get('curves_completed', 0)
        curves_requested = results.get('curves_requested', curves)
        exec_time = results.get('execution_time', 0)
        print(f"\nCompleted {curves_completed}/{curves_requested} curves in {exec_time:.1f}s before interruption")
        print("Skipping submission (--no-submit flag)")

