#!/usr/bin/env python3
import subprocess
import time
import sys
import signal
import threading
import hashlib
import queue
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
        self.graceful_shutdown_requested = False  # First Ctrl+C: finish current work
        self._original_sigint_handler: Optional[Any] = None  # Store original handler for restoration

    # ==================== NEW CONFIG-BASED METHODS ====================
    # These methods use configuration objects for cleaner interfaces

    def run_ecm_v2(self, config: ECMConfig) -> FactorResult:
        """
        Execute ECM with configuration object (simplified interface).

        This is the new, recommended method that uses ECMConfig dataclass.
        It has better type safety, validation, and testability.

        For GPU mode or when saving residues, uses _execute_ecm_primitive directly.
        For CPU mode without residues, delegates to run_multiprocess_v2 with 1 worker.

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
        # Use primitive directly for GPU mode or when saving residues
        # (multiprocess doesn't support these features)
        if config.use_gpu or config.save_residues:
            start_time = time.time()
            residue_path = Path(config.save_residues) if config.save_residues else None

            prim_result = self._execute_ecm_primitive(
                composite=config.composite,
                b1=config.b1,
                b2=config.b2,
                curves=config.curves,
                residue_save=residue_path,
                sigma=config.sigma,
                param=config.parametrization,
                method=config.method,
                use_gpu=config.use_gpu,
                gpu_device=config.gpu_device,
                gpu_curves=config.gpu_curves,
                verbose=config.verbose
            )

            return FactorResult.from_primitive_result(prim_result, time.time() - start_time)

        # For CPU mode without residues, delegate to multiprocess for proper handling
        mp_config = MultiprocessConfig(
            composite=config.composite,
            b1=config.b1,
            b2=config.b2,
            total_curves=config.curves,
            curves_per_process=config.curves,
            num_processes=1,
            parametrization=config.parametrization,
            method=config.method,
            verbose=config.verbose,
            progress_interval=config.progress_interval
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

        # Install graceful shutdown signal handler
        def graceful_sigint_handler(signum, frame):
            """Handle Ctrl+C gracefully: first time finish current work, second time abort."""
            if not self.graceful_shutdown_requested:
                self.graceful_shutdown_requested = True
                print("\n[Ctrl+C] Graceful shutdown initiated - completing current curves...")
                print("[Ctrl+C] Press Ctrl+C again to abort immediately")
                self.logger.info("Graceful shutdown requested - will complete current work")
            else:
                print("\n[Ctrl+C] Immediate abort requested")
                self.logger.info("Immediate abort requested")
                # Restore original handler and re-raise to trigger immediate abort
                if self._original_sigint_handler:
                    signal.signal(signal.SIGINT, self._original_sigint_handler)
                raise KeyboardInterrupt()

        # Save and install new handler
        self._original_sigint_handler = signal.signal(signal.SIGINT, graceful_sigint_handler)

        try:
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
                    verbose=config.verbose
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
            actual_b2 = config.b2 or (actual_b1 * 100)  # Default B2
            self.logger.info(f"Starting stage 2: B1={actual_b1:,}, B2={actual_b2:,}, workers={config.threads}")

            # _run_stage2_multithread returns: (factor, all_factors, curves_completed, execution_time, sigma)
            _factor, all_factors, curves_completed, _stage2_time, sigma = self._run_stage2_multithread(
                residue_file=residue_file,
                b1=actual_b1,
                b2=actual_b2,
                workers=config.threads,
                verbose=config.verbose,
                progress_interval=config.progress_interval
            )

            # Build FactorResult from stage 2 results
            result = FactorResult()
            if all_factors:
                for f in all_factors:
                    result.add_factor(f, sigma)
            result.curves_run = curves_completed
            result.execution_time = time.time() - start_time
            result.success = len(all_factors) > 0

            # Check if execution was gracefully shutdown
            if self.graceful_shutdown_requested:
                result.interrupted = True
                self.logger.info(f"Graceful shutdown completed - processed {curves_completed} curves")

            return result
        finally:
            # Always restore the original signal handler
            if self._original_sigint_handler:
                signal.signal(signal.SIGINT, self._original_sigint_handler)
            # Reset graceful shutdown flag for next execution
            self.graceful_shutdown_requested = False

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
                      result_queue, stop_event, config.progress_interval, progress_queue)
            )
            p.start()
            processes.append(p)

        # Collect results from workers
        all_factors = []
        all_sigmas = []
        all_raw_outputs = []
        total_curves_completed = 0
        completed_workers = 0

        def process_result(result: dict) -> None:
            """Process a single worker result."""
            nonlocal total_curves_completed
            total_curves_completed += result['curves_completed']

            if result['factor_found']:
                factor = result['factor_found']
                # Deduplicate: multiple workers can find the same factor
                if factor not in all_factors:
                    all_factors.append(factor)
                    all_sigmas.append(result.get('sigma_found'))
                    self.logger.info(f"Worker {result['worker_id']} found factor: {factor}")
                else:
                    self.logger.info(f"Worker {result['worker_id']} found same factor: {factor} (already recorded)")
                stop_event.set()  # Signal workers to stop on factor found

            # Collect raw output from each worker
            if 'raw_output' in result:
                all_raw_outputs.append(f"=== Worker {result['worker_id']} ===\n{result['raw_output']}")

        try:
            while completed_workers < len(processes):
                try:
                    result = result_queue.get(timeout=0.5)
                    process_result(result)
                    completed_workers += 1
                except queue.Empty:
                    # Timeout waiting for result - check if processes are still alive
                    if not any(p.is_alive() for p in processes):
                        # Drain any remaining results from the queue before breaking
                        # (processes may have exited after putting results in queue)
                        while True:
                            try:
                                result = result_queue.get_nowait()
                                process_result(result)
                                completed_workers += 1
                            except queue.Empty:
                                break
                        break
                    continue
        except KeyboardInterrupt:
            self.logger.info("Multiprocess ECM interrupted by user")
            self.interrupted = True
            try:
                stop_event.set()  # Signal workers to stop
            except Exception:
                pass  # Manager connection may be broken

        # Wait for all processes to finish
        for p in processes:
            try:
                p.join(timeout=2)
            except Exception:
                pass
            if p.is_alive():
                p.terminate()
                try:
                    p.join(timeout=1)
                except Exception:
                    pass

        # Shutdown the manager
        try:
            manager.shutdown()
        except Exception:
            pass

        if self.interrupted:
            print(f"\nMultiprocess ECM stopped. Completed {total_curves_completed} curves before interrupt.")

        # Build FactorResult with recursive factoring of any composite factors
        result = FactorResult()
        for factor, sigma in zip(all_factors, all_sigmas):
            # Fully factor each discovered factor to get all prime factors
            # This handles cases where ECM finds a composite factor (product of primes)
            # Use _fully_factor_composite which calls primitives directly (no recursion through run_ecm_v2)
            prime_factors = self._fully_factor_composite(factor)
            for prime in prime_factors:
                result.add_factor(prime, sigma)

        result.curves_run = total_curves_completed
        result.execution_time = time.time() - start_time
        result.success = len(result.factors) > 0
        result.raw_output = '\n\n'.join(all_raw_outputs) if all_raw_outputs else None
        result.interrupted = self.interrupted  # Signal if execution was interrupted

        return result

    def _run_tlevel_pipelined(self, config: TLevelConfig) -> FactorResult:
        """
        Execute T-level targeting in pipelined mode (GPU + CPU running concurrently).

        GPU thread produces residues, CPU thread consumes and processes them.
        Stops early when factor found to avoid wasted computation.

        Args:
            config: T-level configuration object with use_two_stage=True

        Returns:
            FactorResult with discovered factors
        """
        from lib.ecm_math import (get_optimal_b1_for_tlevel, calculate_tlevel,
                                  calculate_curves_to_target_direct, TLEVEL_TRANSITION_CACHE)

        self.logger.info(f"Starting PIPELINED progressive ECM with target t{config.target_t_level:.1f} (GPU+CPU concurrent)")
        start_time = time.time()

        # Shared state between threads
        residue_queue = queue.Queue(maxsize=2)  # Small queue for backpressure
        shutdown_event = threading.Event()

        # Results accumulation (protected by lock)
        result_lock = threading.Lock()
        all_factors = []
        all_sigmas = []
        total_curves = 0
        curve_history = []  # Completed curves (for actual t-level)

        # Current t-level tracking
        current_t_level = config.start_t_level

        # Projected t-level tracking (for GPU lookahead)
        # This tracks what the t-level WILL BE once queued work completes
        projected_lock = threading.Lock()
        projected_curve_history = []  # Includes queued but not yet completed curves
        projected_t_level = config.start_t_level

        # Build step targets
        step_targets = []
        t = 20.0
        while t < config.target_t_level:
            if t > current_t_level:
                step_targets.append(t)
            t += 5.0
        if config.target_t_level not in step_targets and config.target_t_level > current_t_level:
            step_targets.append(config.target_t_level)

        def gpu_producer():
            """GPU thread: Run stage 1 for each batch, put residues in queue"""
            nonlocal projected_t_level

            self.logger.info(f"[GPU Thread] Starting, will process {len(step_targets)} t-level steps")

            for step_target in step_targets:
                # Check for shutdown (factor found in CPU thread)
                if shutdown_event.is_set():
                    self.logger.info("[GPU Thread] Factor found, stopping production")
                    break

                # Check for user interrupt
                if self.interrupted:
                    self.logger.info("[GPU Thread] User interrupt, stopping")
                    shutdown_event.set()
                    break

                # Use PROJECTED t-level (includes queued but not completed work)
                # This prevents over-producing curves when GPU is ahead of CPU
                with projected_lock:
                    current_projected = projected_t_level

                # Skip if projected t-level already exceeds target
                if current_projected >= step_target:
                    self.logger.info(f"[GPU Thread] Skipping t{step_target:.1f} (projected t{current_projected:.2f} already exceeds)")
                    continue

                # Get optimal B1
                b1, _ = get_optimal_b1_for_tlevel(step_target)

                # Calculate B2 based on multiplier (for two-stage mode)
                b2_for_calculation = int(b1 * config.b2_multiplier)

                # Calculate curves needed based on PROJECTED t-level
                # Always use B2-aware calculation for two-stage mode (cache assumes default B2)
                curves = calculate_curves_to_target_direct(
                    current_projected, step_target, b1, 3, b2=b2_for_calculation
                )
                if curves:
                    self.logger.info(f"[GPU Thread] B2-aware: t{current_projected:.2f} → t{step_target:.1f} = {curves} curves at B1={b1}, B2={b2_for_calculation}, p=3")

                if curves is None or curves <= 0:
                    self.logger.warning(f"[GPU Thread] Could not calculate curves for t{current_projected:.3f} → t{step_target:.1f}, skipping")
                    continue

                # Chunk large batches if max_batch_curves is set
                # This allows the CPU to start processing sooner, potentially finding factors earlier
                max_batch = config.max_batch_curves
                remaining_curves = curves
                batch_num = 0

                while remaining_curves > 0:
                    # Check for shutdown between batches
                    if shutdown_event.is_set() or self.interrupted:
                        self.logger.info(f"[GPU Thread] Stopping mid-step (shutdown or interrupt)")
                        break

                    # Determine batch size
                    batch_curves = min(remaining_curves, max_batch) if max_batch else remaining_curves
                    batch_num += 1

                    if max_batch and curves > max_batch:
                        self.logger.info(f"[GPU Thread] Starting stage 1 batch {batch_num}: {batch_curves} curves at B1={b1} (projected t{current_projected:.2f} → t{step_target:.1f}, {remaining_curves} remaining)")
                    else:
                        self.logger.info(f"[GPU Thread] Starting stage 1: {batch_curves} curves at B1={b1} (projected t{current_projected:.2f} → t{step_target:.1f})")

                    # Create residue file
                    residue_file = Path(f"/tmp/tlevel_residue_{int(time.time() * 1000)}.txt")

                    # Run stage 1
                    stage1_start = time.time()
                    try:
                        stage1_success, stage1_factor, _, stage1_output, all_stage1_factors = self._run_stage1(
                            composite=config.composite,
                            b1=b1,
                            curves=batch_curves,
                            residue_file=residue_file,
                            use_gpu=True,
                            verbose=config.verbose,
                            gpu_device=None,
                            gpu_curves=None
                        )
                        stage1_time = time.time() - stage1_start

                        # Get ACTUAL curve count from residue file (GPU may run more than requested due to batch rounding)
                        # The ECM output parsing only catches one batch, residue file has the real count
                        if residue_file.exists():
                            residue_info = self._parse_residue_file(residue_file)
                            actual_curves = residue_info.get('curve_count', batch_curves)
                            if actual_curves != batch_curves:
                                self.logger.debug(f"[GPU Thread] Residue file has {actual_curves} curves (requested {batch_curves})")
                        else:
                            actual_curves = batch_curves  # Fallback if no residue (factor found?)

                        # Check for shutdown before putting to queue (prevents deadlock if CPU thread stopped consuming)
                        if shutdown_event.is_set():
                            self.logger.info(f"[GPU Thread] Shutdown detected after stage 1 completion, discarding batch")
                            # Clean up residue file since it won't be processed
                            if residue_file.exists():
                                residue_file.unlink()
                            break

                        # Put work item in queue (blocks if queue is full)
                        b2_for_batch = b1 * 100  # Two-stage uses B1 * 100
                        residue_queue.put({
                            'residue_file': residue_file,
                            'b1': b1,
                            'b2': b2_for_batch,
                            'curves': actual_curves,
                            'stage1_factor': stage1_factor,
                            'all_factors': all_stage1_factors,
                            'stage1_time': stage1_time,
                            'stage1_output': stage1_output,
                            'step_target': step_target,
                            'stage1_success': stage1_success
                        })

                        # Update PROJECTED t-level to account for this queued batch
                        # This prevents over-producing curves while CPU is still catching up
                        with projected_lock:
                            projected_curve_history.append(f"{actual_curves}@{b1},{int(b2_for_batch)},p=3")
                            # Pass starting t-level so new curves add to base, not start from t0
                            projected_t_level = calculate_tlevel(projected_curve_history, base_tlevel=config.start_t_level)
                            current_projected = projected_t_level  # Update for next iteration's logging
                            self.logger.info(f"[GPU Thread] Stage 1 batch complete in {stage1_time:.1f}s, projected t-level now {projected_t_level:.2f}")

                        # Decrement remaining
                        remaining_curves -= actual_curves

                        # If factor found, stop producing
                        if stage1_factor:
                            self.logger.info(f"[GPU Thread] Factor found in stage 1, stopping production")
                            break

                    except Exception as e:
                        self.logger.error(f"[GPU Thread] Error in stage 1: {e}")
                        break  # Exit the batch loop on error

            # Send sentinel (only if CPU thread is still running and consuming)
            if not shutdown_event.is_set():
                self.logger.info("[GPU Thread] All batches complete, signaling CPU thread")
                residue_queue.put(None)
            else:
                self.logger.info("[GPU Thread] All batches complete, CPU already stopped (shutdown event set)")

        def cpu_consumer():
            """CPU thread: Process stage 2 from residue queue"""
            nonlocal current_t_level, total_curves

            self.logger.info(f"[CPU Thread] Starting with {config.threads} workers")

            while True:
                # Check for user interrupt
                if self.interrupted:
                    self.logger.info("[CPU Thread] User interrupt, draining queue")
                    shutdown_event.set()
                    break

                # Get next work item
                self.logger.info("[CPU Thread] Waiting for next work item from queue...")
                work_item = residue_queue.get()

                # Check for sentinel
                if work_item is None:
                    self.logger.info("[CPU Thread] Received stop signal")
                    residue_queue.task_done()
                    break

                self.logger.info(f"[CPU Thread] Received work item (stage1_factor={work_item.get('stage1_factor') is not None})")

                try:
                    residue_file = work_item['residue_file']
                    b1 = work_item['b1']
                    b2 = work_item['b2']
                    curves = work_item['curves']
                    stage1_factor = work_item['stage1_factor']
                    all_stage1_factors = work_item['all_factors']
                    stage1_time = work_item['stage1_time']
                    step_target = work_item['step_target']
                    stage1_success = work_item['stage1_success']

                    # Handle stage 1 factor
                    if stage1_factor:
                        self.logger.info(f"[CPU Thread] Factor found in stage 1: {stage1_factor}")
                        with result_lock:
                            # all_stage1_factors is list of (factor, sigma) tuples - unpack them
                            for f, s in all_stage1_factors:
                                all_factors.append(f)
                                all_sigmas.append(s)
                            total_curves += curves
                            # Track with B2=0 (stage 1 only)
                            curve_history.append(f"{curves}@{b1},0,p=3")
                            current_t_level = calculate_tlevel(curve_history, base_tlevel=config.start_t_level)

                        # Submit results with factor (b2=0 since stage 2 never ran)
                        if not config.no_submit and curves > 0:
                            # Extract just the factor strings from (factor, sigma) tuples
                            factor_strings = [f for f, s in all_stage1_factors]
                            step_results = {
                                'success': True,
                                'factors_found': factor_strings,
                                'curves_completed': curves,
                                'execution_time': stage1_time,
                                'composite': config.composite,
                                'method': 'ecm',
                                'b1': b1,
                                'b2': 0,  # Stage 1 only
                                'parametrization': 3
                            }
                            if config.work_id:
                                step_results['work_id'] = config.work_id
                            self.submit_result(step_results, config.project, 'gmp-ecm-ecm')

                        # Signal GPU to stop
                        self.logger.info("[CPU Thread] Stopping GPU production (factor found)")
                        shutdown_event.set()
                        residue_queue.task_done()
                        # Clean up residue file
                        if residue_file.exists():
                            residue_file.unlink()
                        break

                    # Run stage 2
                    if stage1_success and residue_file.exists():
                        self.logger.info(f"[CPU Thread] Starting stage 2: {curves} curves at B1={b1}, B2={b2}")
                        stage2_start = time.time()

                        _, all_stage2_factors, curves_completed, stage2_time, sigma = self._run_stage2_multithread(
                            residue_file=residue_file,
                            b1=b1,
                            b2=b2,
                            workers=config.threads,
                            verbose=config.verbose,
                            progress_interval=config.progress_interval
                        )

                        self.logger.info(f"[CPU Thread] Stage 2 complete in {stage2_time:.1f}s")

                        # Update results
                        with result_lock:
                            if all_stage2_factors:
                                all_factors.extend(all_stage2_factors)
                                all_sigmas.extend([sigma] * len(all_stage2_factors))
                                self.logger.info(f"[CPU Thread] Factor found in stage 2: {all_stage2_factors}")
                                # Signal GPU to stop
                                shutdown_event.set()

                            total_curves += curves_completed
                            # Track with actual B2
                            curve_history.append(f"{curves_completed}@{b1},{int(b2)},p=3")
                            current_t_level = calculate_tlevel(curve_history, base_tlevel=config.start_t_level)
                            self.logger.info(f"[CPU Thread] Current t-level: {current_t_level:.2f}")

                            if all_stage2_factors:
                                self.logger.info("[CPU Thread] Factor found, will break after cleanup")
                                # Submit results with factor before breaking
                                if not config.no_submit and curves_completed > 0:
                                    step_results = {
                                        'success': True,
                                        'factors_found': all_stage2_factors,
                                        'curves_completed': curves_completed,
                                        'execution_time': stage1_time + stage2_time,
                                        'composite': config.composite,
                                        'method': 'ecm',
                                        'b1': b1,
                                        'b2': b2,
                                        'parametrization': 3
                                    }
                                    if config.work_id:
                                        step_results['work_id'] = config.work_id
                                    self.submit_result(step_results, config.project, 'gmp-ecm-ecm')
                                residue_queue.task_done()
                                # Clean up
                                if residue_file.exists():
                                    residue_file.unlink()
                                break

                        self.logger.info("[CPU Thread] Stage 2 complete (no factor), cleaning up")

                        # Submit results for this batch (outside the lock to avoid blocking)
                        if not config.no_submit and curves_completed > 0:
                            step_results = {
                                'success': True,
                                'factors_found': all_stage2_factors if all_stage2_factors else [],
                                'curves_completed': curves_completed,
                                'execution_time': stage1_time + stage2_time,
                                'composite': config.composite,
                                'method': 'ecm',
                                'b1': b1,
                                'b2': b2,
                                'parametrization': 3  # Two-stage uses GPU param
                            }
                            if config.work_id:
                                step_results['work_id'] = config.work_id
                            submit_response = self.submit_result(step_results, config.project, 'gmp-ecm-ecm')
                            if not submit_response:
                                self.logger.warning(f"[CPU Thread] Failed to submit results for B1={b1}")
                    else:
                        self.logger.warning(f"[CPU Thread] Stage 1 failed or no residue file")

                    # Clean up residue file
                    if residue_file.exists():
                        residue_file.unlink()

                    residue_queue.task_done()
                    self.logger.info("[CPU Thread] Batch complete, looping for next work item")

                except Exception as e:
                    self.logger.error(f"[CPU Thread] Error processing batch: {e}")
                    residue_queue.task_done()
                    continue

            # Drain queue to prevent GPU thread from blocking on put()
            # This happens when CPU breaks early (e.g., factor found)
            drained = 0
            while True:
                try:
                    # Non-blocking get: if queue empty, raises queue.Empty immediately
                    item = residue_queue.get_nowait()
                    if item is not None:
                        # Clean up any residue files that weren't processed
                        residue_file = item.get('residue_file')
                        if residue_file and Path(residue_file).exists():
                            Path(residue_file).unlink()
                        drained += 1
                    residue_queue.task_done()
                except queue.Empty:
                    break
            if drained > 0:
                self.logger.info(f"[CPU Thread] Drained {drained} unprocessed items from queue")

        # Start both threads
        gpu_thread = threading.Thread(target=gpu_producer, name="GPU-Stage1")
        cpu_thread = threading.Thread(target=cpu_consumer, name="CPU-Stage2")

        gpu_thread.start()
        cpu_thread.start()

        # Wait for completion
        gpu_thread.join()
        cpu_thread.join()

        # Build result
        result = FactorResult()
        for factor, sigma in zip(all_factors, all_sigmas):
            result.add_factor(factor, sigma)
        result.curves_run = total_curves
        result.execution_time = time.time() - start_time
        result.success = len(all_factors) > 0
        result.t_level_achieved = current_t_level
        result.interrupted = self.interrupted
        result.curve_summary = self._parse_curve_history(curve_history)

        self.logger.info(f"Pipelined t-level execution complete: {total_curves} curves, t{current_t_level:.2f} achieved")

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
        # Use pipelined mode for two-stage (GPU+CPU concurrent)
        if config.use_two_stage:
            return self._run_tlevel_pipelined(config)

        from lib.ecm_math import (get_optimal_b1_for_tlevel, calculate_tlevel,
                                  calculate_curves_to_target_direct, TLEVEL_TRANSITION_CACHE)

        if config.start_t_level > 0:
            self.logger.info(f"Starting progressive ECM with target t{config.target_t_level:.1f} (starting from t{config.start_t_level:.2f})")
        else:
            self.logger.info(f"Starting progressive ECM with target t{config.target_t_level:.1f}")
        start_time = time.time()

        # Accumulated results
        all_factors = []
        all_sigmas = []
        total_curves = 0
        curve_history = []  # Track B1 values and curves for t-level calculation

        # Progressive loop: start from start_t_level (default 0) and work up to target
        current_t_level = config.start_t_level
        step_targets = []

        # Build step targets (every 5 t-levels from 20 to target, plus final target)
        # Skip steps that are below our starting t-level
        t = 20.0
        while t < config.target_t_level:
            if t > current_t_level:  # Only add steps above our starting point
                step_targets.append(t)
            t += 5.0
        # Always add the final target (might not be a multiple of 5)
        if config.target_t_level not in step_targets and config.target_t_level > current_t_level:
            step_targets.append(config.target_t_level)

        # Run ECM at each step
        try:
            for step_target in step_targets:
                # Check for interruption at start of each step
                if self.interrupted:
                    self.logger.info("T-level execution interrupted, returning partial results")
                    break

                # Skip this step if we've already surpassed it
                if current_t_level >= step_target:
                    self.logger.info(f"Skipping t{step_target:.1f} (already at t{current_t_level:.2f})")
                    continue

                # Get optimal B1 for this t-level
                b1, _ = get_optimal_b1_for_tlevel(step_target)

                # Calculate curves needed to reach this step from current position
                # First try the cached table for standard 5-digit increments
                current_rounded = round(current_t_level)
                target_rounded = round(step_target)
                cache_key = (current_rounded, target_rounded, config.parametrization)

                curves: Optional[int] = None  # Will be set below
                if cache_key in TLEVEL_TRANSITION_CACHE:
                    # Use cached value for standard transition
                    cached_b1, cached_curves = TLEVEL_TRANSITION_CACHE[cache_key]
                    if cached_b1 == b1:  # Only use cache if B1 matches
                        curves = cached_curves
                        self.logger.info(f"Using cached transition: t{current_rounded} → t{target_rounded} = {curves} curves at B1={b1}, p={config.parametrization}")
                    else:
                        # B1 doesn't match, call binary
                        curves = calculate_curves_to_target_direct(current_t_level, step_target, b1, config.parametrization)
                else:
                    # Non-standard transition, call t-level binary directly
                    curves = calculate_curves_to_target_direct(current_t_level, step_target, b1, config.parametrization)

                if curves is None or curves <= 0:
                    self.logger.warning(f"Could not calculate curves for t{current_t_level:.3f} → t{step_target:.1f}, skipping to next level")
                    # Skip this target and continue to next one
                    continue

                self.logger.info(f"Running {curves} curves at B1={b1} (targeting t{step_target:.1f}, currently at t{current_t_level:.2f})")

                # Run this batch based on execution mode
                if config.use_two_stage:
                    # Two-stage mode: GPU stage 1 + CPU stage 2
                    self.logger.info(f"Using two-stage mode (GPU stage 1 + CPU stage 2) with {config.threads} workers")
                    step_result = self.run_two_stage_v2(TwoStageConfig(
                        composite=config.composite,
                        b1=b1,
                        b2=None,  # Use default B2 (B1 * 100)
                        stage1_curves=curves,
                        stage1_parametrization=3,  # GPU uses twisted Edwards
                        stage2_parametrization=1,  # CPU uses Montgomery
                        threads=config.threads,
                        verbose=config.verbose,
                        progress_interval=config.progress_interval,
                        project=config.project,
                        no_submit=True  # T-level mode handles its own submissions
                    ))
                elif config.threads > 1:
                    # Multiprocess mode (CPU only)
                    self.logger.info(f"Using multiprocess mode with {config.threads} workers")
                    step_result = self.run_multiprocess_v2(MultiprocessConfig(
                        composite=config.composite,
                        b1=b1,
                        total_curves=curves,
                        num_processes=config.threads,
                        parametrization=config.parametrization,
                        verbose=config.verbose,
                        progress_interval=config.progress_interval
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
                        progress_interval=config.progress_interval
                    ))

                # Check for interruption after execution
                if self.interrupted:
                    self.logger.info("T-level execution interrupted after ECM run, returning partial results")
                    # Still accumulate results from this batch
                    all_factors.extend(step_result.factors)
                    all_sigmas.extend(step_result.sigmas)
                    total_curves += step_result.curves_run
                    break

                # Accumulate results
                all_factors.extend(step_result.factors)
                all_sigmas.extend(step_result.sigmas)
                total_curves += step_result.curves_run

                # Track curve history for t-level calculation (format: "curves@b1,b2,p=parametrization")
                # Two-stage mode uses GPU (p=3) for stage 1, but the full curve credit is for the combined run
                # For t-level calculation purposes, two-stage effectively uses p=3 parametrization
                effective_param = 3 if config.use_two_stage else config.parametrization

                # CRITICAL: Calculate actual B2 used for accurate t-level tracking
                # Two-stage mode uses B1 * 100, which gives ~1.8 t-levels LESS per 100 curves
                # than GMP's default B2. Must include B2 in curve history to prevent overestimating progress.
                # Example: 100@11e6,p=3 → t33.9 (wrong!), 100@11e6,11e8,p=3 → t32.2 (correct)
                if config.use_two_stage:
                    actual_b2 = b1 * 100
                    # Format B2 as integer (no decimals) for t-level binary compatibility
                    curve_history.append(f"{step_result.curves_run}@{b1},{int(actual_b2)},p={effective_param}")
                else:
                    # Standard mode: omit B2 to let t-level binary use GMP default
                    curve_history.append(f"{step_result.curves_run}@{b1},p={effective_param}")

                # Submit this batch's results if enabled
                if not config.no_submit and step_result.curves_run > 0:
                    # Calculate B2 for submission (two-stage uses B1*100, else use GMP default)
                    submission_b2 = (b1 * 100) if config.use_two_stage else None

                    step_results = {
                        'success': True,
                        'factors_found': step_result.factors,
                        'curves_completed': step_result.curves_run,
                        'execution_time': step_result.execution_time,
                        'raw_output': step_result.raw_output,
                        'composite': config.composite,
                        'method': 'ecm',
                        'b1': b1,
                        'b2': submission_b2,
                        'parametrization': effective_param
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
                    result.t_level_achieved = current_t_level  # Track achieved t-level
                    result.curve_summary = self._parse_curve_history(curve_history)
                    return result

                # Update current t-level after this batch
                try:
                    current_t_level = calculate_tlevel(curve_history, base_tlevel=config.start_t_level)
                    self.logger.info(f"Current t-level: {current_t_level:.2f}")
                except (subprocess.CalledProcessError, ValueError, OSError) as e:
                    # If t-level calculation fails (binary error, parse error), approximate it
                    self.logger.warning(f"T-level calculation failed ({type(e).__name__}), using step target: {step_target:.2f}")
                    current_t_level = step_target

                # Check if we've reached overall target
                if current_t_level >= config.target_t_level:
                    self.logger.info(f"Reached target t-level: {current_t_level:.2f} >= {config.target_t_level:.1f}")
                    break

        except KeyboardInterrupt:
            self.logger.info("T-level execution interrupted by Ctrl+C")
            self.interrupted = True
            # Fall through to build partial result

        # Build final FactorResult
        result = FactorResult()
        for factor, sigma in zip(all_factors, all_sigmas):
            result.add_factor(factor, sigma)
        result.curves_run = total_curves
        result.execution_time = time.time() - start_time
        result.success = len(all_factors) > 0
        result.t_level_achieved = current_t_level  # Track achieved t-level
        result.interrupted = self.interrupted  # Signal that execution was interrupted
        result.curve_summary = self._parse_curve_history(curve_history)

        return result

    def _parse_curve_history(self, curve_history: List[str]) -> List[Dict[str, Any]]:
        """
        Parse curve history strings into structured summary data.

        Args:
            curve_history: List of strings in format "{curves}@{b1},{b2},p={param}" or "{curves}@{b1},p={param}"

        Returns:
            List of dictionaries with B1, B2, curves, and parametrization
        """
        import re
        summary = []

        for entry in curve_history:
            # Pattern: "{curves}@{b1},{b2},p={param}" or "{curves}@{b1},p={param}"
            # Also handle: "{curves}@{b1},0,p={param}" for stage1-only
            match = re.match(r'(\d+)@(\d+),?(\d+)?,p=(\d+)', entry)
            if match:
                curves = int(match.group(1))
                b1 = int(match.group(2))
                b2_str = match.group(3)
                param = int(match.group(4))

                # Parse B2 (None = default, 0 = stage1 only, number = explicit B2)
                if b2_str is None:
                    b2 = None  # GMP-ECM default
                else:
                    b2 = int(b2_str)

                # Determine mode
                if b2 == 0:
                    mode = "Stage1 Only"
                elif b2 is None:
                    mode = "ECM (default B2)"
                else:
                    mode = "Two-Stage"

                summary.append({
                    'b1': b1,
                    'b2': b2,
                    'curves': curves,
                    'parametrization': param,
                    'mode': mode
                })

        return summary

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
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Execute GMP-ECM primitive operation.

        Single source of truth for all ECM execution. Handles 3 modes:
        1. Full ECM (B1 + B2) - default
        2. Stage 1 only - when residue_save is set
        3. Stage 2 only - when residue_load is set

        Note: No timeout is enforced - ECM factorization can run for extended periods.

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

        except KeyboardInterrupt:
            # KeyboardInterrupt doesn't inherit from Exception in Python 3
            # Handle it explicitly so we can return a proper result
            self.logger.info(f"{method.upper()} execution interrupted by user")
            self.interrupted = True
            return {
                'success': False,
                'factors': [],
                'sigmas': [],
                'curves_completed': 0,
                'raw_output': 'Interrupted by user',
                'parametrization': param if param is not None else 3,
                'exit_code': -2,
                'interrupted': True
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
        # Use factory method for base fields, then customize
        execution_time = time.time() - start_time
        result = FactorResult.from_primitive_result(prim_result, execution_time)

        # Clear factors - we'll re-add after recursive factoring
        original_factors = list(zip(prim_result.get('factors', []), prim_result.get('sigmas', [])))
        result.factors = []
        result.sigmas = []

        # Recursively factor each discovered factor to get all primes
        for factor, sigma in original_factors:
            # Fully factor this composite to get all prime factors
            prime_factors = self._fully_factor_found_result(factor, quiet=False)

            # Add each prime factor with the original sigma
            # (the sigma that found the composite factor)
            for prime in prime_factors:
                result.add_factor(prime, sigma)

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
        stage1_attempt_id: Optional[int],
        factor_found: Optional[str],
        client_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Upload residue file to server only if no factor was found.

        When a factor is found in stage 1, there's no point in running stage 2,
        so we skip the residue upload.

        Args:
            residue_file: Path to the residue file
            stage1_attempt_id: Attempt ID (int) from stage1 submission (None if submission failed)
            factor_found: Factor found in stage 1 (None if no factor)
            client_id: Client identifier for upload

        Returns:
            Upload result dict if uploaded, None if skipped or failed
        """
        if not factor_found and stage1_attempt_id:
            # No factor found and we have attempt ID - upload residue for potential stage 2
            print(f"Uploading residue file ({residue_file.stat().st_size} bytes)...")
            api_client = self._get_api_client()
            upload_result = api_client.upload_residue(
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
        """Override base class method to get GMP-ECM version."""
        return self.get_ecm_version()

    def get_ecm_version(self) -> str:
        """Get GMP-ECM version."""
        from lib.parsing_utils import get_binary_version
        return get_binary_version(
            self.config['programs']['gmp_ecm']['path'],
            'ecm'
        )

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

            # Select B1 based on cofactor size - target factors up to half the digits
            cofactor_digits = len(str(current_cofactor))
            target_digits = (cofactor_digits + 1) // 2
            b1 = get_b1_for_digit_length(target_digits)

            # Use more curves for smaller numbers (they're faster)
            curves = max(10, 50 - (target_digits // 2))

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

                        # Divide out from cofactor (use while loop to handle repeated factors)
                        for prime in sub_primes:
                            while current_cofactor % int(prime) == 0:
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

    def _fully_factor_composite(self, factor: str, max_ecm_attempts: int = 5) -> List[str]:
        """
        Recursively factor a composite using primitives directly (no run_ecm_v2).

        This method is called from run_multiprocess_v2 to factor any composite
        factors found. It uses _execute_ecm_primitive directly to avoid infinite
        recursion through run_ecm_v2 -> run_multiprocess_v2 -> _fully_factor_composite.

        Args:
            factor: Factor found by ECM (may be composite)
            max_ecm_attempts: Maximum ECM attempts with increasing B1

        Returns:
            List of prime factors (as strings)
        """
        from lib.ecm_math import trial_division, is_probably_prime, get_b1_for_digit_length

        factor_int = int(factor)

        # Trial division catches small factors quickly
        small_primes, cofactor = trial_division(factor_int, limit=10**7)
        all_primes = [str(p) for p in small_primes]

        if cofactor == 1:
            return all_primes

        # Check if cofactor is prime
        if is_probably_prime(cofactor):
            self.logger.info(f"Cofactor {cofactor} is prime")
            all_primes.append(str(cofactor))
            return all_primes

        # Cofactor is composite - use ECM primitive directly
        digit_length = len(str(cofactor))
        self.logger.info(f"Factoring composite cofactor C{digit_length} using ECM primitive")

        current_cofactor = cofactor
        for attempt in range(max_ecm_attempts):
            if current_cofactor == 1:
                break

            cofactor_digits = len(str(current_cofactor))
            # Target factors up to half the digit length (smallest factor can't be larger)
            target_digits = (cofactor_digits + 1) // 2
            b1 = get_b1_for_digit_length(target_digits)
            curves = max(10, 50 - (target_digits // 2))

            self.logger.info(f"ECM attempt {attempt+1}/{max_ecm_attempts} on C{cofactor_digits} with B1={b1}")

            try:
                # Call primitive directly - no recursive factoring
                prim_result = self._execute_ecm_primitive(
                    composite=str(current_cofactor),
                    b1=b1,
                    curves=curves,
                    verbose=False
                )

                found_factors = prim_result.get('factors', [])

                if found_factors:
                    self.logger.info(f"Found {len(found_factors)} factor(s): {found_factors}")

                    for found_factor in found_factors:
                        # Recursively factor each found factor
                        sub_primes = self._fully_factor_composite(found_factor, max_ecm_attempts)
                        all_primes.extend(sub_primes)

                        # Divide out from cofactor
                        for prime in sub_primes:
                            while current_cofactor % int(prime) == 0:
                                current_cofactor //= int(prime)

                    if current_cofactor == 1:
                        break

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

        # If we still have a composite cofactor, return it as-is
        if current_cofactor > 1:
            self.logger.warning(f"Could not fully factor C{len(str(current_cofactor))}: {current_cofactor}")
            all_primes.append(str(current_cofactor))

        return all_primes
