#!/usr/bin/env python3
"""
Work mode strategy pattern for ECM auto-work execution.

This module implements the Strategy pattern for different auto-work modes,
eliminating the duplicated work loop logic in ecm_client.py.

Each mode implements the same abstract interface:
- request_work() - Get work assignment from server
- execute_work() - Run ECM on the assignment
- submit_results() - Submit results to server
- complete_work() - Mark work as complete
- cleanup_on_failure() - Mode-specific cleanup

The base class provides the work loop template that all modes share.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Dict, TYPE_CHECKING
import argparse
import hashlib
import signal
import time

from .ecm_config import (
    ECMConfig, TwoStageConfig, MultiprocessConfig, TLevelConfig, FactorResult
)
from .ecm_math import get_optimal_b1_for_tlevel
from .work_helpers import print_work_header, print_work_status, request_ecm_work, request_p1_work
from .stage1_helpers import submit_stage1_complete_workflow
from .error_helpers import check_work_limit_reached
from .cleanup_helpers import handle_shutdown
from .results_builder import results_for_stage1
from .arg_parser import resolve_gpu_settings, resolve_worker_count, get_workers_default, get_max_batch_default
from .ecm_arg_helpers import parse_sigma_arg, resolve_param
from .api_client import ResourceNotFoundError

if TYPE_CHECKING:
    from .ecm_executor import ECMWrapper


# Circuit breaker threshold
MAX_CONSECUTIVE_FAILURES = 3


@dataclass
class WorkLoopContext:
    """
    Shared context for all work loop modes.

    Contains all the state and configuration needed to execute work loops.
    Passed to WorkMode constructors to avoid long argument lists.
    """
    wrapper: 'ECMWrapper'
    client_id: str
    args: argparse.Namespace
    work_count_limit: Optional[int] = None
    finish_after_current: bool = field(default=False, init=False)

    def __post_init__(self):
        """Ensure API clients are initialized."""
        self.wrapper._ensure_api_clients()


class WorkMode(ABC):
    """
    Abstract base class for auto-work execution modes.

    Implements the Template Method pattern: the run() method provides
    the work loop structure, while subclasses implement the specific
    behavior for each step.

    Subclasses must implement:
    - mode_name: Human-readable name for logging
    - request_work(): Get work from server
    - execute_work(): Run factorization
    - submit_results(): Submit to API
    - complete_work(): Finalize assignment

    Optional overrides:
    - cleanup_on_failure(): Mode-specific cleanup
    - cleanup_on_shutdown(): Cleanup for graceful shutdown
    """

    # Subclasses should set this
    mode_name: str = "Unknown Mode"

    def __init__(self, ctx: WorkLoopContext):
        self.ctx = ctx
        self.wrapper = ctx.wrapper
        self.api_client = ctx.wrapper._get_api_client()
        self.logger = ctx.wrapper.logger
        self.args = ctx.args

        # Work tracking state
        self.current_work_id: Optional[str] = None
        self.current_residue_id: Optional[int] = None
        self.completed_count: int = 0
        self.consecutive_failures: int = 0

        # Graceful shutdown state (3-level)
        self._first_interrupt_received: bool = False
        self._second_interrupt_received: bool = False
        self._original_sigint_handler: Any = None

    @abstractmethod
    def request_work(self) -> Optional[Dict[str, Any]]:
        """
        Request work assignment from server.

        Returns:
            Work assignment dictionary, or None if no work available.
            Should handle retry/wait logic internally.
        """
        pass

    @abstractmethod
    def execute_work(self, work: Dict[str, Any]) -> FactorResult:
        """
        Execute factorization on the work assignment.

        Args:
            work: Work assignment from request_work()

        Returns:
            FactorResult with execution results
        """
        pass

    @abstractmethod
    def submit_results(self, work: Dict[str, Any], result: FactorResult) -> bool:
        """
        Submit results to API server.

        Args:
            work: Original work assignment
            result: Execution result from execute_work()

        Returns:
            True if submission succeeded, False otherwise
        """
        pass

    @abstractmethod
    def complete_work(self, work: Dict[str, Any]) -> None:
        """
        Mark work assignment as complete on server.

        Args:
            work: Work assignment to complete
        """
        pass

    def cleanup_on_failure(self, work: Optional[Dict[str, Any]], error: BaseException) -> None:
        """
        Mode-specific cleanup after a failure.

        Override in subclasses for custom cleanup behavior.
        Default implementation abandons work if we have a work_id.

        Args:
            work: Work assignment that failed (may be None)
            error: The exception that occurred (can be Exception or KeyboardInterrupt)
        """
        if self.current_work_id:
            self.wrapper.abandon_work(self.current_work_id, reason="execution_error")
            self.current_work_id = None

    def cleanup_on_shutdown(self) -> None:
        """
        Cleanup for graceful shutdown (Ctrl+C).

        Override in subclasses for mode-specific shutdown behavior.
        """
        pass

    def on_work_started(self, work: Dict[str, Any]) -> None:
        """
        Called when work is received, before execution.

        Override to store work-specific state or print headers.
        Default implementation stores work_id.
        """
        self.current_work_id = work.get('work_id')

    def on_work_completed(self, work: Dict[str, Any], result: FactorResult) -> None:
        """
        Called after successful completion.

        Override for mode-specific completion handling.
        """
        self.current_work_id = None
        self.completed_count += 1
        self.consecutive_failures = 0
        # Reset graceful shutdown flags after each work unit completes
        # (in case finish_after_current is cleared and loop continues)
        self.wrapper.graceful_shutdown_requested = False
        self.wrapper.shutdown_level = 0
        self.wrapper.stop_event.clear()

    def _setup_signal_handler(self) -> None:
        """
        Install signal handler for graceful shutdown (3 levels).

        1st Ctrl+C: Set finish_after_current flag - let the entire current
                    assignment finish (all remaining curves), submit results, then exit.
                    Does NOT interrupt execution in progress.
        2nd Ctrl+C: Signal workers to finish current curve then stop.
                    Sets graceful_shutdown_requested + stop_event.
        3rd Ctrl+C: Raise KeyboardInterrupt for immediate abort.
        """
        def handler(signum, frame):
            if not self._first_interrupt_received:
                # First interrupt: finish entire current assignment, then exit
                self._first_interrupt_received = True
                self.ctx.finish_after_current = True
                # Do NOT set graceful_shutdown_requested - let the full assignment complete
                print("\n")
                print("=" * 60)
                print("Will complete current assignment, then exit.")
                print("Press Ctrl+C again to stop after current curve.")
                print("=" * 60)
            elif not self._second_interrupt_received:
                # Second interrupt: stop after current curve
                self._second_interrupt_received = True
                self.wrapper.graceful_shutdown_requested = True
                self.wrapper.stop_event.set()
                print("\n")
                print("=" * 60)
                print("Stopping after current curve...")
                print("Press Ctrl+C again to abort immediately.")
                print("=" * 60)
            else:
                # Third interrupt: immediate abort
                raise KeyboardInterrupt()

        self._original_sigint_handler = signal.signal(signal.SIGINT, handler)

    def _restore_signal_handler(self) -> None:
        """Restore original signal handler and reset graceful shutdown flags."""
        if self._original_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._original_sigint_handler)
            self._original_sigint_handler = None
        # Reset graceful shutdown flags for wrapper
        self.wrapper.graceful_shutdown_requested = False
        self.wrapper.shutdown_level = 0
        self.wrapper.stop_event.clear()

    def _drain_queue(self) -> None:
        """Drain the submission queue, retrying any pending items."""
        queue = self.wrapper.submission_queue
        if queue.count() > 0:
            queue.drain(self.api_client)

    def should_continue(self) -> bool:
        """
        Check if work loop should continue.

        Returns:
            True to continue, False to exit loop
        """
        # Check for graceful shutdown request (first Ctrl+C)
        if self.ctx.finish_after_current:
            return False

        # Check for hard interruption
        if self.wrapper.interrupted:
            return False

        # Check work count limit
        if self.ctx.work_count_limit and self.completed_count >= self.ctx.work_count_limit:
            return False

        return True

    def run(self) -> int:
        """
        Main work loop - Template Method pattern.

        This method provides the skeleton algorithm that all modes share.
        Subclasses customize behavior by overriding the abstract methods.

        Returns:
            Number of work assignments completed
        """
        self._print_startup_banner()
        self._setup_signal_handler()

        # Drain submission queue on startup (retry any pending items from previous runs)
        self._drain_queue()

        try:
            b2_dict = dict()
            k_dict = dict()
            if self.args.b2_dictionary != None:
                try:
                    with open(self.args.b2_dictionary, 'r') as b2_dict_file:
                        for line in b2_dict_file:
                            if line.startswith('#') or line.startswith("'") or line.startswith('--') or not line:
                                continue
                            entries = [l for l in line.strip().split(' ') if l]
                            if len(entries) < 2: # extra entries are treated as comments
                                print("Warning: A line in the B2 dictionary had a wrong format!")
                                continue
                            key = 0
                            value = 0
                            k = 0
                            try:
                                key = int(entries[0]) if 'e' not in entries[0].lower() else int(float(entries[0]) + 0.5)
                                value = int(entries[1]) if 'e' not in entries[1].lower() else int(float(entries[1]) + 0.5)
                                if len(entries) > 2:
                                    k_entry = entries[2]
                                    if not (k_entry.startswith('#') and k_entry.startswith("'") and k_entry.startswith('--')):
                                        k = int(k_entry)
                            except ValueError:
                                print("Warning: An entry in the B2 dictionary had the wrong format!")
                                continue
                            b2_dict[key] = value
                            if k > 0:
                                k_dict[key] = k
                except:
                    print("The B2 file could not be accessed/loaded! Using defaults…")

            while self.should_continue():
                # Drain submission queue before each work request
                self._drain_queue()

                # Request work from server
                work = self.request_work()
                if not work:
                    continue

                if 'b1' in work and work['b1'] in b2_dict:
                    b1 = work['b1']
                    work['b2_from_dict'] = b2_dict[b1]
                    print(f"Using B2 = {b2_dict[b1]} from dictionary.")
                    if b1 in k_dict:
                        work['k_from_dict'] = k_dict[b1]

                # Track work assignment
                self.on_work_started(work)

                try:
                    # Execute factorization
                    result = self.execute_work(work)

                    # After execution returns, check if Ctrl+C was pressed during execution.
                    # The executor's signal handler sets shutdown_level but does NOT set
                    # finish_after_current (which is a work-loop concern). Sync the state here.
                    if self.wrapper.shutdown_level >= 1 and not self.ctx.finish_after_current:
                        self.ctx.finish_after_current = True

                    # Check for hard interruption during execution
                    if self.wrapper.interrupted:
                        self.logger.info(f"{self.mode_name} interrupted by user, cleaning up...")
                        self.cleanup_on_failure(work, KeyboardInterrupt())
                        break

                    # Submit results
                    if not self.submit_results(work, result):
                        self.consecutive_failures += 1
                        self.cleanup_on_failure(work, RuntimeError("Submission failed"))

                        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                            self.logger.error(
                                f"Too many consecutive failures ({self.consecutive_failures}), exiting..."
                            )
                            break
                        continue

                    # Mark complete
                    self.complete_work(work)
                    self.on_work_completed(work, result)

                    # Print status and check limit
                    if print_work_status(self.mode_name, self.completed_count, self.ctx.work_count_limit):
                        break

                except Exception as e:
                    self.consecutive_failures += 1
                    self.logger.exception(f"Error in {self.mode_name}: {e}")
                    self.cleanup_on_failure(work, e)

                    if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        self.logger.error(
                            f"Too many consecutive failures ({self.consecutive_failures}), exiting..."
                        )
                        break

                    if check_work_limit_reached(self.completed_count, self.ctx.work_count_limit):
                        break

            # Check if we exited due to graceful shutdown
            if self.ctx.finish_after_current:
                self._handle_graceful_exit()

        except KeyboardInterrupt:
            self._handle_keyboard_interrupt()

        finally:
            self._restore_signal_handler()

        return self.completed_count

    def _print_startup_banner(self) -> None:
        """Print mode startup banner."""
        print("=" * 60)
        if self.ctx.work_count_limit:
            print(f"{self.mode_name} - will process {self.ctx.work_count_limit} assignment(s)")
        else:
            print(f"{self.mode_name} - requesting work from server")
        print("Ctrl+C once: finish current assignment, then exit")
        print("Ctrl+C twice: stop after current curve")
        print("Ctrl+C three times: abort immediately")
        print("=" * 60)
        print()

    def _handle_graceful_exit(self) -> None:
        """Handle graceful exit after completing current work."""
        print()
        print("=" * 60)
        print(f"{self.mode_name} - graceful shutdown complete")
        print(f"Completed {self.completed_count} assignment(s)")
        print("=" * 60)

    def _handle_keyboard_interrupt(self) -> None:
        """Handle immediate abort (second Ctrl+C)."""
        self.cleanup_on_shutdown()
        handle_shutdown(
            wrapper=self.wrapper,
            current_work_id=self.current_work_id,
            current_residue_id=self.current_residue_id,
            mode_name=self.mode_name,
            completed_count=self.completed_count
        )


class Stage1ProducerMode(WorkMode):
    """
    Stage 1 Producer mode: GPU execution, upload residues to server.

    This mode:
    1. Requests regular ECM work from server
    2. Runs stage 1 only (B2=0) to generate residues
    3. Submits stage 1 results
    4. Uploads residue file for stage 2 consumers

    GPU-specific Ctrl+C handling (2 levels):
    - 1st Ctrl+C: Print message, GPU keeps running. Submit + upload when done, then exit.
    - 2nd Ctrl+C: Immediate abort.
    """

    mode_name = "Stage 1 Producer (GPU)"

    def __init__(self, ctx: WorkLoopContext):
        super().__init__(ctx)
        self.residue_file: Optional[Path] = None

    def _setup_signal_handler(self) -> None:
        """
        GPU-specific signal handler with 2 levels.

        GPU can't be interrupted mid-batch, so level 1 just flags for exit after
        GPU finishes. Level 2 aborts immediately.
        """
        def handler(signum, frame):
            if not self._first_interrupt_received:
                # First interrupt: GPU keeps running, submit when done, then exit
                self._first_interrupt_received = True
                self.ctx.finish_after_current = True
                print("\n")
                print("=" * 60)
                print("GPU batch will complete. Results will be submitted, then exit.")
                print("Press Ctrl+C again to abort immediately.")
                print("=" * 60)
            else:
                # Second interrupt: immediate abort
                raise KeyboardInterrupt()

        self._original_sigint_handler = signal.signal(signal.SIGINT, handler)

    def _print_startup_banner(self) -> None:
        """Print GPU mode startup banner."""
        print("=" * 60)
        if self.ctx.work_count_limit:
            print(f"{self.mode_name} - will process {self.ctx.work_count_limit} assignment(s)")
        else:
            print(f"{self.mode_name} - requesting work from server")
        print("Ctrl+C once: finish GPU batch, submit results, then exit")
        print("Ctrl+C twice: abort immediately")
        print("=" * 60)
        print()

    def request_work(self) -> Optional[Dict[str, Any]]:
        return request_ecm_work(
            self.api_client,
            self.ctx.client_id,
            self.args,
            self.logger
        )

    # Minimum B1 for stage1-only mode to prevent overloading server with too-fast submissions
    # Low B1 values (e.g., 11000 for t20) complete in seconds on GPU, causing submission spam
    MIN_STAGE1_B1 = 250000  # ~t30 level

    def _calculate_stage1_params(self, work: Dict[str, Any]) -> tuple:
        """
        Calculate optimal B1/curves for stage 1 based on t-level info.

        If --b1 and --curves are specified, uses those.
        Otherwise picks appropriate B1 for current t-level and runs one GPU batch.

        For stage1-only mode, the goal is simple:
        - Pick the right B1 for where we are
        - Run one batch (GPU batch size)
        - Let server track actual t-level achieved

        Returns:
            Tuple of (b1, curves)
        """
        # If B1 is explicitly specified, use it (but still enforce minimum)
        if self.args.b1 is not None:
            b1 = self.args.b1
            if b1 < self.MIN_STAGE1_B1:
                self.logger.warning(f"B1={b1} below minimum {self.MIN_STAGE1_B1} for stage1-only, using minimum")
                b1 = self.MIN_STAGE1_B1
            curves = self.args.curves if self.args.curves is not None else \
                     self.wrapper.config['programs']['gmp_ecm']['default_curves']
            return b1, curves

        # Get current t-level to determine appropriate B1
        current_t = work.get('current_t_level', 0.0) or 0.0

        # Get optimal B1 for the current t-level (rounds up to next standard level)
        # e.g., t54.7 -> use B1 for t55
        target_for_b1 = max(20, int(current_t) + 1)  # At least t20
        b1, _ = get_optimal_b1_for_tlevel(target_for_b1)

        # Enforce minimum B1 to prevent submission spam from too-fast GPU runs
        if b1 < self.MIN_STAGE1_B1:
            self.logger.info(f"B1={b1} (t{target_for_b1}) below minimum, using B1={self.MIN_STAGE1_B1}")
            b1 = self.MIN_STAGE1_B1

        # Use GPU batch size for curves (one batch per work unit)
        # Check args.curves first, then config, then default
        # NOTE: GMP-ECM GPU rounds up to its natural batch size (e.g., 2304, 3072).
        # If we request MORE than the batch size, it runs multiple full batches.
        # Default to 1000 which is always <= any GPU batch size, ensuring exactly one batch.
        if self.args.curves is not None:
            curves = self.args.curves
        else:
            gpu_config = self.wrapper.config.get('programs', {}).get('gmp_ecm', {}).get('gpu', {})
            curves = gpu_config.get('curves_per_batch', 1000)

        self.logger.info(f"Stage 1: t{current_t:.1f} using B1={b1}, curves={curves} (one batch)")
        return b1, curves

    def on_work_started(self, work: Dict[str, Any]) -> None:
        super().on_work_started(work)

        # Store work for execute_work to use
        self._current_work = work

        # Calculate B1/curves (may use t-level info)
        b1, curves = self._calculate_stage1_params(work)
        self._stage1_b1 = b1
        self._stage1_curves = curves

        print_work_header(
            work_id=self.current_work_id,
            composite=work['composite'],
            digit_length=work['digit_length'],
            params={'B1': b1, 'curves': curves,
                    'T-level': f"{work.get('current_t_level', 0):.1f} -> {work.get('target_t_level', 0):.1f}"}
        )

    def execute_work(self, work: Dict[str, Any]) -> FactorResult:
        composite = work['composite']

        # Use pre-calculated B1/curves from on_work_started
        b1 = self._stage1_b1
        curves = self._stage1_curves

        # Resolve GPU settings
        use_gpu, gpu_device, gpu_curves = resolve_gpu_settings(self.args, self.wrapper.config)

        # Generate residue file path
        residue_dir = Path(self.wrapper.config['execution'].get('residue_dir', 'data/residues'))
        residue_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.residue_file = residue_dir / f"stage1_{timestamp}_{composite[:20]}.txt"

        # Resolve parameters
        sigma = parse_sigma_arg(self.args)
        param = resolve_param(self.args, use_gpu)

        print(f"Running ECM stage 1 (B1={b1}, curves={curves})...")
        print(f"Saving residues to: {self.residue_file}")

        # Run stage 1
        success, factor, actual_curves, raw_output, all_factors = self.wrapper._run_stage1(
            composite=composite,
            b1=b1,
            curves=curves,
            residue_file=self.residue_file,
            sigma=sigma,
            param=param,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            gpu_curves=gpu_curves,
            verbose=self.args.verbose
        )

        # Build FactorResult
        result = FactorResult()
        result.success = success
        result.curves_run = actual_curves
        result.raw_output = raw_output

        for f, s in all_factors:
            result.add_factor(f, s)

        # Store factor for submit_results
        self._last_factor = factor
        self._last_param = param if param is not None else 3
        self._last_curves = actual_curves
        self._last_output = raw_output
        self._last_all_factors = all_factors

        return result

    def submit_results(self, work: Dict[str, Any], result: FactorResult) -> bool:
        if not result.success:
            self.logger.error("Stage 1 execution failed")
            return False

        composite = work['composite']

        # Build results using ResultsBuilder (use pre-calculated B1)
        builder = (results_for_stage1(composite, self._stage1_b1, self._last_curves, self._last_param)
            .with_curves(self._last_curves, self._last_curves)
            .with_factors(self._last_all_factors)
            .add_raw_output(self._last_output)
            .with_execution_time(result.execution_time))

        if self.current_work_id:
            builder.with_work_id(self.current_work_id)

        results = builder.build()

        # Submit stage 1 results and handle workflow
        assert self.residue_file is not None  # Set in execute_work
        stage1_attempt_id = submit_stage1_complete_workflow(
            wrapper=self.wrapper,
            results=results,
            residue_file=self.residue_file,
            work_id=self.current_work_id,
            project=self.args.project,
            client_id=self.ctx.client_id,
            factor_found=self._last_factor,
            cleanup_residue=True
        )

        return stage1_attempt_id is not None

    def complete_work(self, work: Dict[str, Any]) -> None:
        assert self.current_work_id is not None  # Set in on_work_started
        try:
            if not self.api_client.complete_work(self.current_work_id, self.ctx.client_id):
                self.wrapper.submission_queue.enqueue_work_completion(
                    self.current_work_id, self.ctx.client_id
                )
        except ResourceNotFoundError:
            self.logger.warning(f"Work {self.current_work_id} already expired/completed on server, skipping")

    def cleanup_on_failure(self, work: Optional[Dict[str, Any]], error: BaseException) -> None:
        if self.current_work_id:
            self.wrapper.abandon_work(self.current_work_id, reason="stage1_failed")
            self.current_work_id = None

        # Clean up residue file
        if self.residue_file and self.residue_file.exists():
            self.residue_file.unlink()
            self.residue_file = None


class Stage2ConsumerMode(WorkMode):
    """
    Stage 2 Consumer mode: Download residues from server, CPU processing.

    This mode:
    1. Requests residue work from server
    2. Downloads the residue file
    3. Runs stage 2 processing
    4. Submits results (supersedes stage 1 attempt)
    5. Completes residue work
    """

    mode_name = "Stage 2 Consumer (CPU)"

    def __init__(self, ctx: WorkLoopContext):
        super().__init__(ctx)
        self._b2 = None
        self._k = None
        self.local_residue_file: Optional[Path] = None
        self._residue_checksum: Optional[str] = None
        # Track curves for completion validation
        self._expected_curves: int = 0
        self._curves_completed: int = 0
        self._found_factor: bool = False
        self._raw_output: str = ""  # Aggregated ECM output from workers
        self._primary_submission_failed: bool = False
        # Import here to avoid circular dependency
        from .stage2_executor import Stage2Executor
        self.Stage2Executor = Stage2Executor

    def _compute_file_checksum(self, filepath: Path) -> str:
        """Compute SHA-256 checksum of file for residue verification."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def request_work(self) -> Optional[Dict[str, Any]]:
        residue_work = self.api_client.get_residue_work(
            client_id=self.ctx.client_id,
            min_target_tlevel=getattr(self.args, 'min_target_tlevel', None),
            max_target_tlevel=getattr(self.args, 'max_target_tlevel', None),
            min_priority=getattr(self.args, 'priority', None),
            min_b1=getattr(self.args, 'min_b1', None),
            max_b1=getattr(self.args, 'max_b1', None),
            claim_timeout_hours=24
        )

        if not residue_work:
            self.logger.info("No residue work available, waiting 30 seconds before retry...")
            time.sleep(30)
            return None

        return residue_work

    def on_work_started(self, work: Dict[str, Any]) -> None:
        self.current_residue_id = work['residue_id']
        self.current_work_id = None  # Stage 2 uses residue_id, not work_id

        # Track expected curves for completion validation
        self._expected_curves = work['curve_count']
        self._curves_completed = 0
        self._found_factor = False
        self._raw_output = ""

        b1 = work['b1']

        # Determine B2 and optionally k
        k = 0
        if 'b2_from_dict' in work:
            b2 = work['b2_from_dict']
            if 'k_from_dict' in work:
                k = work['k_from_dict']
        elif self.args.b2 is not None:
            b2 = self.args.b2
        elif hasattr(self.args, 'b2_multiplier') and self.args.b2_multiplier is not None:
            b2 = int(b1 * self.args.b2_multiplier)
            print(f"Using dynamic B2 = B1 * {self.args.b2_multiplier} = {b2}")
        else:
            b2 = work.get('suggested_b2', b1 * 100)

        self._b2 = b2
        self._k = k if k > 0 else None
        b2_display = "GMP-ECM default" if b2 == -1 else str(b2)

        print_work_header(
            work_id=str(self.current_residue_id) if self.current_residue_id else None,
            composite=work['composite'],
            digit_length=work['digit_length'],
            params={
                'B1': b1,
                'B2': b2_display,
                'curves': work['curve_count'],
                'Stage 1 attempt ID': work.get('stage1_attempt_id')
            }
        )

    def execute_work(self, work: Dict[str, Any]) -> FactorResult:
        # Download residue file
        residue_dir = Path(self.wrapper.config['execution'].get('residue_dir', 'data/residues'))
        residue_dir.mkdir(parents=True, exist_ok=True)
        self.local_residue_file = residue_dir / f"s2_residue_{self.current_residue_id}.txt"

        print("Downloading residue file...")
        assert self.current_residue_id is not None  # Set in on_work_started
        download_success = self.api_client.download_residue(
            client_id=self.ctx.client_id,
            residue_id=self.current_residue_id,
            output_path=str(self.local_residue_file)
        )

        if not download_success:
            result = FactorResult()
            result.success = False
            result.error_message = "Failed to download residue file"
            return result

        file_size = self.local_residue_file.stat().st_size
        print(f"Downloaded {file_size} bytes")

        # Compute checksum for residue verification
        self._residue_checksum = self._compute_file_checksum(self.local_residue_file)
        self.logger.debug(f"Residue checksum: {self._residue_checksum}")

        # Get workers count
        workers = getattr(self.args, 'workers', None) or \
                  get_workers_default(self.wrapper.config)

        # Run stage 2
        print(f"Running stage 2 with {workers} workers...")
        executor = self.Stage2Executor(
            self.wrapper,
            self.local_residue_file,
            work['b1'],
            self._b2,
            self._k,
            workers,
            self.args.verbose
        )

        factor, all_factors, curves, exec_time, sigma = executor.execute(
            early_termination=not getattr(self.args, 'continue_after_factor', False),
            progress_interval=getattr(self.args, 'progress_interval', 0)
        )

        # Build FactorResult
        result = FactorResult()
        result.success = True
        result.curves_run = curves
        result.execution_time = exec_time

        if all_factors:
            for f in all_factors:
                result.add_factor(f, sigma)

        # Store for submit_results and complete_work
        self._work = work
        self._factor = factor
        self._sigma = sigma
        self._curves_completed = curves
        self._found_factor = bool(all_factors)
        self._raw_output = executor.raw_output  # Aggregated output from all workers

        return result

    def submit_results(self, work: Dict[str, Any], result: FactorResult) -> bool:
        if not result.success:
            self.logger.error(result.error_message or "Stage 2 execution failed")
            return False

        # Build results dict
        results = {
            'composite': work['composite'],
            'b1': work['b1'],
            'b2': None if self._b2 == -1 else self._b2,
            'curves_requested': work['curve_count'],
            'curves_completed': result.curves_run,
            'factors_found': result.factors,
            'factor_found': self._factor,
            'sigma': self._sigma,
            'raw_output': self._raw_output or f"Stage 2 from residue {self.current_residue_id}",
            'method': 'ecm',
            'parametrization': work.get('parametrization', 3),
            'execution_time': result.execution_time,
            'residue_checksum': self._residue_checksum,  # For orphan detection
        }

        print("Submitting stage 2 results...")
        program_name = 'gmp-ecm-ecm'
        submit_response = self.wrapper.submit_result(results, self.args.project, program_name)

        if not submit_response:
            self.logger.error("Failed to submit stage 2 results")
            return False

        # Use primary endpoint's response for attempt_id (needed for complete_residue)
        primary = submit_response.primary_response
        if primary:
            stage2_attempt_id = primary.get('attempt_id')
            if not stage2_attempt_id:
                self.logger.error("No attempt_id returned from primary endpoint")
                return False
            print(f"Stage 2 attempt ID: {stage2_attempt_id}")
            self._stage2_attempt_id = stage2_attempt_id
            self._primary_submission_failed = False
        else:
            # Primary failed but another endpoint succeeded - can't call complete_residue
            self.logger.warning("Primary endpoint submission failed (other endpoints may have succeeded)")
            self.logger.warning("Skipping residue completion - failed submission saved for retry via resend_failed.py")
            self._stage2_attempt_id = None
            self._primary_submission_failed = True

        return True

    def complete_work(self, work: Dict[str, Any]) -> None:
        assert self.current_residue_id is not None  # Set in on_work_started

        # If primary endpoint submission failed, we can't complete the residue
        # (the attempt_id would be from a different endpoint)
        if self._primary_submission_failed:
            self.logger.warning(
                f"Skipping complete_residue for residue {self.current_residue_id} - "
                "primary endpoint submission failed, resubmit via resend_failed.py first"
            )
            self.api_client.abandon_residue(self.ctx.client_id, self.current_residue_id)
            return

        # Server requires 75% completion if no factor found
        # If we didn't complete enough curves (e.g., graceful shutdown), abandon instead
        completion_ratio = self._curves_completed / self._expected_curves if self._expected_curves > 0 else 0
        min_completion = 0.75

        if not self._found_factor and completion_ratio < min_completion:
            # Not enough curves completed - abandon to release back to pool
            print(f"Abandoning residue (only {self._curves_completed}/{self._expected_curves} curves = {completion_ratio:.1%}, need {min_completion:.0%})")
            self.api_client.abandon_residue(self.ctx.client_id, self.current_residue_id)
        else:
            # Completed enough curves or found a factor - mark as complete
            print("Completing residue work...")
            try:
                complete_result = self.api_client.complete_residue(
                    client_id=self.ctx.client_id,
                    residue_id=self.current_residue_id,
                    stage2_attempt_id=self._stage2_attempt_id
                )

                if complete_result:
                    new_t_level = complete_result.get('new_t_level')
                    if new_t_level is not None:
                        print(f"T-level updated to {new_t_level:.2f}")
                else:
                    self.logger.warning("Failed to complete residue on server - queuing for retry")
                    self.wrapper.submission_queue.enqueue_residue_completion(
                        residue_id=self.current_residue_id,
                        client_id=self.ctx.client_id,
                        stage2_attempt_id=self._stage2_attempt_id
                    )
            except ResourceNotFoundError:
                self.logger.warning(f"Residue {self.current_residue_id} already expired/completed on server, skipping")

        # Clean up local residue file
        if self.local_residue_file and self.local_residue_file.exists():
            self.local_residue_file.unlink()
            self.logger.info(f"Deleted local residue file: {self.local_residue_file}")

    def on_work_completed(self, work: Dict[str, Any], result: FactorResult) -> None:
        self.current_residue_id = None
        self.local_residue_file = None
        self._primary_submission_failed = False
        super().on_work_completed(work, result)

    def cleanup_on_failure(self, work: Optional[Dict[str, Any]], error: BaseException) -> None:
        if self.current_residue_id:
            self.api_client.abandon_residue(self.ctx.client_id, self.current_residue_id)
            self.current_residue_id = None

        if self.local_residue_file and self.local_residue_file.exists():
            self.local_residue_file.unlink()
            self.local_residue_file = None

    def cleanup_on_shutdown(self) -> None:
        if self.local_residue_file and self.local_residue_file.exists():
            self.local_residue_file.unlink()

    def _handle_keyboard_interrupt(self) -> None:
        """Override to handle residue-specific cleanup."""
        self.cleanup_on_shutdown()
        handle_shutdown(
            wrapper=self.wrapper,
            current_work_id=None,
            current_residue_id=self.current_residue_id,
            mode_name=self.mode_name,
            completed_count=self.completed_count,
            local_residue_file=self.local_residue_file
        )


class P1WorkMode(WorkMode):
    """
    P-1/P+1 sweep mode: Run PM1 and/or PP1 across composites from server.

    Uses the /ecm-work endpoint (same as standard ECM), calculates B1 one step
    above the composite's target t-level, and sweeps across composites (one
    composite per work assignment).

    Supports three flags (mutually exclusive):
    - --pm1: Run P-1 only (1 curve per composite)
    - --pp1: Run P+1 only (N curves per composite, default 3)
    - --p1:  Run P-1 (1 curve) + P+1 (N curves) per composite

    B1 calculation: One step above the target t-level in the optimal B1 table,
    capped by config pm1_b1/pp1_b1.
    B2: Omitted (let GMP-ECM use its default ratio).
    """

    mode_name = "P-1/P+1 Sweep"

    def __init__(self, ctx: WorkLoopContext):
        super().__init__(ctx)

        # Determine which methods to run
        self._run_pm1 = getattr(self.args, 'pm1', False) or getattr(self.args, 'p1', False)
        self._run_pp1 = getattr(self.args, 'pp1', False) or getattr(self.args, 'p1', False)
        self._pp1_curves = getattr(self.args, 'pp1_curves', 3)

        # Set human-readable mode name
        if self._run_pm1 and self._run_pp1:
            self.mode_name = "P-1/P+1 Sweep"
        elif self._run_pm1:
            self.mode_name = "P-1 Sweep"
        else:
            self.mode_name = "P+1 Sweep"

        # Get B1 caps from typed config (handles scientific notation safely)
        gmp = self.wrapper.typed_config.programs.gmp_ecm
        self._pm1_b1_cap = gmp.pm1_b1
        self._pp1_b1_cap = gmp.pp1_b1

        # Per-assignment state
        self._pm1_result: Optional[FactorResult] = None
        self._pp1_result: Optional[FactorResult] = None
        self._pm1_b1: int = 0
        self._pp1_b1: int = 0

    def request_work(self) -> Optional[Dict[str, Any]]:
        return request_p1_work(
            self.api_client,
            self.ctx.client_id,
            self.args,
            self.logger
        )

    def on_work_started(self, work: Dict[str, Any]) -> None:
        super().on_work_started(work)

        # Reset per-assignment state
        self._pm1_result = None
        self._pp1_result = None

        # Read B1 from server response, cap by config values
        server_pm1_b1 = work.get('pm1_b1') or 0
        server_pp1_b1 = work.get('pp1_b1') or 0

        self._pm1_b1 = min(server_pm1_b1, self._pm1_b1_cap) if self._run_pm1 and server_pm1_b1 else 0
        self._pp1_b1 = min(server_pp1_b1, self._pp1_b1_cap) if self._run_pp1 and server_pp1_b1 else 0

        # Build params display
        params: Dict[str, Any] = {
            'T-level': f"{work.get('current_t_level', 0):.1f} -> {work.get('target_t_level', 0):.1f}",
        }
        methods = []
        if self._run_pm1:
            params['PM1 B1'] = self._pm1_b1
            methods.append("P-1 (1 curve)")
        if self._run_pp1:
            params['PP1 B1'] = self._pp1_b1
            methods.append(f"P+1 ({self._pp1_curves} curves)")
        params['Methods'] = ' + '.join(methods)

        print_work_header(
            work_id=self.current_work_id,
            composite=work['composite'],
            digit_length=work['digit_length'],
            params=params
        )

    def execute_work(self, work: Dict[str, Any]) -> FactorResult:
        composite = work['composite']
        combined_result = FactorResult()
        combined_result.success = True
        factor_found = False

        # Run PM1 if applicable
        if self._run_pm1 and not factor_found:
            print(f"Running P-1 (B1={self._pm1_b1}, B2=GMP-ECM default, 1 curve)...")
            pm1_config = ECMConfig(
                composite=composite,
                b1=self._pm1_b1,
                b2=None,  # Let GMP-ECM use default
                curves=1,
                method='pm1',
                parametrization=1,
                verbose=self.args.verbose,
                progress_interval=getattr(self.args, 'progress_interval', 0),
            )
            self._pm1_result = self.wrapper.run_ecm_v2(pm1_config)
            combined_result.curves_run += self._pm1_result.curves_run
            combined_result.execution_time += self._pm1_result.execution_time

            if self._pm1_result.factors:
                for f, s in self._pm1_result.factor_sigma_pairs:
                    combined_result.add_factor(f, s)
                factor_found = True
                print(f"Factor found by P-1: {self._pm1_result.factors[0]}")

        # Run PP1 if applicable and no factor found yet
        if self._run_pp1 and not factor_found:
            print(f"Running P+1 (B1={self._pp1_b1}, B2=GMP-ECM default, {self._pp1_curves} curves)...")
            pp1_config = ECMConfig(
                composite=composite,
                b1=self._pp1_b1,
                b2=None,  # Let GMP-ECM use default
                curves=self._pp1_curves,
                method='pp1',
                parametrization=1,
                verbose=self.args.verbose,
                progress_interval=getattr(self.args, 'progress_interval', 0),
            )
            self._pp1_result = self.wrapper.run_ecm_v2(pp1_config)
            combined_result.curves_run += self._pp1_result.curves_run
            combined_result.execution_time += self._pp1_result.execution_time

            if self._pp1_result.factors:
                for f, s in self._pp1_result.factor_sigma_pairs:
                    combined_result.add_factor(f, s)
                print(f"Factor found by P+1: {self._pp1_result.factors[0]}")

        return combined_result

    def submit_results(self, work: Dict[str, Any], result: FactorResult) -> bool:
        composite = work['composite']

        # Check that at least one method actually ran curves
        pm1_curves = self._pm1_result.curves_run if self._pm1_result else 0
        pp1_curves = self._pp1_result.curves_run if self._pp1_result else 0
        if pm1_curves == 0 and pp1_curves == 0:
            self.logger.error("Zero curves completed for P-1/P+1, execution may have failed (check ECM binary path)")
            return False

        success = True

        # Submit PM1 results if we ran PM1
        if self._pm1_result and self._pm1_result.curves_run > 0:
            pm1_dict = self._pm1_result.to_dict(composite, 'pm1')
            pm1_dict['b1'] = self._pm1_b1
            pm1_dict['b2'] = None  # Used GMP-ECM default
            pm1_dict['curves_requested'] = 1
            pm1_dict['parametrization'] = 1
            pm1_dict['work_id'] = self.current_work_id

            submit_response = self.wrapper.submit_result(
                pm1_dict, self.args.project, 'gmp-ecm-pm1'
            )
            if not submit_response:
                self.logger.error("Failed to submit PM1 results")
                success = False

        # Submit PP1 results if we ran PP1
        if self._pp1_result and self._pp1_result.curves_run > 0:
            pp1_dict = self._pp1_result.to_dict(composite, 'pp1')
            pp1_dict['b1'] = self._pp1_b1
            pp1_dict['b2'] = None  # Used GMP-ECM default
            pp1_dict['curves_requested'] = self._pp1_curves
            pp1_dict['parametrization'] = 1
            pp1_dict['work_id'] = self.current_work_id

            submit_response = self.wrapper.submit_result(
                pp1_dict, self.args.project, 'gmp-ecm-pp1'
            )
            if not submit_response:
                self.logger.error("Failed to submit PP1 results")
                success = False

        return success

    def complete_work(self, work: Dict[str, Any]) -> None:
        assert self.current_work_id is not None
        try:
            if not self.api_client.complete_work(self.current_work_id, self.ctx.client_id):
                self.wrapper.submission_queue.enqueue_work_completion(
                    self.current_work_id, self.ctx.client_id
                )
        except ResourceNotFoundError:
            self.logger.warning(f"Work {self.current_work_id} already expired/completed on server, skipping")



class StandardAutoWorkMode(WorkMode):
    """
    Standard auto-work mode: T-level or B1/B2 based execution.

    This mode:
    1. Requests ECM work from server
    2. Executes using t-level mode (default) or B1/B2 mode
    3. Submits results (t-level mode submits after each batch)
    4. Completes work assignment
    """

    mode_name = "Auto-work"

    def request_work(self) -> Optional[Dict[str, Any]]:
        return request_ecm_work(
            self.api_client,
            self.ctx.client_id,
            self.args,
            self.logger
        )

    def on_work_started(self, work: Dict[str, Any]) -> None:
        super().on_work_started(work)

        print_work_header(
            work_id=self.current_work_id,
            composite=work['composite'],
            digit_length=work['digit_length'],
            params={
                'T-level': f"{work.get('current_t_level', 0):.1f} -> {work.get('target_t_level', 0):.1f}"
            }
        )

    def execute_work(self, work: Dict[str, Any]) -> FactorResult:
        composite = work['composite']

        has_b1_b2 = self.args.b1 is not None and self.args.b2 is not None
        has_client_tlevel = hasattr(self.args, 'tlevel') and self.args.tlevel is not None

        # Determine execution mode
        if has_client_tlevel or (not has_b1_b2 and not has_client_tlevel):
            return self._execute_tlevel_mode(work, composite, has_client_tlevel)
        else:
            return self._execute_b1b2_mode(work, composite)

    def _execute_tlevel_mode(self, work: Dict[str, Any], composite: str,
                             has_client_tlevel: bool) -> FactorResult:
        """Execute using progressive t-level targeting."""
        target_tlevel = self.args.tlevel if has_client_tlevel else work.get('target_t_level', 35.0)

        if hasattr(self.args, 'start_tlevel') and self.args.start_tlevel is not None:
            start_tlevel = self.args.start_tlevel
        else:
            start_tlevel = work.get('current_t_level', 0.0)

        mode_desc = "client t-level" if has_client_tlevel else "server t-level"
        print(f"Mode: {mode_desc} (start: {start_tlevel:.1f}, target: {target_tlevel:.1f})")

        # Use workers for multiprocess mode OR two-stage mode (for CPU stage 2)
        two_stage = getattr(self.args, 'two_stage', False)
        if self.args.multiprocess or two_stage:
            workers = resolve_worker_count(self.args, self.wrapper.config)
        else:
            workers = 1

        # Resolve max_batch from args or config
        max_batch = getattr(self.args, 'max_batch', None) or get_max_batch_default(self.wrapper.config)

        config = TLevelConfig(
            composite=composite,
            target_t_level=target_tlevel,
            start_t_level=start_tlevel,
            threads=workers,
            verbose=self.args.verbose,
            progress_interval=getattr(self.args, 'progress_interval', 0),
            max_batch_curves=max_batch,
            use_two_stage=getattr(self.args, 'two_stage', False),
            b2_multiplier=getattr(self.args, 'b2_multiplier', None) or 100.0,
            project=self.args.project,
            no_submit=False,
            work_id=self.current_work_id
        )

        result = self.wrapper.run_tlevel_v2(config)

        # Store for submit_results - t-level mode submits internally
        self._is_tlevel_mode = True
        self._results_dict = result.to_dict(composite, self.args.method)

        return result

    def _execute_b1b2_mode(self, work: Dict[str, Any], composite: str) -> FactorResult:
        """Execute using explicit B1/B2 parameters."""
        b1 = self.args.b1
        b2 = 0
        k = 0
        if 'b2_from_dict' in work:
            b2 = work['b2_from_dict']
            if 'k_from_dict' in work:
                k = work['k_from_dict']
        else:
            b2 = self.args.b2
        curves = self.args.curves if self.args.curves else \
                 (1 if self.args.two_stage else self.wrapper.config['programs']['gmp_ecm']['default_curves'])

        use_gpu, _, _ = resolve_gpu_settings(self.args, self.wrapper.config)
        sigma = parse_sigma_arg(self.args)
        param = resolve_param(self.args, use_gpu)

        self._is_tlevel_mode = False
        result: FactorResult

        if self.args.two_stage and self.args.method == 'ecm':
            workers = resolve_worker_count(self.args, self.wrapper.config)
            print(f"Mode: two-stage GPU+CPU (B1={b1}, B2={b2}, curves={curves}, workers={workers})")

            two_stage_config = TwoStageConfig(
                composite=composite,
                b1=b1,
                b2=b2,
                stage1_curves=curves,
                stage1_device="GPU" if use_gpu else "CPU",
                stage2_device="CPU",
                stage1_parametrization=param if param else 3,
                threads=workers,
                verbose=self.args.verbose,
                progress_interval=getattr(self.args, 'progress_interval', 0)
            )
            result = self.wrapper.run_two_stage_v2(two_stage_config)

        elif self.args.multiprocess:
            workers = resolve_worker_count(self.args, self.wrapper.config)
            print(f"Mode: multiprocess (B1={b1}, B2={b2}, curves={curves}, workers={workers})")

            mp_config = MultiprocessConfig(
                composite=composite,
                b1=b1,
                b2=b2,
                total_curves=curves,
                num_processes=workers,
                parametrization=param if param else 3,
                method=self.args.method,
                verbose=self.args.verbose,
                progress_interval=getattr(self.args, 'progress_interval', 0)
            )
            result = self.wrapper.run_multiprocess_v2(mp_config)

        else:
            print(f"Mode: standard (B1={b1}, B2={b2}, curves={curves})")

            # sigma can be str or int from parse_sigma_arg, ECMConfig accepts both
            ecm_config = ECMConfig(
                composite=composite,
                b1=b1,
                b2=b2,
                curves=curves,
                sigma=int(sigma) if sigma and str(sigma).isdigit() else None,
                parametrization=param if param else 3,
                method=self.args.method,
                verbose=self.args.verbose,
                progress_interval=getattr(self.args, 'progress_interval', 0)
            )
            result = self.wrapper.run_ecm_v2(ecm_config)

        self._results_dict = result.to_dict(composite, self.args.method)

        # Add ECM parameters that aren't in FactorResult
        self._results_dict['b1'] = self.args.b1
        self._results_dict['b2'] = b2
        self._results_dict['curves_requested'] = self.args.curves
        use_gpu, _, _ = resolve_gpu_settings(self.args, self.wrapper.config)
        self._results_dict['parametrization'] = self.args.param or (3 if use_gpu else 1)

        return result

    def submit_results(self, work: Dict[str, Any], result: FactorResult) -> bool:
        # T-level mode submits internally after each batch
        if self._is_tlevel_mode:
            if result.curves_run == 0:
                self.logger.error("T-level mode ran zero curves, execution may have failed")
                return False
            return True

        # Reject if no curves were actually run (e.g. binary not found)
        if result.curves_run == 0:
            self.logger.error("Zero curves completed, execution may have failed (check ECM binary path)")
            return False

        # B1/B2 modes need to submit here
        self._results_dict['work_id'] = self.current_work_id
        program_name = f"gmp-ecm-{self._results_dict.get('method', 'ecm')}"
        submit_response = self.wrapper.submit_result(
            self._results_dict,
            self.args.project,
            program_name
        )

        if not submit_response:
            self.logger.error("Failed to submit results, abandoning work assignment")
            return False

        return True

    def complete_work(self, work: Dict[str, Any]) -> None:
        assert self.current_work_id is not None  # Set in on_work_started
        try:
            if not self.api_client.complete_work(self.current_work_id, self.ctx.client_id):
                self.wrapper.submission_queue.enqueue_work_completion(
                    self.current_work_id, self.ctx.client_id
                )
        except ResourceNotFoundError:
            self.logger.warning(f"Work {self.current_work_id} already expired/completed on server, skipping")


class CompositeTargetMode(StandardAutoWorkMode):
    """
    Target a specific composite by querying the server for its t-level status.

    This mode:
    1. Queries the server for the composite's current/target t-level
    2. Runs ECM using t-level mode to make optimal progress
    3. Submits results to server
    4. Does NOT create/complete work assignments (just t-level work)

    The composite must already exist in the server database.
    """

    mode_name = "Composite Target"

    def __init__(self, ctx: WorkLoopContext):
        super().__init__(ctx)
        self.target_composite = ctx.args.composite
        self._work_done = False

    def request_work(self) -> Optional[Dict[str, Any]]:
        """Query composite status from server and build work dict."""
        # Only run once
        if self._work_done:
            return None

        self._work_done = True

        # Query composite status from server
        status = self.api_client.get_composite_status(self.target_composite)

        if status is None:
            print(f"Error: Composite not found in server database")
            print(f"  {self.target_composite[:50]}...")
            print(f"\nTo add this composite, use the server admin interface or API.")
            return None

        # Check if already fully factored
        if status.get('status') == 'fully_factored':
            print(f"Composite is already fully factored!")
            print(f"  Factors: {status.get('factors_found', [])}")
            return None

        if status.get('status') == 'prime':
            print(f"Number is prime, no factorization needed.")
            return None

        # Check if t-level target is reached
        current_t = status.get('current_t_level', 0) or 0
        target_t = status.get('target_t_level', 0) or 0

        if target_t > 0 and current_t >= target_t:
            print(f"Target t-level already reached: {current_t:.1f} >= {target_t:.1f}")
            print(f"\nTo continue, increase the target t-level on the server.")
            return None

        # Build work dict compatible with StandardAutoWorkMode
        # Use current_composite if available (partially factored), else original
        composite_to_factor = status.get('current_composite') or status.get('composite') or self.target_composite

        work = {
            'work_id': None,  # No formal work assignment
            'composite': composite_to_factor,
            'composite_id': None,  # Not needed for submission
            'digit_length': status.get('digit_length') or len(composite_to_factor),
            'current_t_level': current_t,
            'target_t_level': target_t if target_t > 0 else 35.0,  # Default t35 if not set
        }

        return work

    def on_work_started(self, work: Dict[str, Any]) -> None:
        """Log work start without setting work_id."""
        # Don't call super() - we don't have a real work_id
        self.current_work_id = None

        print_work_header(
            work_id="(direct)",
            composite=work['composite'],
            digit_length=work['digit_length'],
            params={
                'T-level': f"{work.get('current_t_level', 0):.1f} -> {work.get('target_t_level', 0):.1f}"
            }
        )

    def complete_work(self, work: Dict[str, Any]) -> None:
        """No work assignment to complete in composite target mode."""
        # Results are already submitted via t-level mode's internal submission
        pass


def get_work_mode(ctx: WorkLoopContext) -> WorkMode:
    """
    Factory function to create the appropriate WorkMode based on args.

    Args:
        ctx: Work loop context with wrapper, client_id, and args

    Returns:
        Appropriate WorkMode subclass instance
    """
    args = ctx.args

    if getattr(args, 'composite', None):
        return CompositeTargetMode(ctx)
    elif getattr(args, 'pm1', False) or getattr(args, 'pp1', False) or getattr(args, 'p1', False):
        return P1WorkMode(ctx)
    elif getattr(args, 'stage1_only', False):
        return Stage1ProducerMode(ctx)
    elif getattr(args, 'stage2_only', False):
        return Stage2ConsumerMode(ctx)
    else:
        return StandardAutoWorkMode(ctx)
