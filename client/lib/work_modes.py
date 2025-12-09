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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING
import argparse
import time

from .ecm_config import (
    ECMConfig, TwoStageConfig, MultiprocessConfig, TLevelConfig, FactorResult
)
from .work_helpers import print_work_header, print_work_status, request_ecm_work
from .stage1_helpers import submit_stage1_complete_workflow
from .error_helpers import check_work_limit_reached
from .cleanup_helpers import handle_shutdown
from .results_builder import results_for_stage1
from .arg_parser import resolve_gpu_settings, resolve_worker_count, get_stage2_workers_default
from .ecm_arg_helpers import parse_sigma_arg, resolve_param, resolve_stage2_workers

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

    def should_continue(self) -> bool:
        """
        Check if work loop should continue.

        Returns:
            True to continue, False to exit loop
        """
        # Check for interruption
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

        try:
            while self.should_continue():
                # Request work from server
                work = self.request_work()
                if not work:
                    continue

                # Track work assignment
                self.on_work_started(work)

                try:
                    # Execute factorization
                    result = self.execute_work(work)

                    # Check for interruption during execution
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

        except KeyboardInterrupt:
            self._handle_keyboard_interrupt()

        return self.completed_count

    def _print_startup_banner(self) -> None:
        """Print mode startup banner."""
        print("=" * 60)
        if self.ctx.work_count_limit:
            print(f"{self.mode_name} - will process {self.ctx.work_count_limit} assignment(s)")
        else:
            print(f"{self.mode_name} - requesting work from server")
            print("Press Ctrl+C to stop")
        print("=" * 60)
        print()

    def _handle_keyboard_interrupt(self) -> None:
        """Handle Ctrl+C gracefully."""
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
    """

    mode_name = "Stage 1 Producer (GPU)"

    def __init__(self, ctx: WorkLoopContext):
        super().__init__(ctx)
        self.residue_file: Optional[Path] = None

    def request_work(self) -> Optional[Dict[str, Any]]:
        return request_ecm_work(
            self.api_client,
            self.ctx.client_id,
            self.args,
            self.logger
        )

    def on_work_started(self, work: Dict[str, Any]) -> None:
        super().on_work_started(work)

        # Resolve GPU settings
        use_gpu, gpu_device, gpu_curves = resolve_gpu_settings(self.args, self.wrapper.config)

        # Determine curves
        curves = self.args.curves if self.args.curves is not None else \
                 self.wrapper.config['programs']['gmp_ecm']['default_curves']

        print_work_header(
            work_id=self.current_work_id,
            composite=work['composite'],
            digit_length=work['digit_length'],
            params={'B1': self.args.b1, 'curves': curves}
        )

    def execute_work(self, work: Dict[str, Any]) -> FactorResult:
        composite = work['composite']

        # Resolve GPU settings
        use_gpu, gpu_device, gpu_curves = resolve_gpu_settings(self.args, self.wrapper.config)

        # Determine curves
        curves = self.args.curves if self.args.curves is not None else \
                 self.wrapper.config['programs']['gmp_ecm']['default_curves']

        # Generate residue file path
        residue_dir = Path(self.wrapper.config['execution'].get('residue_dir', 'data/residues'))
        residue_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.residue_file = residue_dir / f"stage1_{timestamp}_{composite[:20]}.txt"

        # Resolve parameters
        sigma = parse_sigma_arg(self.args)
        param = resolve_param(self.args, use_gpu)

        print(f"Running ECM stage 1 (B1={self.args.b1}, curves={curves})...")
        print(f"Saving residues to: {self.residue_file}")

        # Run stage 1
        success, factor, actual_curves, raw_output, all_factors = self.wrapper._run_stage1(
            composite=composite,
            b1=self.args.b1,
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

        # Build results using ResultsBuilder
        builder = (results_for_stage1(composite, self.args.b1, self._last_curves, self._last_param)
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
        self.api_client.complete_work(self.current_work_id, self.ctx.client_id)

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
        self.local_residue_file: Optional[Path] = None
        # Import here to avoid circular dependency
        from .stage2_executor import Stage2Executor
        self.Stage2Executor = Stage2Executor

    def request_work(self) -> Optional[Dict[str, Any]]:
        residue_work = self.api_client.get_residue_work(
            client_id=self.ctx.client_id,
            min_digits=getattr(self.args, 'min_digits', None),
            max_digits=getattr(self.args, 'max_digits', None),
            min_priority=getattr(self.args, 'priority', None),
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

        b1 = work['b1']

        # Determine B2
        if self.args.b2 is not None:
            b2 = self.args.b2
        elif hasattr(self.args, 'b2_multiplier') and self.args.b2_multiplier is not None:
            b2 = int(b1 * self.args.b2_multiplier)
            print(f"Using dynamic B2 = B1 * {self.args.b2_multiplier} = {b2}")
        else:
            b2 = work.get('suggested_b2', b1 * 100)

        self._b2 = b2
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

        print(f"Downloaded {self.local_residue_file.stat().st_size} bytes")

        # Get stage2 workers
        stage2_workers = getattr(self.args, 'stage2_workers', None) or \
                        get_stage2_workers_default(self.wrapper.config)

        # Run stage 2
        print(f"Running stage 2 with {stage2_workers} workers...")
        executor = self.Stage2Executor(
            self.wrapper,
            self.local_residue_file,
            work['b1'],
            self._b2,
            stage2_workers,
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

        # Store for submit_results
        self._work = work
        self._factor = factor
        self._sigma = sigma

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
            'raw_output': f"Stage 2 from residue {self.current_residue_id}",
            'method': 'ecm',
            'parametrization': work.get('parametrization', 3),
            'execution_time': result.execution_time,
        }

        print("Submitting stage 2 results...")
        program_name = 'gmp-ecm-ecm'
        submit_response = self.wrapper.submit_result(results, self.args.project, program_name)

        if not submit_response:
            self.logger.error("Failed to submit stage 2 results")
            return False

        # Extract attempt_id
        stage2_attempt_id = submit_response.get('attempt_id')
        if not stage2_attempt_id:
            self.logger.error("No attempt_id returned from submit")
            return False

        print(f"Stage 2 attempt ID: {stage2_attempt_id}")
        self._stage2_attempt_id = stage2_attempt_id

        return True

    def complete_work(self, work: Dict[str, Any]) -> None:
        print("Completing residue work...")
        assert self.current_residue_id is not None  # Set in on_work_started
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
            self.logger.warning("Failed to complete residue on server")

        # Clean up local residue file
        if self.local_residue_file and self.local_residue_file.exists():
            self.local_residue_file.unlink()
            self.logger.info(f"Deleted local residue file: {self.local_residue_file}")

    def on_work_completed(self, work: Dict[str, Any], result: FactorResult) -> None:
        self.current_residue_id = None
        self.local_residue_file = None
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

        workers = resolve_worker_count(self.args) if self.args.multiprocess else 1

        config = TLevelConfig(
            composite=composite,
            target_t_level=target_tlevel,
            start_t_level=start_tlevel,
            threads=workers,
            verbose=self.args.verbose,
            progress_interval=getattr(self.args, 'progress_interval', 0),
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
        b2 = self.args.b2
        curves = self.args.curves if self.args.curves else \
                 (1 if self.args.two_stage else self.wrapper.config['programs']['gmp_ecm']['default_curves'])

        use_gpu, _, _ = resolve_gpu_settings(self.args, self.wrapper.config)
        sigma = parse_sigma_arg(self.args)
        param = resolve_param(self.args, use_gpu)

        self._is_tlevel_mode = False
        result: FactorResult

        if self.args.two_stage and self.args.method == 'ecm':
            print(f"Mode: two-stage GPU+CPU (B1={b1}, B2={b2}, curves={curves})")
            stage2_workers = resolve_stage2_workers(self.args, self.wrapper.config)

            two_stage_config = TwoStageConfig(
                composite=composite,
                b1=b1,
                b2=b2,
                stage1_curves=curves,
                stage1_device="GPU" if use_gpu else "CPU",
                stage2_device="CPU",
                stage1_parametrization=param if param else 3,
                threads=stage2_workers,
                verbose=self.args.verbose,
                progress_interval=getattr(self.args, 'progress_interval', 0)
            )
            result = self.wrapper.run_two_stage_v2(two_stage_config)

        elif self.args.multiprocess:
            workers = resolve_worker_count(self.args)
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
        self._results_dict['b2'] = self.args.b2
        self._results_dict['curves_requested'] = self.args.curves
        use_gpu, _, _ = resolve_gpu_settings(self.args, self.wrapper.config)
        self._results_dict['parametrization'] = self.args.param or (3 if use_gpu else 1)

        return result

    def submit_results(self, work: Dict[str, Any], result: FactorResult) -> bool:
        # T-level mode submits internally after each batch
        if self._is_tlevel_mode:
            return True

        # B1/B2 modes need to submit here
        if result.curves_run > 0:
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
        self.api_client.complete_work(self.current_work_id, self.ctx.client_id)


def get_work_mode(ctx: WorkLoopContext) -> WorkMode:
    """
    Factory function to create the appropriate WorkMode based on args.

    Args:
        ctx: Work loop context with wrapper, client_id, and args

    Returns:
        Appropriate WorkMode subclass instance
    """
    args = ctx.args

    if getattr(args, 'stage1_only', False):
        return Stage1ProducerMode(ctx)
    elif getattr(args, 'stage2_only', False):
        return Stage2ConsumerMode(ctx)
    else:
        return StandardAutoWorkMode(ctx)
