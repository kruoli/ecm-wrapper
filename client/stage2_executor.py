#!/usr/bin/env python3
"""
Stage2Executor - Unified Stage 2 execution with worker pool management.

This class eliminates ~270 lines of duplicated Stage 2 logic across:
- ECMWrapper._run_stage2_multithread()
- run_batch_pipeline.py cpu_worker() function

Handles:
- Residue file splitting
- Multi-threaded worker pool execution
- Early termination when factor found
- Progress tracking and reporting
- Process cleanup
"""
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Tuple, List, Callable, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed

from parsing_utils import parse_ecm_output

if TYPE_CHECKING:
    from ecm_wrapper import ECMWrapper


class Stage2Executor:
    """Manages Stage 2 execution with configurable worker pools."""

    def __init__(self, wrapper: 'ECMWrapper', residue_file: Path, b1: int, b2: int,
                 workers: int, verbose: bool = False):
        """
        Initialize Stage 2 executor.

        Args:
            wrapper: ECMWrapper instance (for config and residue manager)
            residue_file: Path to residue file from Stage 1
            b1: B1 parameter (will be extracted from residue file if available)
            b2: B2 parameter for Stage 2
            workers: Number of parallel workers
            verbose: Enable verbose output
        """
        self.wrapper = wrapper
        self.residue_file = residue_file
        self.b1 = b1
        self.b2 = b2
        self.workers = workers
        self.verbose = verbose
        self.logger = wrapper.logger
        self.ecm_path = wrapper.config['programs']['gmp_ecm']['path']

        # Shared state for worker coordination
        self.factor_found: Optional[Tuple[str, str]] = None
        self.factor_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.running_processes: List[subprocess.Popen] = []
        self.process_lock = threading.Lock()

    def execute(self, early_termination: bool = True,
                progress_interval: int = 0) -> Optional[Tuple[str, str]]:
        """
        Execute Stage 2 with worker pool.

        Args:
            early_termination: If True, stop all workers when first factor found
            progress_interval: Report progress every N curves (0 = no progress reporting)

        Returns:
            Tuple of (factor, sigma) or None if no factor found
        """
        # Extract B1 from residue file to ensure consistency
        residue_info = self.wrapper._parse_residue_file(self.residue_file)
        actual_b1 = residue_info['b1']
        if actual_b1 > 0 and actual_b1 != self.b1:
            self.logger.info(f"Using B1={actual_b1} from residue file (overriding parameter B1={self.b1})")
        b1_to_use = actual_b1 if actual_b1 > 0 else self.b1

        # Split residue file into chunks for workers
        residue_chunks = self._split_residue_file()

        if not residue_chunks:
            self.logger.error("Failed to split residue file")
            return None

        # Run workers in parallel
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            for i, chunk_file in enumerate(residue_chunks):
                future = executor.submit(
                    self._worker_stage2, chunk_file, i + 1, b1_to_use,
                    early_termination, progress_interval
                )
                futures.append(future)

            # Wait for completion or first factor
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        self.factor_found = result  # This is (factor, sigma) tuple
                        self.stop_event.set()  # Ensure all workers are signaled to stop
                        break
                except Exception as e:
                    self.logger.error(f"Worker thread error: {e}")

            # Ensure all remaining processes are terminated
            with self.process_lock:
                for process in self.running_processes:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()

        # Cleanup temporary chunk files and directory
        self._cleanup_chunks(residue_chunks)

        return self.factor_found

    def _split_residue_file(self) -> List[Path]:
        """Split residue file into chunks for parallel processing."""
        import tempfile
        chunk_dir = tempfile.mkdtemp(prefix="ecm_chunks_")
        self.logger.debug(f"Creating chunks in temporary directory: {chunk_dir}")

        # Use ResidueFileManager to split the file
        chunk_paths = self.wrapper.residue_manager.split_into_chunks(
            str(self.residue_file), self.workers, chunk_dir
        )

        # Convert string paths to Path objects
        return [Path(p) for p in chunk_paths]

    def _worker_stage2(self, chunk_file: Path, worker_id: int, b1: int,
                       early_termination: bool, progress_interval: int) -> Optional[Tuple[str, str]]:
        """Worker function for Stage 2 processing."""
        cmd = [self.ecm_path, '-resume', str(chunk_file)]
        if self.verbose:
            cmd.append('-v')
        cmd.extend([str(b1), str(self.b2)])

        # Count total lines in this worker's chunk for progress reporting
        total_lines = 0
        if self.verbose:
            try:
                with open(chunk_file, 'r') as f:
                    total_lines = sum(1 for _ in f)
            except:
                total_lines = 0

        try:
            self.logger.info(f"Worker {worker_id} starting Stage 2" +
                           (f" ({total_lines} curves)" if self.verbose and total_lines > 0 else ""))
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            # Register process for potential termination
            with self.process_lock:
                self.running_processes.append(process)

            # Stream output for progress tracking if progress_interval is set
            if progress_interval > 0:
                full_output = self._stream_worker_output(
                    process, worker_id, total_lines, progress_interval, early_termination
                )
            else:
                # Original behavior - get all output at once
                stdout, stderr = process.communicate()
                full_output = stdout if stdout else ""

                # Check if we should terminate early due to factor found elsewhere
                if early_termination and self.stop_event.is_set():
                    self.logger.info(f"Worker {worker_id} terminating due to factor found elsewhere")
                    return None

            # Count curve completions from output
            curves_completed = full_output.count("Step 2 took")

            # Progress reporting in verbose mode
            if self.verbose and total_lines > 0:
                percentage = (curves_completed / total_lines) * 100
                self.logger.info(f"Worker {worker_id} progress: {curves_completed}/{total_lines} curves - {percentage:.1f}% complete")

            # Save raw output to file for debugging
            self._save_worker_output(worker_id, full_output)

            # Check for factor - enable debug mode if worker stopped early
            enable_debug = self._should_enable_debug(curves_completed, total_lines)
            factor, sigma_from_output = parse_ecm_output(full_output, debug=enable_debug)

            if factor:
                # Ensure sigma is not None for type safety
                sigma_value = sigma_from_output if sigma_from_output is not None else ""
                with self.factor_lock:
                    if not self.factor_found:  # First factor wins
                        self.factor_found = (factor, sigma_value)
                        if early_termination:
                            self.stop_event.set()  # Signal other workers to stop
                        self.logger.info(f"Worker {worker_id} found factor: {factor} (sigma: {sigma_value})")
                        # Kill other processes if early termination enabled
                        if early_termination:
                            self._terminate_other_processes(process)
                return (factor, sigma_value)

            # If no factor found, report completion with diagnostic
            output_size = len(full_output)
            self.logger.info(f"Worker {worker_id} completed (no factor) - {curves_completed} curves, {output_size} bytes output")

            # Show diagnostic if worker stopped early
            if self._should_enable_debug(curves_completed, total_lines):
                if total_lines > 0:
                    self.logger.warning(f"Worker {worker_id} stopped early at {curves_completed}/{total_lines} curves")
                else:
                    self.logger.warning(f"Worker {worker_id} stopped early at {curves_completed} curves (expected ~384)")
                self.logger.debug(f"Worker {worker_id} output preview (first 500 chars):\n{full_output[:500]}")
                self.logger.debug(f"Worker {worker_id} output preview (last 500 chars):\n{full_output[-500:]}")

            return None

        except Exception as e:
            self.logger.error(f"Worker {worker_id} failed: {e}")
            return None
        finally:
            # Remove process from tracking
            with self.process_lock:
                if process in self.running_processes:
                    self.running_processes.remove(process)

    def _stream_worker_output(self, process: subprocess.Popen, worker_id: int,
                              total_lines: int, progress_interval: int,
                              early_termination: bool) -> str:
        """Stream output from worker with progress tracking."""
        full_output = ""
        last_progress_report = 0

        if not process.stdout:
            return full_output

        while True:
            line = process.stdout.readline()
            if not line:
                break

            full_output += line

            # Check if we should terminate early
            if early_termination and self.stop_event.is_set():
                process.terminate()
                self.logger.info(f"Worker {worker_id} terminating due to factor found elsewhere")
                break

            # Check for curve completion and progress reporting
            if "Step 2 took" in line:
                curves_completed = full_output.count("Step 2 took")

                # Report progress at intervals
                if curves_completed - last_progress_report >= progress_interval:
                    if total_lines > 0:
                        percentage = (curves_completed / total_lines) * 100
                        self.logger.info(f"Worker {worker_id}: {curves_completed}/{total_lines} curves ({percentage:.1f}%)")
                    else:
                        self.logger.info(f"Worker {worker_id}: {curves_completed} curves completed")
                    last_progress_report = curves_completed

        process.wait()

        # CRITICAL: Drain any remaining buffered output after process exits
        if process.stdout:
            remaining = process.stdout.read()
            if remaining:
                full_output += remaining
                self.logger.debug(f"Worker {worker_id}: Drained {len(remaining)} chars from buffer after process exit")

        return full_output

    def _terminate_other_processes(self, current_process: subprocess.Popen):
        """Terminate all processes except the current one."""
        with self.process_lock:
            processes_to_terminate = [p for p in self.running_processes
                                     if p != current_process and p.poll() is None]
            for p in processes_to_terminate:
                p.terminate()

    def _should_enable_debug(self, curves_completed: int, total_lines: int) -> bool:
        """Determine if debug output should be enabled based on completion rate."""
        if total_lines > 0:
            return curves_completed < total_lines * 0.9
        else:
            # If we don't know total_lines, assume each chunk should do ~300+ curves
            return curves_completed < 300

    def _save_worker_output(self, worker_id: int, output: str):
        """Save worker output to file for debugging."""
        try:
            import tempfile
            output_dir = Path(tempfile.gettempdir()) / "ecm_stage2_logs"
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"worker_{worker_id}_{int(time.time())}.log"
            with open(output_file, 'w') as f:
                f.write(output)
            self.logger.debug(f"Worker {worker_id} output saved to: {output_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save worker {worker_id} output: {e}")

    def _cleanup_chunks(self, residue_chunks: List[Path]):
        """Cleanup temporary chunk files and directories."""
        chunk_dirs_to_cleanup = set()
        for chunk_file in residue_chunks:
            try:
                chunk_dirs_to_cleanup.add(chunk_file.parent)
                chunk_file.unlink()
            except:
                pass

        # Clean up temporary chunk directories
        for chunk_dir in chunk_dirs_to_cleanup:
            try:
                import shutil
                shutil.rmtree(chunk_dir)
                self.logger.debug(f"Cleaned up chunk directory: {chunk_dir}")
            except:
                pass
