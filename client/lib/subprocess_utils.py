"""
Unified subprocess execution utilities for factorization wrappers.

This module provides a single, well-tested subprocess execution function
that handles various output modes (verbose streaming, silent capture,
progress updates, etc.) to eliminate code duplication across wrappers.
"""

import subprocess
import logging
from typing import List, Optional, Callable, Dict, Any, Tuple


logger = logging.getLogger(__name__)


def execute_subprocess(
    cmd: List[str],
    composite: Optional[str] = None,
    verbose: bool = False,
    progress_interval: int = 0,
    line_callback: Optional[Callable[[str, List[str]], None]] = None,
    timeout: Optional[int] = None,
    log_prefix: str = "",
    stop_event: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Unified subprocess execution with flexible output handling.

    This function handles all common subprocess execution patterns:
    - Verbose mode: stream output to console in real-time
    - Silent mode: capture all output without printing
    - Progress mode: show periodic updates every N lines
    - Callback mode: call function for each output line
    - Early termination: check stop_event for coordinated shutdown

    Args:
        cmd: Command and arguments to execute
        composite: Optional composite number to send to stdin
        verbose: If True, stream output to console in real-time
        progress_interval: If > 0, show progress every N lines (overrides verbose)
        line_callback: Optional function called for each line: callback(line, all_lines_so_far)
        timeout: Optional timeout in seconds
        log_prefix: Prefix for log messages (e.g., "Worker 1", "Stage1")
        stop_event: Optional multiprocessing.Event for early termination

    Returns:
        Dictionary with:
        - stdout: Complete output as string
        - output_lines: List of output lines
        - returncode: Process exit code
        - terminated_early: True if stopped via stop_event

    Example:
        >>> result = execute_subprocess(
        ...     ["ecm", "1000000"],
        ...     composite="123456789",
        ...     verbose=True,
        ...     timeout=3600
        ... )
        >>> print(result['stdout'])
    """
    terminated_early = False
    output_lines = []
    line_count = 0

    try:
        # Determine stdin mode: PIPE if we're sending input, otherwise None
        stdin_mode = subprocess.PIPE if composite else None

        process = subprocess.Popen(
            cmd,
            stdin=stdin_mode,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )

        # Send composite to stdin if provided
        if composite and process.stdin:
            process.stdin.write(composite)
            process.stdin.close()

        # Stream output line by line
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                # Check for early termination signal
                if stop_event and stop_event.is_set():
                    process.terminate()
                    terminated_early = True
                    if verbose or log_prefix:
                        msg = f"{log_prefix}: Terminated early due to stop event" if log_prefix else "Terminated early due to stop event"
                        if logger:
                            logger.info(msg)
                        else:
                            print(msg)
                    break

                line = line.rstrip()
                if line:
                    output_lines.append(line)
                    line_count += 1

                    # Handle different output modes
                    if progress_interval > 0 and line_count % progress_interval == 0:
                        # Progress update mode
                        msg = f"{log_prefix}: Processed {line_count} lines" if log_prefix else f"Processed {line_count} lines"
                        if logger:
                            logger.info(msg)
                        else:
                            print(msg)
                    elif verbose:
                        # Verbose streaming mode
                        display_line = f"{log_prefix}: {line}" if log_prefix else line
                        print(display_line)

                    # Call line callback if provided
                    if line_callback:
                        line_callback(line, output_lines)

        # Wait for process to complete
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise

        stdout = '\n'.join(output_lines)

        return {
            'stdout': stdout,
            'output_lines': output_lines,
            'returncode': process.returncode,
            'terminated_early': terminated_early
        }

    except subprocess.TimeoutExpired as e:
        if logger:
            logger.error(f"Process timed out after {timeout} seconds")
        raise
    except Exception as e:
        if logger:
            logger.error(f"Subprocess execution failed: {e}")
        raise


def execute_subprocess_simple(
    cmd: List[str],
    input_text: Optional[str] = None,
    timeout: Optional[int] = None
) -> Tuple[str, int]:
    """
    Simple subprocess execution for quick commands (like t-level calculations).

    Args:
        cmd: Command and arguments to execute
        input_text: Optional text to send to stdin
        timeout: Optional timeout in seconds

    Returns:
        Tuple of (stdout, returncode)

    Example:
        >>> output, code = execute_subprocess_simple(["t-level", "-q", "100@50000"])
        >>> print(output)
        t25.4
    """
    result = subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False
    )
    return result.stdout, result.returncode


def create_subprocess_silent(
    cmd: List[str],
    composite: Optional[str] = None
) -> Tuple[subprocess.Popen, List[str]]:
    """
    Create and run subprocess, capturing output silently.

    This is for cases where you need the Popen object for manual handling
    but want silent output capture.

    Args:
        cmd: Command and arguments to execute
        composite: Optional composite number to send to stdin

    Returns:
        Tuple of (process, output_lines)

    Example:
        >>> process, lines = create_subprocess_silent(["ecm", "1000"], "12345")
        >>> process.wait()
        >>> print(lines)
    """
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if composite else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # Send composite to stdin if provided
    if composite and process.stdin:
        process.stdin.write(composite)
        process.stdin.close()

    # Capture all output
    output_lines = []
    if process.stdout:
        for line in iter(process.stdout.readline, ''):
            line = line.rstrip()
            if line:
                output_lines.append(line)

    return process, output_lines
