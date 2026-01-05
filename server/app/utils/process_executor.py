"""
External Program Executor Utility

Provides a unified interface for executing external programs with consistent
error handling, timeout management, and output parsing.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional, Union, Tuple, List

logger = logging.getLogger(__name__)


class ExternalProgramExecutor:
    """
    Execute external programs with consistent error handling and output parsing.

    This utility consolidates subprocess execution patterns used across the codebase,
    particularly for external binaries like t-level and PARI/GP.
    """

    def __init__(self, binary_path: Union[str, Path], binary_name: str = "program"):
        """
        Initialize executor for a specific binary.

        Args:
            binary_path: Path to the binary executable
            binary_name: Human-readable name for logging (e.g., "PARI/GP", "t-level")
        """
        self.binary_path = str(binary_path)
        self.binary_name = binary_name
        self.logger = logging.getLogger(f"{__name__}.{binary_name}")

    def execute(
        self,
        args: List[str],
        input_data: Optional[str] = None,
        timeout: int = 30,
        check_returncode: bool = True
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Execute the external program with arguments.

        Args:
            args: Command-line arguments to pass to the binary
            input_data: Optional string to send to stdin
            timeout: Timeout in seconds
            check_returncode: Whether to treat non-zero return codes as errors

        Returns:
            Tuple of (success, stdout, stderr)
            - success: True if execution succeeded
            - stdout: Decoded stdout output (None on error)
            - stderr: Decoded stderr output (None if no error)
        """
        cmd = [self.binary_path] + args

        # Log the command being executed
        self.logger.info(f"Executing command: {' '.join(repr(arg) for arg in cmd)}")

        try:
            # Prepare input
            input_bytes = input_data.encode('utf-8') if input_data else None

            # Execute subprocess
            result = subprocess.run(
                cmd,
                input=input_bytes,
                capture_output=True,
                timeout=timeout,
                check=False  # We'll handle return codes ourselves
            )

            # Decode output
            stdout = self._decode_output(result.stdout)
            stderr = self._decode_output(result.stderr)

            # Check return code
            if check_returncode and result.returncode != 0:
                self.logger.error(
                    f"{self.binary_name} failed with return code {result.returncode}: {stderr}"
                )
                return False, None, stderr

            return True, stdout, stderr

        except subprocess.TimeoutExpired:
            self.logger.error(
                f"{self.binary_name} timed out after {timeout} seconds"
            )
            return False, None, f"Timeout after {timeout} seconds"

        except FileNotFoundError:
            self.logger.error(
                f"{self.binary_name} binary not found at: {self.binary_path}"
            )
            return False, None, f"Binary not found: {self.binary_path}"

        except Exception as e:
            self.logger.error(
                f"{self.binary_name} execution failed: {e}"
            )
            return False, None, str(e)

    def execute_and_parse_lines(
        self,
        args: List[str],
        input_data: Optional[str] = None,
        timeout: int = 30,
        filter_empty: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Execute program and return output as a list of lines.

        Args:
            args: Command-line arguments
            input_data: Optional stdin input
            timeout: Timeout in seconds
            filter_empty: Whether to filter out empty lines

        Returns:
            Tuple of (success, lines)
            - success: True if execution succeeded
            - lines: List of output lines (empty list on error)
        """
        success, stdout, stderr = self.execute(
            args, input_data, timeout, check_returncode=True
        )

        if not success or stdout is None:
            return False, []

        lines = stdout.split('\n')

        if filter_empty:
            lines = [line.strip() for line in lines if line.strip()]

        return True, lines

    def execute_and_get_last_line(
        self,
        args: List[str],
        input_data: Optional[str] = None,
        timeout: int = 30
    ) -> Tuple[bool, Optional[str]]:
        """
        Execute program and return the last non-empty line of output.

        Useful for programs that output a final result on the last line.

        Args:
            args: Command-line arguments
            input_data: Optional stdin input
            timeout: Timeout in seconds

        Returns:
            Tuple of (success, last_line)
            - success: True if execution succeeded
            - last_line: Last non-empty output line (None on error)
        """
        success, lines = self.execute_and_parse_lines(
            args, input_data, timeout, filter_empty=True
        )

        if not success or not lines:
            return False, None

        return True, lines[-1]

    @staticmethod
    def _decode_output(output: Optional[bytes]) -> Optional[str]:
        """
        Decode bytes output to string, handling various encodings.

        Args:
            output: Raw bytes from subprocess

        Returns:
            Decoded string, or None if output was None
        """
        if output is None:
            return None

        if isinstance(output, str):
            return output.strip()

        try:
            return output.decode('utf-8').strip()
        except UnicodeDecodeError:
            # Fallback to latin-1 which accepts all byte values
            return output.decode('latin-1').strip()

    def check_binary_exists(self) -> bool:
        """
        Check if the binary exists and is executable.

        Returns:
            True if binary is accessible
        """
        binary_path = Path(self.binary_path)

        if not binary_path.exists():
            self.logger.warning(
                f"{self.binary_name} binary not found at: {self.binary_path}"
            )
            return False

        if not binary_path.is_file():
            self.logger.warning(
                f"{self.binary_name} path is not a file: {self.binary_path}"
            )
            return False

        return True
