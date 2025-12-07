"""
User Output Abstraction

Provides a unified interface for user-facing output, separating
user messages from debug logging. This allows:
- Consistent output formatting across the codebase
- Easy redirection/suppression of user output
- Clear separation between user messages and log messages
"""

import logging
import sys
from typing import Any, Optional, TextIO


class UserOutput:
    """
    Unified handler for user-facing output.

    This class separates user-facing messages (status updates, results, errors)
    from debug logging. All user messages go to stdout/stderr, while debug
    information goes to the logger.

    Usage:
        output = UserOutput()
        output.info("Processing composite...")
        output.success("Factor found: 12345")
        output.error("Failed to connect to API")
        output.warning("B2 not specified, using default")

        # With sections
        output.section("Stage 1 Results")
        output.item("Curves run", 100)
        output.item("Time", "45.2s")
    """

    def __init__(
        self,
        stdout: Optional[TextIO] = None,
        stderr: Optional[TextIO] = None,
        quiet: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize output handler.

        Args:
            stdout: Output stream for normal messages (default: sys.stdout)
            stderr: Output stream for errors (default: sys.stderr)
            quiet: If True, suppress all non-error output
            logger: Optional logger for debug messages
        """
        self.stdout = stdout or sys.stdout
        self.stderr = stderr or sys.stderr
        self.quiet = quiet
        self.logger = logger or logging.getLogger(__name__)

    def info(self, message: str, log: bool = False) -> None:
        """
        Print informational message to user.

        Args:
            message: Message to display
            log: If True, also log to debug logger
        """
        if not self.quiet:
            print(message, file=self.stdout)
        if log:
            self.logger.info(message)

    def success(self, message: str, log: bool = True) -> None:
        """
        Print success message to user.

        Args:
            message: Success message to display
            log: If True, also log to info logger
        """
        if not self.quiet:
            print(message, file=self.stdout)
        if log:
            self.logger.info(message)

    def warning(self, message: str, log: bool = True) -> None:
        """
        Print warning message to user.

        Args:
            message: Warning message to display
            log: If True, also log to warning logger
        """
        if not self.quiet:
            print(f"Warning: {message}", file=self.stdout)
        if log:
            self.logger.warning(message)

    def error(self, message: str, log: bool = True) -> None:
        """
        Print error message to user (always shown, even in quiet mode).

        Args:
            message: Error message to display
            log: If True, also log to error logger
        """
        print(f"Error: {message}", file=self.stderr)
        if log:
            self.logger.error(message)

    def section(self, title: str) -> None:
        """
        Print a section header.

        Args:
            title: Section title
        """
        if not self.quiet:
            print(f"\n{title}", file=self.stdout)

    def separator(self, char: str = "=", width: int = 80) -> None:
        """
        Print a separator line.

        Args:
            char: Character to use for separator
            width: Width of separator line
        """
        if not self.quiet:
            print(char * width, file=self.stdout)

    def item(self, label: str, value: Any, indent: int = 2) -> None:
        """
        Print a labeled item (key-value pair).

        Args:
            label: Item label
            value: Item value
            indent: Number of spaces to indent
        """
        if not self.quiet:
            prefix = " " * indent
            print(f"{prefix}{label}: {value}", file=self.stdout)

    def bullet(self, message: str, indent: int = 2) -> None:
        """
        Print a bullet point item.

        Args:
            message: Message to display
            indent: Number of spaces to indent
        """
        if not self.quiet:
            prefix = " " * indent
            print(f"{prefix}{message}", file=self.stdout)

    def blank(self) -> None:
        """Print a blank line."""
        if not self.quiet:
            print(file=self.stdout)

    def mode_header(self, mode: str, details: dict) -> None:
        """
        Print a standardized mode header with details.

        Args:
            mode: Mode name (e.g., "Stage 1 Only Mode", "T-level Mode")
            details: Dictionary of key-value pairs to display
        """
        if self.quiet:
            return
        print(f"{mode}", file=self.stdout)
        for key, value in details.items():
            print(f"  {key}: {value}", file=self.stdout)

    def result_summary(
        self,
        curves_run: int,
        execution_time: float,
        factors: Optional[list] = None
    ) -> None:
        """
        Print a standardized execution result summary.

        Args:
            curves_run: Number of curves completed
            execution_time: Execution time in seconds
            factors: List of factors found (if any)
        """
        if self.quiet:
            return
        print(f"\nExecution Summary:", file=self.stdout)
        print(f"  Curves completed: {curves_run}", file=self.stdout)
        print(f"  Execution time: {execution_time:.2f}s", file=self.stdout)
        if factors:
            print(f"  Factors found: {', '.join(factors)}", file=self.stdout)
        else:
            print(f"  No factors found", file=self.stdout)


# Global default instance for convenience
_default_output: Optional[UserOutput] = None


def get_output(quiet: bool = False) -> UserOutput:
    """
    Get the default UserOutput instance.

    Args:
        quiet: If True and no instance exists, create a quiet one

    Returns:
        UserOutput instance
    """
    global _default_output
    if _default_output is None:
        _default_output = UserOutput(quiet=quiet)
    return _default_output


def set_output(output: UserOutput) -> None:
    """
    Set the default UserOutput instance.

    Args:
        output: UserOutput instance to use as default
    """
    global _default_output
    _default_output = output
