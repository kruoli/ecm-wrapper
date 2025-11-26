#!/usr/bin/env python3
"""
Error handling utilities for ECM wrapper.

Standardizes exception handling patterns used across the codebase.
"""
from typing import Optional


def handle_work_failure(
    wrapper,
    current_work_id: Optional[str],
    consecutive_failures: int,
    max_failures: int,
    error_msg: str
) -> bool:
    """
    Handle work assignment failure with circuit breaker pattern.

    Args:
        wrapper: ECMWrapper instance
        current_work_id: Current work assignment ID (will be abandoned)
        consecutive_failures: Current count of consecutive failures
        max_failures: Maximum allowed consecutive failures
        error_msg: Error message to log

    Returns:
        True if circuit breaker triggered (should exit), False otherwise

    Side Effects:
        - Logs error message
        - Abandons current work assignment
        - May trigger circuit breaker exit
    """
    wrapper.logger.error(error_msg)

    # Abandon current work assignment
    if current_work_id:
        wrapper.abandon_work(current_work_id, reason="execution_error")

    # Check circuit breaker threshold
    if consecutive_failures >= max_failures:
        wrapper.logger.error(
            f"Too many consecutive failures ({consecutive_failures}), exiting..."
        )
        return True  # Signal to exit

    return False  # Continue processing


def check_work_limit_reached(
    completed_count: int,
    work_count_limit: Optional[int]
) -> bool:
    """
    Check if work count limit has been reached.

    Args:
        completed_count: Number of completed work assignments
        work_count_limit: Maximum work assignments to process (None = unlimited)

    Returns:
        True if limit reached and should exit, False otherwise
    """
    if work_count_limit and completed_count >= work_count_limit:
        print(f"Reached work count limit ({work_count_limit}), exiting...")
        return True

    return False
