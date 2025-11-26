#!/usr/bin/env python3
"""
Helper functions for ECM work assignment display and management.

These utilities eliminate duplicated work handling code in auto-work mode.
"""
from typing import Optional, Dict, Any
import argparse
import time


def print_work_header(work_id: str, composite: str, digit_length: int,
                     params: Dict[str, Any]) -> None:
    """
    Print formatted work assignment header.

    Args:
        work_id: Work assignment ID
        composite: Composite number being factored
        digit_length: Number of digits in composite
        params: Dictionary of parameters to display (e.g., {'B1': 50000, 'curves': 100})
    """
    print()
    print("=" * 60)
    print(f"Processing work assignment {work_id}")

    # Truncate composite for display
    composite_display = composite[:50] + "..." if len(composite) > 50 else composite
    print(f"Composite: {composite_display} ({digit_length} digits)")

    # Format parameters
    if params:
        param_strs = [f"{k}={v}" for k, v in params.items()]
        print(f"Parameters: {', '.join(param_strs)}")

    print("=" * 60)
    print()


def print_work_status(stage_name: str, completed_count: int,
                     work_count_limit: Optional[int] = None) -> bool:
    """
    Print work completion status and check if limit reached.

    Args:
        stage_name: Name of the stage/mode (e.g., "Stage 1", "Stage 2 work")
        completed_count: Number of work assignments completed
        work_count_limit: Optional limit on number of assignments

    Returns:
        True if work count limit has been reached, False otherwise
    """
    print()

    # Display completion status
    if work_count_limit:
        print(f"{stage_name} complete ({completed_count}/{work_count_limit})")
    else:
        print(f"{stage_name} complete (total: {completed_count})")

    print("=" * 60)
    print()

    # Check if limit reached
    if work_count_limit and completed_count >= work_count_limit:
        print(f"Reached work count limit ({work_count_limit}), exiting...")
        return True

    return False


def request_ecm_work(api_client, client_id: str, args: argparse.Namespace,
                    logger) -> Optional[Dict[str, Any]]:
    """
    Request ECM work from server with automatic retry on failure.

    Args:
        api_client: API client instance
        client_id: Client identifier
        args: Command-line arguments containing filter parameters
        logger: Logger instance for info messages

    Returns:
        Work assignment dictionary or None if no work available after retry
    """
    work = api_client.get_ecm_work(
        client_id=client_id,
        min_digits=args.min_digits if hasattr(args, 'min_digits') else None,
        max_digits=args.max_digits if hasattr(args, 'max_digits') else None,
        priority=args.priority if hasattr(args, 'priority') else None,
        work_type=args.work_type if hasattr(args, 'work_type') else 'standard'
    )

    if not work:
        logger.info("No work available, waiting 30 seconds before retry...")
        time.sleep(30)

    return work
