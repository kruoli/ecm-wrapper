#!/usr/bin/env python3
"""
Helper functions for ECM work assignment display and management.

These utilities eliminate duplicated work handling code in auto-work mode.
"""
from typing import Optional, Dict, Any
import argparse
import time

from .user_output import UserOutput


def print_work_header(work_id: Optional[str], composite: str, digit_length: int,
                     params: Dict[str, Any], output: Optional[UserOutput] = None) -> None:
    """
    Print formatted work assignment header.

    Args:
        work_id: Work assignment ID (may be None for residue work)
        composite: Composite number being factored
        digit_length: Number of digits in composite
        params: Dictionary of parameters to display (e.g., {'B1': 50000, 'curves': 100})
        output: Optional UserOutput instance (creates one if not provided)
    """
    if output is None:
        output = UserOutput()

    output.blank()
    output.separator()
    work_id_display = work_id if work_id else "(no work_id)"
    output.info(f"Processing work assignment {work_id_display}")

    # Truncate composite for display
    composite_display = composite[:50] + "..." if len(composite) > 50 else composite
    output.info(f"Composite: {composite_display} ({digit_length} digits)")

    # Format parameters
    if params:
        param_strs = [f"{k}={v}" for k, v in params.items()]
        output.info(f"Parameters: {', '.join(param_strs)}")

    output.separator()
    output.blank()


def print_work_status(stage_name: str, completed_count: int,
                     work_count_limit: Optional[int] = None,
                     output: Optional[UserOutput] = None) -> bool:
    """
    Print work completion status and check if limit reached.

    Args:
        stage_name: Name of the stage/mode (e.g., "Stage 1", "Stage 2 work")
        completed_count: Number of work assignments completed
        work_count_limit: Optional limit on number of assignments
        output: Optional UserOutput instance (creates one if not provided)

    Returns:
        True if work count limit has been reached, False otherwise
    """
    if output is None:
        output = UserOutput()

    output.blank()

    # Display completion status
    if work_count_limit:
        output.info(f"{stage_name} complete ({completed_count}/{work_count_limit})")
    else:
        output.info(f"{stage_name} complete (total: {completed_count})")

    output.separator()
    output.blank()

    # Check if limit reached
    if work_count_limit and completed_count >= work_count_limit:
        output.info(f"Reached work count limit ({work_count_limit}), exiting...")
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
        min_target_tlevel=args.min_target_tlevel if hasattr(args, 'min_target_tlevel') else None,
        max_target_tlevel=args.max_target_tlevel if hasattr(args, 'max_target_tlevel') else None,
        priority=args.priority if hasattr(args, 'priority') else None,
        min_digits=args.min_digits if hasattr(args, 'min_digits') else None,
        max_digits=args.max_digits if hasattr(args, 'max_digits') else None,
        work_type=args.work_type if hasattr(args, 'work_type') else 'standard'
    )

    if not work:
        logger.info("No work available, waiting 30 seconds before retry...")
        time.sleep(30)

    return work


def request_p1_work(api_client, client_id: str, args: argparse.Namespace,
                    logger) -> Optional[Dict[str, Any]]:
    """
    Request P-1/P+1 work from server with automatic retry on failure.

    Calls the /p1-work endpoint which only assigns composites that haven't
    had PM1/PP1 done at the required B1 level.

    Args:
        api_client: API client instance
        client_id: Client identifier
        args: Command-line arguments containing filter parameters and method flags
        logger: Logger instance for info messages

    Returns:
        Work assignment dictionary or None if no work available after retry
    """
    # Determine method from args
    if getattr(args, 'pm1', False):
        method = 'pm1'
    elif getattr(args, 'pp1', False):
        method = 'pp1'
    else:
        method = 'p1'

    work = api_client.get_p1_work(
        client_id=client_id,
        method=method,
        min_target_tlevel=args.min_target_tlevel if hasattr(args, 'min_target_tlevel') else None,
        max_target_tlevel=args.max_target_tlevel if hasattr(args, 'max_target_tlevel') else None,
        priority=args.priority if hasattr(args, 'priority') else None,
        min_digits=args.min_digits if hasattr(args, 'min_digits') else None,
        max_digits=args.max_digits if hasattr(args, 'max_digits') else None,
        work_type=args.work_type if hasattr(args, 'work_type') else 'standard'
    )

    if not work:
        logger.info("No P1 work available, waiting 30 seconds before retry...")
        time.sleep(30)

    return work
