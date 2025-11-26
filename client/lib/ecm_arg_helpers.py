#!/usr/bin/env python3
"""
Helper functions for parsing and resolving ECM command-line arguments.

These utilities eliminate duplicated argument handling code across the codebase.
"""
from typing import Optional, Union, Dict, Any
import argparse


def parse_sigma_arg(args: argparse.Namespace) -> Optional[Union[str, int]]:
    """
    Parse sigma parameter from command line arguments.

    Handles both formats:
    - Integer format: "12345"
    - Parametrization prefix format: "3:12345"

    Args:
        args: Parsed command-line arguments

    Returns:
        Sigma value as str (if contains ':') or int, or None if not provided
    """
    if not hasattr(args, 'sigma') or not args.sigma:
        return None

    # If sigma contains ':', keep as string for parametrization format
    if ':' in args.sigma:
        return args.sigma

    # Otherwise convert to integer
    return int(args.sigma)


def resolve_param(args: argparse.Namespace, use_gpu: bool) -> Optional[int]:
    """
    Resolve ECM parametrization from arguments with GPU default.

    Parametrization values:
    - 0: (x0, y0) coordinates
    - 1: Montgomery curves (CPU default)
    - 2: Weierstrass curves
    - 3: Twisted Edwards curves (GPU default)

    Args:
        args: Parsed command-line arguments
        use_gpu: Whether GPU mode is enabled

    Returns:
        Parametrization value (0-3) or None if not specified
    """
    # Check if param explicitly specified in args
    if hasattr(args, 'param') and args.param is not None:
        return args.param

    # Default to param 3 for GPU mode, None otherwise
    return 3 if use_gpu else None


def resolve_stage2_workers(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """
    Resolve stage2 worker count from arguments with config default.

    Priority:
    1. Command-line argument (if not default value of 4)
    2. Config file setting
    3. Fallback to 4

    Args:
        args: Parsed command-line arguments
        config: Configuration dictionary

    Returns:
        Number of stage2 workers to use
    """
    # Check if explicitly set via command line (not the default 4)
    if hasattr(args, 'stage2_workers') and args.stage2_workers != 4:
        return args.stage2_workers

    # Otherwise use config default
    return get_stage2_workers_default(config)


def get_stage2_workers_default(config: Dict[str, Any]) -> int:
    """
    Get default stage2 worker count from config.

    Args:
        config: Configuration dictionary

    Returns:
        Default number of stage2 workers (from config or 4)
    """
    return config.get('execution', {}).get('stage2_workers', 4)
