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


def resolve_param(args: argparse.Namespace, use_gpu: bool) -> int:
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
        Parametrization value (0-3)
    """
    # Check if param explicitly specified in args
    if hasattr(args, 'param') and args.param is not None:
        return args.param

    # Default to param 3 for GPU mode, param 1 for CPU mode
    return 3 if use_gpu else 1


def resolve_workers(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    """
    Resolve worker count from arguments with config default.

    Used for both multiprocess workers and stage2 threads.

    Priority:
    1. Command-line --workers argument (if set and > 0)
    2. Config file setting
    3. Fallback to 4

    Args:
        args: Parsed command-line arguments
        config: Configuration dictionary

    Returns:
        Number of workers to use
    """
    # Check if explicitly set via command line
    if hasattr(args, 'workers') and args.workers and args.workers > 0:
        return args.workers

    # Otherwise use config default
    return get_workers_default(config)


# Backward compatibility alias
resolve_stage2_workers = resolve_workers


def get_workers_default(config: Dict[str, Any]) -> int:
    """
    Get default worker count from config.

    Args:
        config: Configuration dictionary

    Returns:
        Default number of workers (from config or 4)
    """
    return config.get('execution', {}).get('workers', 4)


# Backward compatibility alias
get_stage2_workers_default = get_workers_default
