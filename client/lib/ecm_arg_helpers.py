#!/usr/bin/env python3
"""
Helper functions for parsing and resolving ECM command-line arguments.

These utilities eliminate duplicated argument handling code across the codebase.
"""
from typing import Optional, Union, TYPE_CHECKING
import argparse

if TYPE_CHECKING:
    from .work_args import WorkArgs

ArgsLike = Union[argparse.Namespace, "WorkArgs"]


def parse_sigma_arg(args: ArgsLike) -> Optional[Union[str, int]]:
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
    if not args.sigma:
        return None

    # If sigma contains ':', keep as string for parametrization format
    if ':' in args.sigma:
        return args.sigma

    # Otherwise convert to integer
    return int(args.sigma)


def resolve_param(args: ArgsLike, use_gpu: bool) -> int:
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
    if args.param is not None:
        return args.param

    # Default to param 3 for GPU mode, param 1 for CPU mode
    return 3 if use_gpu else 1
