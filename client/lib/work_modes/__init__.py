#!/usr/bin/env python3
"""
Work mode strategy pattern for ECM auto-work execution.

Each mode implements the same abstract interface:
- request_work() - Get work assignment from server
- execute_work() - Run ECM on the assignment
- submit_results() - Submit results to server
- complete_work() - Mark work as complete
- cleanup_on_failure() - Mode-specific cleanup

The base class provides the work loop template that all modes share.

This package was split out of the original `lib/work_modes.py` so each mode
lives in its own module. Public symbols are re-exported here so existing
imports (`from lib.work_modes import WorkLoopContext, get_work_mode`) keep
working unchanged.
"""

from .base import WorkMode, WorkLoopContext, MAX_CONSECUTIVE_FAILURES
from .stage1_producer import Stage1ProducerMode
from .stage2_consumer import Stage2ConsumerMode
from .p1_sweep import P1WorkMode
from .standard import StandardAutoWorkMode
from .composite_target import CompositeTargetMode
from .adaptive import AdaptiveCPUMode


def get_work_mode(ctx: WorkLoopContext) -> WorkMode:
    """
    Factory function to create the appropriate WorkMode based on args.

    Priority order:
    1. Explicit mode flags (--composite, --pm1, --stage1-only, --stage2-only)
    2. Adaptive CPU mode (--adaptive or interactive selection)
    3. Standard auto-work mode (legacy default with explicit --standard flag,
       or when B1/tlevel args are provided)

    Args:
        ctx: Work loop context with wrapper, client_id, and args

    Returns:
        Appropriate WorkMode subclass instance
    """
    args = ctx.args

    if getattr(args, 'composite', None):
        return CompositeTargetMode(ctx)
    elif getattr(args, 'pm1', False) or getattr(args, 'pp1', False) or getattr(args, 'p1', False):
        return P1WorkMode(ctx)
    elif getattr(args, 'stage1_only', False):
        return Stage1ProducerMode(ctx)
    elif getattr(args, 'stage2_only', False):
        return Stage2ConsumerMode(ctx)
    elif getattr(args, 'adaptive', False):
        return AdaptiveCPUMode(ctx)
    else:
        return StandardAutoWorkMode(ctx)


__all__ = [
    'WorkMode',
    'WorkLoopContext',
    'MAX_CONSECUTIVE_FAILURES',
    'Stage1ProducerMode',
    'Stage2ConsumerMode',
    'P1WorkMode',
    'StandardAutoWorkMode',
    'CompositeTargetMode',
    'AdaptiveCPUMode',
    'get_work_mode',
]
