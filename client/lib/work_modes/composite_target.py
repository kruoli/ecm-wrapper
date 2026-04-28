#!/usr/bin/env python3
"""
Composite Target mode: factor a specific composite by querying server status.
"""

from typing import Any, Optional, Dict

from ..work_helpers import print_work_header
from .base import WorkLoopContext
from .standard import StandardAutoWorkMode


class CompositeTargetMode(StandardAutoWorkMode):
    """
    Target a specific composite by querying the server for its t-level status.

    This mode:
    1. Queries the server for the composite's current/target t-level
    2. Runs ECM using t-level mode to make optimal progress
    3. Submits results to server
    4. Does NOT create/complete work assignments (just t-level work)

    The composite must already exist in the server database.
    """

    mode_name = "Composite Target"

    def __init__(self, ctx: WorkLoopContext):
        super().__init__(ctx)
        self.target_composite = ctx.args.composite
        self._work_done = False

    def request_work(self) -> Optional[Dict[str, Any]]:
        """Query composite status from server and build work dict."""
        # Only run once
        if self._work_done:
            return None

        self._work_done = True

        # Query composite status from server
        status = self.api_client.get_composite_status(self.target_composite)

        if status is None:
            print(f"Error: Composite not found in server database")
            print(f"  {self.target_composite[:50]}...")
            print(f"\nTo add this composite, use the server admin interface or API.")
            return None

        # Check if already fully factored
        if status.get('status') == 'fully_factored':
            print(f"Composite is already fully factored!")
            print(f"  Factors: {status.get('factors_found', [])}")
            return None

        if status.get('status') == 'prime':
            print(f"Number is prime, no factorization needed.")
            return None

        # Check if t-level target is reached
        current_t = status.get('current_t_level', 0) or 0
        target_t = status.get('target_t_level', 0) or 0

        if target_t > 0 and current_t >= target_t:
            print(f"Target t-level already reached: {current_t:.1f} >= {target_t:.1f}")
            print(f"\nTo continue, increase the target t-level on the server.")
            return None

        # Build work dict compatible with StandardAutoWorkMode
        # Use current_composite if available (partially factored), else original
        composite_to_factor = status.get('current_composite') or status.get('composite') or self.target_composite

        work = {
            'work_id': None,  # No formal work assignment
            'composite': composite_to_factor,
            'composite_id': None,  # Not needed for submission
            'digit_length': status.get('digit_length') or len(composite_to_factor),
            'current_t_level': current_t,
            'target_t_level': target_t if target_t > 0 else 35.0,  # Default t35 if not set
        }

        return work

    def on_work_started(self, work: Dict[str, Any]) -> None:
        """Log work start without setting work_id."""
        # Don't call super() - we don't have a real work_id
        self.current_work_id = None

        print_work_header(
            work_id="(direct)",
            composite=work['composite'],
            digit_length=work['digit_length'],
            params={
                'T-level': f"{work.get('current_t_level', 0):.1f} -> {work.get('target_t_level', 0):.1f}"
            }
        )

    def complete_work(self, work: Dict[str, Any]) -> None:
        """No work assignment to complete in composite target mode."""
        # Results are already submitted via t-level mode's internal submission
        pass
