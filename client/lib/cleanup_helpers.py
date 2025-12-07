#!/usr/bin/env python3
"""
Cleanup and shutdown utilities for ECM wrapper.

Handles graceful shutdown with work abandonment and resource cleanup.
"""
from typing import Optional, Union
from pathlib import Path
import sys

from .user_output import UserOutput


def handle_shutdown(
    wrapper,
    current_work_id: Optional[str],
    current_residue_id: Optional[Union[str, int]],
    mode_name: str,
    completed_count: int,
    local_residue_file: Optional[Path] = None,
    output: Optional[UserOutput] = None
) -> None:
    """
    Handle graceful shutdown with cleanup.

    Consolidates shutdown logic from multiple auto-work modes:
    - Prints shutdown message
    - Abandons active work assignments
    - Abandons active residue work
    - Cleans up local residue files
    - Prints completion summary
    - Exits cleanly

    Args:
        wrapper: ECMWrapper instance
        current_work_id: Active work assignment ID to abandon
        current_residue_id: Active residue work ID to abandon
        mode_name: Human-readable mode name (e.g., "Stage 1 Producer", "Auto-work")
        completed_count: Number of work assignments completed
        local_residue_file: Optional local residue file to clean up
        output: Optional UserOutput instance (creates one if not provided)

    Side Effects:
        - Abandons work assignments
        - Deletes local files
        - Calls sys.exit(0)
    """
    if output is None:
        output = UserOutput()

    output.section("Shutdown requested...")

    # Abandon active work assignment
    if current_work_id:
        output.info(f"Abandoning work assignment {current_work_id}...")
        wrapper.abandon_work(current_work_id, reason="client_interrupted")

    # Abandon active residue work
    if current_residue_id:
        output.info(f"Abandoning residue work {current_residue_id}...")
        wrapper.api_client.abandon_residue(
            wrapper.client_id,
            current_residue_id,
            reason="client_interrupted"
        )

    # Clean up local residue file
    if local_residue_file and local_residue_file.exists():
        local_residue_file.unlink()

    # Print completion summary
    output.info(f"{mode_name} stopped - completed {completed_count} assignment(s)")
    sys.exit(0)
