#!/usr/bin/env python3
"""
Helper functions for Stage 1 ECM result submission and residue management.

Consolidates the complete stage1 submission workflow that appears multiple times
in the codebase (auto-work, manual mode, two-stage pipeline).
"""
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path


def submit_stage1_complete_workflow(
    wrapper,
    results: Dict[str, Any],
    residue_file: Path,
    work_id: Optional[str],
    project: Optional[str],
    client_id: str,
    factor_found: Optional[str],
    cleanup_residue: bool = True
) -> Optional[str]:
    """
    Complete Stage 1 submission workflow: submit results, upload residue, cleanup.

    This consolidates the full workflow that appears in multiple locations:
    - Submit stage1 results to API
    - Handle submission failures (abandon work, cleanup)
    - Extract attempt_id from response
    - Upload residue file if no factor found
    - Clean up local residue file

    Args:
        wrapper: ECMWrapper instance (for API client and logger)
        results: Stage1 results dictionary (from _build_stage1_results())
        residue_file: Path to residue file
        work_id: Optional work assignment ID (for auto-work mode)
        project: Optional project name
        client_id: Client identifier
        factor_found: Factor string if found, None otherwise
        cleanup_residue: Whether to delete local residue file after processing

    Returns:
        Stage1 attempt_id from API response, or None if submission failed

    Side Effects:
        - Submits results to API
        - May abandon work assignment on failure
        - May upload residue file to server
        - May delete local residue file
    """
    print("Submitting stage 1 results...")
    program_name = 'gmp-ecm-ecm'

    # Submit stage1 results to API
    submit_response = wrapper.submit_result(results, project, program_name)

    if not submit_response:
        # Submission failed
        wrapper.logger.error("Failed to submit stage 1 results")

        # Abandon work assignment if this was auto-work
        if work_id:
            wrapper.abandon_work(work_id, reason="submission_failed")

        # Clean up residue file on failure
        if cleanup_residue and residue_file.exists():
            residue_file.unlink()

        return None

    # Extract attempt_id from response
    stage1_attempt_id = submit_response.get('attempt_id')
    if stage1_attempt_id:
        print(f"Stage 1 attempt ID: {stage1_attempt_id}")

    # Upload residue file if needed (skips if factor found)
    wrapper._upload_residue_if_needed(
        residue_file=residue_file,
        stage1_attempt_id=stage1_attempt_id,
        factor_found=factor_found,
        client_id=client_id
    )

    # Clean up local residue file
    if cleanup_residue and residue_file.exists():
        residue_file.unlink()

    return stage1_attempt_id


def handle_stage1_failure(
    wrapper,
    work_id: Optional[str],
    residue_file: Optional[Path],
    error_msg: str
) -> None:
    """
    Handle stage1 execution failure: log error, abandon work, cleanup.

    Args:
        wrapper: ECMWrapper instance
        work_id: Optional work assignment ID to abandon
        residue_file: Optional residue file path to cleanup
        error_msg: Error message to log
    """
    wrapper.logger.error(error_msg)

    # Abandon work assignment if applicable
    if work_id:
        wrapper.abandon_work(work_id, reason="execution_error")

    # Clean up residue file if it exists
    if residue_file and residue_file.exists():
        residue_file.unlink()
