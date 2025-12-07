#!/usr/bin/env python3
"""
ECM Client - Server-coordinated factorization work

This entry point handles all server-coordinated ECM modes:
- Auto-work with server-provided composites
- Stage 1 only (GPU producer): Upload residues to server
- Stage 2 only (CPU consumer): Download and process residues from server

All modes get composites from the server - no --composite flag needed.
"""

import sys
import time
from pathlib import Path

# Import ECMWrapper from lib
from lib.ecm_executor import ECMWrapper
from lib.ecm_config import ECMConfig, TwoStageConfig, MultiprocessConfig, TLevelConfig
from lib.arg_parser import (
    resolve_gpu_settings, get_method_defaults,
    resolve_worker_count, get_stage2_workers_default
)
from lib.ecm_arg_helpers import parse_sigma_arg, resolve_param, resolve_stage2_workers
from lib.work_helpers import print_work_header, print_work_status, request_ecm_work
from lib.stage1_helpers import submit_stage1_complete_workflow
from lib.error_helpers import handle_work_failure, check_work_limit_reached
from lib.cleanup_helpers import handle_shutdown
from lib.results_builder import results_for_stage1
from lib.stage2_executor import Stage2Executor


def create_client_parser():
    """Create argument parser for ecm_client.py (server-coordinated modes)."""
    import argparse

    parser = argparse.ArgumentParser(
        description='ECM Client - Server-coordinated factorization work',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-work with server defaults
  python3 ecm_client.py

  # Process 10 work items with client-specified B1/B2
  python3 ecm_client.py --work-count 10 --b1 50000 --b2 5000000 --curves 100

  # Stage 1 only - upload residues to server
  python3 ecm_client.py --stage1-only --b1 110000000 --curves 3000

  # Stage 2 only - download and process residues
  python3 ecm_client.py --stage2-only --b2 11000000000000 --stage2-workers 8
"""
    )

    # Work filtering
    parser.add_argument('--work-count', type=int,
                       help='Number of work items to process (default: unlimited)')
    parser.add_argument('--min-digits', type=int,
                       help='Minimum composite size (digits)')
    parser.add_argument('--max-digits', type=int,
                       help='Maximum composite size (digits)')
    parser.add_argument('--priority', type=int,
                       help='Minimum priority level')

    # Execution parameters (override server defaults)
    parser.add_argument('--tlevel', type=float,
                       help='Target t-level (overrides server t-level)')
    parser.add_argument('--b1', type=int,
                       help='B1 parameter (overrides server default)')
    parser.add_argument('--b2', type=int,
                       help='B2 parameter (overrides server default, -1 for GMP-ECM default)')
    parser.add_argument('--b2-multiplier', type=float,
                       help='Dynamic B2 = B1 * multiplier (for stage2-only mode)')
    parser.add_argument('--curves', type=int,
                       help='Curves per batch')
    parser.add_argument('--method', choices=['ecm', 'pm1', 'pp1'], default='ecm',
                       help='Factorization method (default: ecm)')

    # Execution modes
    parser.add_argument('--multiprocess', action='store_true',
                       help='Use multiprocess parallelization')
    parser.add_argument('--workers', type=int,
                       help='Number of worker processes (default: CPU count)')
    parser.add_argument('--two-stage', action='store_true',
                       help='Use two-stage GPU+CPU mode')

    # Decoupled two-stage modes (mutually exclusive)
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument('--stage1-only', action='store_true',
                            help='Stage 1 only: upload residue to server')
    stage_group.add_argument('--stage2-only', action='store_true',
                            help='Stage 2 only: download residue from server')

    # Stage 2 specific
    parser.add_argument('--stage2-workers', type=int,
                       help='Number of worker threads for stage 2')

    # GPU/compute
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration')
    parser.add_argument('--gpu-device', type=int,
                       help='GPU device number')
    parser.add_argument('--param', type=int, choices=[0, 1, 2, 3],
                       help='ECM parametrization (0-3)')
    parser.add_argument('--sigma', type=str,
                       help='Sigma value (integer or parametrization:value)')

    # Execution behavior
    parser.add_argument('--continue-after-factor', action='store_true',
                       help='Continue running curves even after finding a factor')
    parser.add_argument('--progress-interval', type=int, default=0,
                       help='Report progress every N curves (0=disable)')

    # API settings
    parser.add_argument('--project', type=str,
                       help='Project name for submissions')
    parser.add_argument('--no-submit', action='store_true',
                       help='Skip result submission to server')

    # Logging
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')

    # Hidden: for backward compatibility, auto-work is implied
    parser.add_argument('--auto-work', action='store_true', dest='auto_work_explicit',
                       help=argparse.SUPPRESS)

    return parser


if __name__ == '__main__':
    parser = create_client_parser()
    args = parser.parse_args()

    # ecm_client.py always operates in auto-work mode (implied)
    args.auto_work = True

    wrapper = ECMWrapper('client.yaml')

    # === AUTO-WORK MODE SECTION ===
    # This is the extracted auto-work code from the original ecm_client.py lines 1171-1706

    work_count_limit = args.work_count if hasattr(args, 'work_count') and args.work_count else None

    # Check for decoupled two-stage modes
    is_stage1_only = hasattr(args, 'stage1_only') and args.stage1_only
    is_stage2_work = hasattr(args, 'stage2_only') and args.stage2_only  # Renamed from stage2_work

    print("=" * 60)
    if is_stage1_only:
        mode_name = "Stage 1 Producer (GPU)"
    elif is_stage2_work:
        mode_name = "Stage 2 Consumer (CPU)"
    else:
        mode_name = "Auto-work"

    if work_count_limit:
        print(f"{mode_name} mode enabled - will process {work_count_limit} assignment(s)")
    else:
        print(f"{mode_name} mode enabled - requesting work from server")
        print("Press Ctrl+C to stop")
    print("=" * 60)
    print()

    # Initialize API clients for auto-work mode
    wrapper._ensure_api_clients()

    # Get client ID from config
    client_id = wrapper.config['client']['username']
    current_work_id = None
    current_residue_id = None  # For stage2 mode
    completed_count = 0
    consecutive_failures = 0  # Track consecutive failures to prevent infinite loops
    MAX_CONSECUTIVE_FAILURES = 3

    # Stage 1 Only Mode (upload residues to server)
    if is_stage1_only:
        try:
            while not wrapper.interrupted:
                # Request regular ECM work
                work = request_ecm_work(wrapper.api_client, client_id, args, wrapper.logger)

                if not work:
                    continue

                current_work_id = work['work_id']
                composite = work['composite']
                digit_length = work['digit_length']

                # Resolve GPU settings
                use_gpu, gpu_device, gpu_curves = resolve_gpu_settings(args, wrapper.config)

                # Determine curves (use config default if not specified)
                curves = args.curves if args.curves is not None else wrapper.config['programs']['gmp_ecm']['default_curves']

                print_work_header(
                    work_id=current_work_id,
                    composite=composite,
                    digit_length=digit_length,
                    params={'B1': args.b1, 'curves': curves}
                )

                try:
                    # Generate residue file path
                    residue_dir = Path(wrapper.config['execution'].get('residue_dir', 'data/residues'))
                    residue_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    residue_file = residue_dir / f"stage1_{timestamp}_{composite[:20]}.txt"

                    # Run stage 1 only (B2=0)
                    sigma = parse_sigma_arg(args)
                    param = resolve_param(args, use_gpu)

                    print(f"Running ECM stage 1 (B1={args.b1}, curves={curves})...")
                    print(f"Saving residues to: {residue_file}")

                    success, factor, actual_curves, raw_output, all_factors = wrapper._run_stage1(
                        composite=composite,
                        b1=args.b1,
                        curves=curves,
                        residue_file=residue_file,
                        sigma=sigma,
                        param=param,
                        use_gpu=use_gpu,
                        gpu_device=gpu_device,
                        gpu_curves=gpu_curves,
                        verbose=args.verbose
                    )

                    # Check if we're interrupted (don't treat user cancellation as failure)
                    if wrapper.interrupted:
                        wrapper.logger.info("Stage 1 interrupted by user, cleaning up...")
                        # Try to abandon work (might fail if already expired)
                        try:
                            wrapper.abandon_work(current_work_id, reason="user_cancelled")
                        except Exception:
                            pass  # Ignore errors when abandoning (work might have expired)
                        current_work_id = None
                        break  # Exit the work loop

                    if not success:
                        wrapper.logger.error("Stage 1 execution failed")
                        wrapper.abandon_work(current_work_id, reason="stage1_failed")
                        current_work_id = None
                        continue

                    # Build results using ResultsBuilder
                    builder = (results_for_stage1(composite, args.b1, actual_curves, param if param is not None else 3)
                        .with_curves(actual_curves, actual_curves)
                        .with_factors(all_factors)
                        .add_raw_output(raw_output)
                        .with_execution_time(0))  # Will be filled by subprocess
                    if current_work_id:
                        builder.with_work_id(current_work_id)  # For failed submission recovery
                    results = builder.build()

                    # Submit stage 1 results and handle workflow
                    stage1_attempt_id = submit_stage1_complete_workflow(
                        wrapper=wrapper,
                        results=results,
                        residue_file=residue_file,
                        work_id=current_work_id,
                        project=args.project,
                        client_id=client_id,
                        factor_found=factor,
                        cleanup_residue=True
                    )

                    # Check if submission failed
                    if not stage1_attempt_id:
                        current_work_id = None
                        continue

                    # Mark work as complete
                    wrapper.api_client.complete_work(current_work_id, client_id)
                    current_work_id = None
                    completed_count += 1
                    consecutive_failures = 0  # Reset on success

                    if print_work_status("Stage 1", completed_count, work_count_limit):
                        break

                except Exception as e:
                    consecutive_failures += 1

                    # Handle work failure with circuit breaker
                    if handle_work_failure(
                        wrapper=wrapper,
                        current_work_id=current_work_id,
                        consecutive_failures=consecutive_failures,
                        max_failures=MAX_CONSECUTIVE_FAILURES,
                        error_msg=f"Error in stage 1 processing: {e}"
                    ):
                        break

                    current_work_id = None

                    # Check if work limit reached
                    if check_work_limit_reached(completed_count, work_count_limit):
                        break

        except KeyboardInterrupt:
            handle_shutdown(
                wrapper=wrapper,
                current_work_id=current_work_id,
                current_residue_id=None,
                mode_name="Stage 1 Producer mode",
                completed_count=completed_count
            )

        # Exit after stage1-only mode completes
        sys.exit(0)

    # Stage 2 Only Mode (download residues from server)
    elif is_stage2_work:
        # Initialize variables for KeyboardInterrupt handler
        local_residue_file = None

        try:
            while not wrapper.interrupted:
                # Request residue work from server
                residue_work = wrapper.api_client.get_residue_work(
                    client_id=client_id,
                    min_digits=args.min_digits if hasattr(args, 'min_digits') else None,
                    max_digits=args.max_digits if hasattr(args, 'max_digits') else None,
                    min_priority=args.priority if hasattr(args, 'priority') else None,
                    claim_timeout_hours=24
                )

                if not residue_work:
                    wrapper.logger.info("No residue work available, waiting 30 seconds before retry...")
                    time.sleep(30)
                    continue

                current_residue_id = residue_work['residue_id']
                composite = residue_work['composite']
                digit_length = residue_work['digit_length']
                b1 = residue_work['b1']
                curve_count = residue_work['curve_count']
                stage1_attempt_id = residue_work.get('stage1_attempt_id')
                suggested_b2 = residue_work.get('suggested_b2', b1 * 100)

                # Determine B2 (priority: explicit --b2 > --b2-multiplier > server suggestion)
                if args.b2 is not None:
                    # Explicit B2 specified (0 means GMP-ECM default)
                    b2 = args.b2
                elif hasattr(args, 'b2_multiplier') and args.b2_multiplier is not None:
                    # Dynamic calculation based on B1
                    b2 = int(b1 * args.b2_multiplier)
                    print(f"Using dynamic B2 = B1 * {args.b2_multiplier} = {b2}")
                else:
                    # Use server suggestion (default)
                    b2 = suggested_b2

                b2_display = "GMP-ECM default" if b2 == -1 else str(b2)
                print_work_header(
                    work_id=current_residue_id,
                    composite=composite,
                    digit_length=digit_length,
                    params={
                        'B1': b1,
                        'B2': b2_display,
                        'curves': curve_count,
                        'Stage 1 attempt ID': stage1_attempt_id
                    }
                )

                try:
                    # Download residue file
                    residue_dir = Path(wrapper.config['execution'].get('residue_dir', 'data/residues'))
                    residue_dir.mkdir(parents=True, exist_ok=True)
                    local_residue_file = residue_dir / f"s2_residue_{current_residue_id}.txt"

                    print(f"Downloading residue file...")
                    download_success = wrapper.api_client.download_residue(
                        client_id=client_id,
                        residue_id=current_residue_id,
                        output_path=str(local_residue_file)
                    )

                    if not download_success:
                        wrapper.logger.error("Failed to download residue file")
                        wrapper.api_client.abandon_residue(client_id, current_residue_id)
                        current_residue_id = None
                        continue

                    print(f"Downloaded {local_residue_file.stat().st_size} bytes")

                    # Get stage2 workers
                    stage2_workers = args.stage2_workers if hasattr(args, 'stage2_workers') else get_stage2_workers_default(wrapper.config)

                    # Run stage 2 on residue file
                    print(f"Running stage 2 with {stage2_workers} workers...")
                    stage2_executor = Stage2Executor(
                        wrapper, local_residue_file, b1, b2, stage2_workers, args.verbose
                    )
                    stage2_factor, stage2_all_factors, stage2_curves, stage2_time, stage2_sigma = stage2_executor.execute(
                        early_termination=not (hasattr(args, 'continue_after_factor') and args.continue_after_factor),
                        progress_interval=args.progress_interval if hasattr(args, 'progress_interval') else 0
                    )

                    # Build results
                    results = {
                        'composite': composite,
                        'b1': b1,
                        'b2': None if b2 == -1 else b2,  # -1 means GMP-ECM default, submit as None
                        'curves_requested': curve_count,
                        'curves_completed': stage2_curves,
                        'factors_found': stage2_all_factors if stage2_all_factors else [],
                        'factor_found': stage2_factor,
                        'sigma': stage2_sigma,
                        'raw_output': f"Stage 2 from residue {current_residue_id}",
                        'method': 'ecm',
                        'parametrization': residue_work.get('parametrization', 3),
                        'execution_time': stage2_time,
                    }

                    # Submit stage 2 results
                    print("Submitting stage 2 results...")
                    program_name = 'gmp-ecm-ecm'
                    submit_response = wrapper.submit_result(results, args.project, program_name)

                    if not submit_response:
                        wrapper.logger.error("Failed to submit stage 2 results")
                        wrapper.api_client.abandon_residue(client_id, current_residue_id)
                        current_residue_id = None
                        if local_residue_file.exists():
                            local_residue_file.unlink()
                        continue

                    # Extract attempt_id from response
                    stage2_attempt_id = submit_response.get('attempt_id')
                    if stage2_attempt_id:
                        print(f"Stage 2 attempt ID: {stage2_attempt_id}")
                    else:
                        wrapper.logger.error("No attempt_id returned from submit")
                        wrapper.api_client.abandon_residue(client_id, current_residue_id)
                        current_residue_id = None
                        if local_residue_file.exists():
                            local_residue_file.unlink()
                        continue

                    # Complete residue (supersedes stage 1, deletes server file)
                    print("Completing residue work...")
                    complete_result = wrapper.api_client.complete_residue(
                        client_id=client_id,
                        residue_id=current_residue_id,
                        stage2_attempt_id=stage2_attempt_id
                    )

                    if complete_result:
                        new_t_level = complete_result.get('new_t_level')
                        if new_t_level is not None:
                            print(f"T-level updated to {new_t_level:.2f}")
                    else:
                        wrapper.logger.warning("Failed to complete residue on server")

                    # Clean up local residue file
                    if local_residue_file.exists():
                        local_residue_file.unlink()
                        wrapper.logger.info(f"Deleted local residue file: {local_residue_file}")

                    current_residue_id = None
                    completed_count += 1
                    consecutive_failures = 0  # Reset on success

                    if print_work_status("Stage 2", completed_count, work_count_limit):
                        break

                except Exception as e:
                    consecutive_failures += 1

                    # Abandon residue work
                    if current_residue_id:
                        wrapper.api_client.abandon_residue(client_id, current_residue_id)
                        current_residue_id = None

                    # Clean up local residue file
                    if local_residue_file and local_residue_file.exists():
                        local_residue_file.unlink()

                    # Check circuit breaker
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        wrapper.logger.error(
                            f"Too many consecutive failures ({consecutive_failures}), exiting..."
                        )
                        break

                    # Check if work limit reached
                    if check_work_limit_reached(completed_count, work_count_limit):
                        break

        except KeyboardInterrupt:
            handle_shutdown(
                wrapper=wrapper,
                current_work_id=None,
                current_residue_id=current_residue_id,
                mode_name="Stage 2 Consumer mode",
                completed_count=completed_count,
                local_residue_file=local_residue_file
            )

        # Exit after stage2-work mode completes to avoid falling through to standard mode
        sys.exit(0)

    # Standard auto-work mode
    try:
        while not wrapper.interrupted:
            # Request work from server
            work = request_ecm_work(wrapper.api_client, client_id, args, wrapper.logger)

            if not work:
                continue

            # Store current work ID for cleanup on interrupt
            current_work_id = work['work_id']
            composite = work['composite']
            digit_length = work['digit_length']

            print_work_header(
                work_id=current_work_id,
                composite=composite,
                digit_length=digit_length,
                params={
                    'T-level': f"{work.get('current_t_level', 0):.1f} → {work.get('target_t_level', 0):.1f}"
                }
            )

            # Execute ECM - determine mode from parameters
            try:
                has_b1_b2 = args.b1 is not None and args.b2 is not None
                has_client_tlevel = hasattr(args, 'tlevel') and args.tlevel is not None

                # Determine execution mode
                if has_client_tlevel or (not has_b1_b2 and not has_client_tlevel):
                    # T-level mode (client-specified or server default)
                    target_tlevel = args.tlevel if has_client_tlevel else work.get('target_t_level', 35.0)

                    # Start from user-specified level, server's current level, or 0
                    if hasattr(args, 'start_tlevel') and args.start_tlevel is not None:
                        start_tlevel = args.start_tlevel
                    else:
                        start_tlevel = work.get('current_t_level', 0.0)

                    mode_desc = "client t-level" if has_client_tlevel else "server t-level"
                    print(f"Mode: {mode_desc} (start: {start_tlevel:.1f}, target: {target_tlevel:.1f})")

                    # Resolve worker count for multiprocess
                    workers = resolve_worker_count(args) if args.multiprocess else 1

                    # T-level mode (v2 API) - submits after each step internally
                    config = TLevelConfig(
                        composite=composite,
                        target_t_level=target_tlevel,
                        threads=workers,
                        verbose=args.verbose,
                        project=args.project,
                        no_submit=False,
                        work_id=current_work_id
                    )
                    result = wrapper.run_tlevel_v2(config)

                    # Note: Batches were submitted individually during execution
                    # Just need to track results for factor detection
                    results = result.to_dict(composite, args.method)

                else:
                    # B1/B2 mode with optional two-stage or multiprocess
                    b1 = args.b1
                    b2 = args.b2
                    curves = args.curves if args.curves else (1 if args.two_stage else wrapper.config['programs']['gmp_ecm']['default_curves'])

                    # Common parameters
                    use_gpu, gpu_device, gpu_curves = resolve_gpu_settings(args, wrapper.config)
                    sigma = parse_sigma_arg(args)
                    param = resolve_param(args, use_gpu)
                    continue_after_factor = args.continue_after_factor if hasattr(args, 'continue_after_factor') else False

                    if args.two_stage and args.method == 'ecm':
                        # Two-stage mode (v2 API)
                        print(f"Mode: two-stage GPU+CPU (B1={b1}, B2={b2}, curves={curves})")
                        stage2_workers = resolve_stage2_workers(args, wrapper.config)

                        config = TwoStageConfig(
                            composite=composite,
                            b1=b1,
                            b2=b2,
                            stage1_curves=curves,
                            stage1_device="GPU" if use_gpu else "CPU",
                            stage2_device="CPU",
                            stage1_parametrization=param if param else 3,
                            threads=stage2_workers,
                            verbose=args.verbose
                        )
                        result = wrapper.run_two_stage_v2(config)

                        # Convert FactorResult to dict
                        results = result.to_dict(composite, args.method)

                    elif args.multiprocess:
                        # Multiprocess mode (v2 API)
                        workers = resolve_worker_count(args)
                        print(f"Mode: multiprocess (B1={b1}, B2={b2}, curves={curves}, workers={workers})")

                        config = MultiprocessConfig(
                            composite=composite,
                            b1=b1,
                            b2=b2,
                            total_curves=curves,
                            num_processes=workers,
                            parametrization=param if param else 3,
                            method=args.method,
                            verbose=args.verbose
                        )
                        result = wrapper.run_multiprocess_v2(config)

                        # Convert FactorResult to dict
                        results = result.to_dict(composite, args.method)

                    else:
                        # Standard mode (v2 API)
                        print(f"Mode: standard (B1={b1}, B2={b2}, curves={curves})")

                        config = ECMConfig(
                            composite=composite,
                            b1=b1,
                            b2=b2,
                            curves=curves,
                            sigma=sigma,
                            parametrization=param if param else 3,
                            method=args.method,
                            verbose=args.verbose
                        )
                        result = wrapper.run_ecm_v2(config)

                        # Convert FactorResult to dict
                        results = result.to_dict(composite, args.method)

                    # Submit results for B1/B2 modes
                    if results.get('curves_completed', 0) > 0:
                        # Include work_id for failed submission recovery
                        results['work_id'] = current_work_id
                        program_name = f'gmp-ecm-{results.get("method", "ecm")}'
                        submit_response = wrapper.submit_result(results, args.project, program_name)

                        if not submit_response:
                            wrapper.logger.error("Failed to submit results, abandoning work assignment")
                            wrapper.abandon_work(current_work_id, reason="submission_failed")
                            current_work_id = None
                            continue

                # Mark work as complete
                wrapper.api_client.complete_work(current_work_id, client_id)
                current_work_id = None
                completed_count += 1

                # Check if we've reached the work count limit
                if print_work_status("Work assignment completed successfully", completed_count, work_count_limit):
                    break

            except Exception as e:
                wrapper.logger.exception(f"Error processing work assignment: {e}")
                if current_work_id:
                    wrapper.abandon_work(current_work_id, reason="execution_error")
                    current_work_id = None

    except KeyboardInterrupt:
        handle_shutdown(
            wrapper=wrapper,
            current_work_id=current_work_id,
            current_residue_id=None,
            mode_name="Auto-work mode",
            completed_count=completed_count
        )

    # Exit after auto-work mode completes to avoid falling through to standard mode
    sys.exit(0)
