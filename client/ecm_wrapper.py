#!/usr/bin/env python3
"""
ECM Wrapper - Local/Manual Factorization Modes

This script provides local factorization modes that require explicit composite input.
For server-coordinated work (auto-work), use ecm_client.py instead.

Modes:
  - Standard ECM: Basic factorization with B1/B2 bounds
  - Two-stage: GPU stage 1 + CPU stage 2 pipeline
  - Multiprocess: Parallel ECM execution across CPU cores
  - T-level: Target-based progressive factorization
  - Stage 1 only: Save residue to local file (optionally upload with --upload)
  - Stage 2 only: Load residue from local file
"""

import sys
import signal
from pathlib import Path
from typing import Optional

from lib.ecm_executor import ECMWrapper
from lib.ecm_config import ECMConfig, TwoStageConfig, MultiprocessConfig, TLevelConfig, FactorResult
from lib.arg_parser import create_ecm_parser, resolve_gpu_settings, get_workers_default, get_max_batch_default, parse_int_with_scientific
from lib.stage1_helpers import submit_stage1_complete_workflow
from lib.results_builder import results_for_stage1
from lib.user_output import UserOutput
from lib.ecm_math import calculate_target_tlevel, is_probably_prime


def main():
    """Main entry point for local/manual ECM factorization."""
    # Use existing ECM parser from lib/arg_parser.py
    parser = create_ecm_parser()
    args = parser.parse_args()

    # Initialize user output handler
    output = UserOutput()

    # Require --composite for local mode, except for --stage2-only
    # (--stage2-only extracts composite from residue file)
    if not args.composite and not args.stage2_only:
        output.error("--composite is required for local/manual factorization")
        output.info("For server-coordinated work, use ecm_client.py instead")
        sys.exit(1)

    # Initialize wrapper (this loads and merges client.yaml + client.local.yaml)
    wrapper = ECMWrapper(args.config)

    # Resolve GPU settings from args + config (uses existing helper)
    use_gpu, gpu_device, gpu_curves = resolve_gpu_settings(args, wrapper.config)

    # Get workers default from config
    workers = args.workers if args.workers else get_workers_default(wrapper.config)

    # Get max_batch default from config (for two-stage GPU batching)
    max_batch = getattr(args, 'max_batch', None) or get_max_batch_default(wrapper.config)

    # Resolve B1 from args or config based on method
    # This provides a sensible default when --b1 is not specified
    # Config values may be strings with scientific notation (e.g., "25e9")
    def get_b1_from_config(key: str, default: int) -> int:
        val = wrapper.config['programs']['gmp_ecm'].get(key, default)
        if isinstance(val, str):
            return parse_int_with_scientific(val)
        return val

    method = args.method or 'ecm'
    if args.b1:
        b1 = args.b1
    elif method == 'pm1':
        b1 = get_b1_from_config('pm1_b1', 2900000000)
    elif method == 'pp1':
        b1 = get_b1_from_config('pp1_b1', 110000000)
    else:
        b1 = get_b1_from_config('default_b1', 110000000)

    # Detect mode and execute
    result = None
    # Track composite for submission - may come from args or residue file
    composite = args.composite

    # Stage 2 Only Mode - load residue from local file
    # Note: --stage2-only contains the residue file path
    if args.stage2_only:
        if not args.b2:
            output.error("--stage2-only requires --b2")
            sys.exit(1)

        residue_path = Path(args.stage2_only)
        if not residue_path.exists():
            output.error(f"Residue file not found: {args.stage2_only}")
            sys.exit(1)

        # Parse residue file for B1 and composite
        residue_info = wrapper._parse_residue_file(residue_path)
        b1 = residue_info.get('b1', 0)
        composite_from_residue = residue_info.get('composite', 'unknown')
        composite = composite_from_residue  # Use for submission

        output.mode_header("Stage 2 Only Mode", {
            "Residue": args.stage2_only,
            "Composite": composite_from_residue[:40] + "..." if len(composite_from_residue) > 40 else composite_from_residue,
            "B2": args.b2,
            "Workers": workers
        })

        if b1 == 0:
            output.warning("Could not parse B1 from residue file, using 0")

        # Install graceful shutdown signal handler
        def graceful_sigint_handler(signum, frame):
            """Handle Ctrl+C gracefully: first time finish current work, second time abort."""
            if not wrapper.graceful_shutdown_requested:
                wrapper.graceful_shutdown_requested = True
                print("\n[Ctrl+C] Graceful shutdown initiated - completing current curves...")
                print("[Ctrl+C] Press Ctrl+C again to abort immediately")
            else:
                print("\n[Ctrl+C] Immediate abort requested")
                # Restore original handler and re-raise to trigger immediate abort
                if wrapper._original_sigint_handler:
                    signal.signal(signal.SIGINT, wrapper._original_sigint_handler)
                raise KeyboardInterrupt()

        wrapper._original_sigint_handler = signal.signal(signal.SIGINT, graceful_sigint_handler)

        try:
            # Run stage 2 using multithread executor
            factor, all_factors, curves_completed, exec_time, sigma = wrapper._run_stage2_multithread(
                residue_file=residue_path,
                b1=b1,
                b2=args.b2,
                workers=workers,
                verbose=args.verbose or False,
                progress_interval=args.progress_interval or 0
            )

            # Build FactorResult
            result = FactorResult()
            result.success = True
            result.curves_run = curves_completed
            result.execution_time = exec_time
            if all_factors:
                for f in all_factors:
                    result.add_factor(f, sigma)

            # Mark as interrupted if graceful shutdown occurred
            if wrapper.graceful_shutdown_requested:
                result.interrupted = True
                output.info(f"Graceful shutdown completed - processed {curves_completed} curves")
        finally:
            # Restore original signal handler
            if wrapper._original_sigint_handler:
                signal.signal(signal.SIGINT, wrapper._original_sigint_handler)
            wrapper.graceful_shutdown_requested = False

    # Stage 1 Only Mode - save residue to local file (optionally upload with --upload)
    elif args.stage1_only:
        if not args.b1:
            output.error("--stage1-only requires --b1")
            sys.exit(1)

        import time
        residue_dir = Path("data/residues")
        residue_dir.mkdir(parents=True, exist_ok=True)
        save_path = args.save_residues or str(residue_dir / f"residue_{hash(args.composite) % 100000}_{int(time.time())}.txt")

        output.mode_header("Stage 1 Only Mode", {
            "Save to": save_path,
            "Composite": args.composite,
            "B1": b1,
            "Curves": args.curves or 1
        })
        if args.upload:
            output.item("Upload", "Will upload residue to server after completion")

        # Create config for stage 1 execution
        param = args.param or (3 if use_gpu else 1)
        config = ECMConfig(
            composite=args.composite,
            b1=b1,
            b2=0,  # Stage 1 only
            curves=args.curves or 1,
            sigma=args.sigma,
            parametrization=param,
            threads=1,
            verbose=args.verbose or False,
            save_residues=save_path,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            gpu_curves=gpu_curves,
            method=method,
            progress_interval=args.progress_interval or 0
        )

        result = wrapper.run_ecm_v2(config)

        if result.success or result.curves_run > 0:
            output.section("Stage 1 complete:")
            output.item("Residue saved", save_path)
            output.item("Curves run", result.curves_run)
            if result.factors:
                output.item("Factors found", result.factors)

            # Upload residue to server if --upload flag is set
            if args.upload and not args.no_submit:
                residue_path = Path(save_path)
                if residue_path.exists():
                    # Get client_id from config
                    client_id = wrapper.config.get('client', {}).get('username', 'unknown')

                    # Build factor info for upload
                    factor_found = result.factors[0] if result.factors else None
                    all_factors = [(f, result.sigmas[i] if i < len(result.sigmas) else None)
                                   for i, f in enumerate(result.factors)]

                    # Build results dict for submission
                    results = (results_for_stage1(args.composite, b1, result.curves_run, param)
                        .with_curves(result.curves_run, result.curves_run)
                        .with_factors(all_factors)
                        .with_execution_time(result.execution_time)
                        .add_raw_output(result.raw_output or "")
                        .build())

                    # Use the consolidated workflow to submit + upload
                    attempt_id = submit_stage1_complete_workflow(
                        wrapper=wrapper,
                        results=results,
                        residue_file=residue_path,
                        work_id=None,  # No work assignment for manual mode
                        project=args.project,
                        client_id=client_id,
                        factor_found=factor_found,
                        cleanup_residue=False  # Keep local copy
                    )

                    if attempt_id:
                        output.item("Residue uploaded", f"attempt_id: {attempt_id}")
                    else:
                        output.warning("Failed to upload residue to server")
                else:
                    output.warning(f"Residue file not found at {save_path}")

            # Mark that we've already submitted if --upload was used
            if args.upload:
                args.no_submit = True  # Prevent double submission

    # T-level Mode (including progressive factorization when --tlevel given without value)
    elif args.tlevel is not None:
        # Determine if we're in progressive mode (auto t-level calculation)
        is_progressive = args.tlevel < 0  # -1.0 sentinel means auto-calculate

        # Current state for progressive factorization
        current_composite = args.composite
        current_t_level = args.start_tlevel or 0.0
        all_factors = []

        # Progressive factorization loop
        while True:
            digit_length = len(current_composite)

            # Calculate or use explicit target t-level
            if is_progressive:
                target_t_level = calculate_target_tlevel(digit_length)
            else:
                target_t_level = args.tlevel

            # Skip if we've already exceeded the target
            if current_t_level >= target_t_level:
                output.info(f"Already at t{current_t_level:.2f} >= target t{target_t_level:.1f}")
                break

            mode_name = "Progressive T-level Mode" if is_progressive else "T-level Mode"
            output.mode_header(mode_name, {
                "Target": f"t{target_t_level:.1f}",
                "Current": f"t{current_t_level:.2f}" if current_t_level > 0 else "t0",
                "Composite": f"C{digit_length} ({current_composite[:20]}...)" if len(current_composite) > 25 else f"C{digit_length}"
            })

            config = TLevelConfig(
                composite=current_composite,
                target_t_level=target_t_level,
                start_t_level=current_t_level,
                b1_strategy='optimal',
                parametrization=args.param or (3 if args.two_stage else 1),
                threads=args.workers or 1,
                verbose=args.verbose or False,
                workers=args.workers or 1,
                use_two_stage=args.two_stage or False,
                progress_interval=args.progress_interval or 0,
                max_batch_curves=max_batch,
                b2_multiplier=getattr(args, 'b2_multiplier', None) or 100.0,
                project=args.project,
                no_submit=args.no_submit or False
            )

            result = wrapper.run_tlevel_v2(config)

            # Collect factors
            if result.factors:
                all_factors.extend(result.factors)
                output.success(f"Found {len(result.factors)} factor(s): {', '.join(result.factors[:3])}{'...' if len(result.factors) > 3 else ''}")

                # Update composite by dividing out factors
                composite_int = int(current_composite)
                for factor in result.factors:
                    factor_int = int(factor)
                    while composite_int % factor_int == 0:
                        composite_int //= factor_int

                # Check if fully factored
                if composite_int == 1:
                    output.success("Fully factored!")
                    break

                # Check if remaining cofactor is prime
                if is_probably_prime(composite_int):
                    output.success(f"Cofactor C{len(str(composite_int))} is prime - factorization complete!")
                    all_factors.append(str(composite_int))
                    break

                # Continue with cofactor in progressive mode
                if is_progressive:
                    current_composite = str(composite_int)
                    # T-level achieved carries over to cofactor
                    current_t_level = result.t_level_achieved if result.t_level_achieved else 0.0
                    output.info(f"Continuing with cofactor C{len(current_composite)} from t{current_t_level:.2f}")
                else:
                    # Explicit t-level mode: stop after finding factors
                    break
            else:
                # No factors found - we've reached the target t-level
                if is_progressive:
                    output.info(f"Reached t{target_t_level:.1f} with no factor found")
                break

            # Check for interrupt
            if result.interrupted:
                output.warning("Interrupted by user")
                break

        # Build final result with all collected factors
        if all_factors:
            # Create aggregate result
            result = FactorResult()
            for f in all_factors:
                result.add_factor(f, None)
            result.success = True

    # Multiprocess Mode
    elif args.multiprocess:
        output.mode_header("Multiprocess Mode", {
            "Workers": args.workers or "auto",
            "Composite": args.composite,
            "B1": b1,
            "B2": args.b2 or "default"
        })

        config = MultiprocessConfig(
            composite=args.composite,
            b1=b1,
            b2=args.b2,
            total_curves=args.curves or 1000,
            curves_per_process=100,
            num_processes=args.workers,
            parametrization=args.param or 1,
            method=args.method or 'ecm',
            verbose=args.verbose or False,
            continue_after_factor=False,
            progress_interval=args.progress_interval or 0
        )

        result = wrapper.run_multiprocess_v2(config)

    # Two-stage Mode
    elif args.two_stage:
        # Two-stage mode always uses GPU for stage 1 (that's the whole point)
        output.mode_header("Two-stage Mode", {
            "Pipeline": "GPU stage 1 + CPU stage 2",
            "Composite": args.composite,
            "B1": b1,
            "B2": args.b2 if args.b2 is not None else "default"
        })

        config = TwoStageConfig(
            composite=args.composite,
            b1=b1,
            b2=args.b2,
            stage1_curves=args.curves or 100,  # Use --curves for stage 1
            stage2_curves_per_residue=1000,     # Default for stage 2
            stage1_device="GPU",  # Two-stage always uses GPU for stage 1
            stage2_device="CPU",
            stage1_parametrization=args.param or 3,
            stage2_parametrization=1,
            threads=workers,
            verbose=args.verbose or False,
            save_residues=args.save_residues,
            gpu_device=gpu_device,
            gpu_curves=gpu_curves,
            continue_after_factor=False,
            progress_interval=args.progress_interval or 0,
            project=args.project,
            no_submit=args.no_submit or False
        )

        result = wrapper.run_two_stage_v2(config)

    # Standard Mode
    else:
        # B1 and method already resolved at top of function
        output.mode_header("Standard ECM Mode", {
            "Composite": args.composite,
            "B1": b1,
            "B2": args.b2 if args.b2 is not None else "default",
            "Curves": args.curves or 1
        })

        config = ECMConfig(
            composite=args.composite,
            b1=b1,
            b2=args.b2,
            curves=args.curves or 1,
            sigma=args.sigma,
            parametrization=args.param or (3 if use_gpu else 1),
            threads=1,
            verbose=args.verbose or False,
            save_residues=args.save_residues,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            gpu_curves=gpu_curves,
            method=method,
            progress_interval=args.progress_interval or 0
        )

        result = wrapper.run_ecm_v2(config)

    # Submit results if available
    # Note: T-level mode and two-stage mode handle their own submissions internally,
    # so we skip post-execution submission for those modes to avoid double submission
    mode_handles_own_submission = args.tlevel or args.two_stage
    if result and not args.no_submit and not mode_handles_own_submission:
        results_dict = result.to_dict(composite, method)

        # Add ECM parameters that aren't in FactorResult
        # b1 and method resolved at top of function for all modes
        results_dict['b1'] = b1
        # Stage1-only mode should always submit b2=0, not None
        results_dict['b2'] = 0 if getattr(args, 'stage1_only', False) else args.b2
        results_dict['curves_requested'] = args.curves
        results_dict['parametrization'] = args.param or (3 if getattr(args, 'gpu', False) else 1)

        # Add project if specified
        if args.project:
            results_dict['project'] = args.project

        # Submit via API
        if result.success and result.factors:
            output.info(f"\nSubmitting {len(result.factors)} factor(s) to API...")
            wrapper.submit_result(results_dict, args.project, f"gmp-ecm-{method}")
        elif result.curves_run > 0:
            output.info(f"\nSubmitting {result.curves_run} curves (no factors) to API...")
            wrapper.submit_result(results_dict, args.project, f"gmp-ecm-{method}")

    # Print curve summary for t-level runs
    if result and result.curve_summary:
        result.print_curve_summary(show_parametrization=args.verbose)

    # Print summary
    if result:
        output.result_summary(result.curves_run, result.execution_time, result.factors)
        sys.exit(0 if result.success else 1)
    else:
        output.error("No result returned from execution")
        sys.exit(1)


if __name__ == '__main__':
    main()
