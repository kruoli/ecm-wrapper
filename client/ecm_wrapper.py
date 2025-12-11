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
from pathlib import Path
from typing import Optional

from lib.ecm_executor import ECMWrapper
from lib.ecm_config import ECMConfig, TwoStageConfig, MultiprocessConfig, TLevelConfig
from lib.arg_parser import create_ecm_parser, resolve_gpu_settings, get_workers_default, parse_int_with_scientific
from lib.stage1_helpers import submit_stage1_complete_workflow
from lib.results_builder import results_for_stage1
from lib.user_output import UserOutput


def main():
    """Main entry point for local/manual ECM factorization."""
    # Use existing ECM parser from lib/arg_parser.py
    parser = create_ecm_parser()
    args = parser.parse_args()

    # Initialize user output handler
    output = UserOutput()

    # Require --composite for local mode (not needed for auto-work which uses ecm_client.py)
    if not args.composite:
        output.error("--composite is required for local/manual factorization")
        output.info("For server-coordinated work, use ecm_client.py instead")
        sys.exit(1)

    # Initialize wrapper (this loads and merges client.yaml + client.local.yaml)
    wrapper = ECMWrapper(args.config)

    # Resolve GPU settings from args + config (uses existing helper)
    use_gpu, gpu_device, gpu_curves = resolve_gpu_settings(args, wrapper.config)

    # Get workers default from config
    workers = args.workers if args.workers else get_workers_default(wrapper.config)

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

        output.mode_header("Stage 2 Only Mode", {
            "Residue": args.stage2_only,
            "Composite": args.composite,
            "B2": args.b2,
            "Workers": workers
        })

        # Parse residue file for B1
        residue_info = wrapper._parse_residue_file(residue_path)
        b1 = residue_info.get('b1', 0)
        if b1 == 0:
            output.warning("Could not parse B1 from residue file, using 0")

        # Run stage 2 using multithread executor
        from lib.ecm_config import FactorResult
        factor, all_factors, curves_completed, exec_time, sigma = wrapper._run_stage2_multithread(
            residue_file=residue_path,
            b1=b1,
            b2=args.b2,
            workers=workers,
            verbose=args.verbose or False
        )

        # Build FactorResult
        result = FactorResult()
        result.success = True
        result.curves_run = curves_completed
        result.execution_time = exec_time
        if all_factors:
            for f in all_factors:
                result.add_factor(f, sigma)

    # Stage 1 Only Mode - save residue to local file (optionally upload with --upload)
    elif args.stage1_only:
        if not args.b1:
            output.error("--stage1-only requires --b1")
            sys.exit(1)

        import time
        from pathlib import Path
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

    # T-level Mode
    elif args.tlevel:
        output.mode_header("T-level Mode", {
            "Target": f"t{args.tlevel}",
            "Composite": args.composite
        })

        config = TLevelConfig(
            composite=args.composite,
            target_t_level=args.tlevel,
            start_t_level=args.start_tlevel or 0.0,
            b1_strategy='optimal',
            parametrization=args.param or (3 if args.two_stage else 1),
            threads=args.workers or 1,
            verbose=args.verbose or False,
            workers=args.workers or 1,
            use_two_stage=args.two_stage or False,
            progress_interval=args.progress_interval or 0,
            project=args.project,
            no_submit=args.no_submit or False
        )

        result = wrapper.run_tlevel_v2(config)

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
    if result and not args.no_submit:
        results_dict = result.to_dict(args.composite, method)

        # Add ECM parameters that aren't in FactorResult
        # b1 and method resolved at top of function for all modes
        results_dict['b1'] = b1
        results_dict['b2'] = args.b2
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

    # Print summary
    if result:
        output.result_summary(result.curves_run, result.execution_time, result.factors)
        sys.exit(0 if result.success else 1)
    else:
        output.error("No result returned from execution")
        sys.exit(1)


if __name__ == '__main__':
    main()
