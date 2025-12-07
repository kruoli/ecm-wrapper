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
  - Stage 1 only: Save residue to local file
  - Stage 2 only: Load residue from local file
"""

import sys
from pathlib import Path
from typing import Optional

from lib.ecm_executor import ECMWrapper
from lib.ecm_config import ECMConfig, TwoStageConfig, MultiprocessConfig, TLevelConfig
from lib.arg_parser import create_parser, apply_config_defaults


def create_wrapper_parser():
    """Create argument parser for ecm_wrapper.py (local/manual modes)."""
    import argparse

    parser = argparse.ArgumentParser(
        description='ECM Wrapper - Local factorization with manual composite input',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard ECM
  python3 ecm_wrapper.py --composite "123456789012345" --b1 50000 --curves 100

  # Two-stage (GPU + CPU)
  python3 ecm_wrapper.py --composite "123456789012345" --two-stage --b1 110000000 --curves 100

  # Multiprocess
  python3 ecm_wrapper.py --composite "123456789012345" --multiprocess --workers 8 --b1 50000

  # T-level targeting
  python3 ecm_wrapper.py --composite "123456789012345" --tlevel 35

  # Stage 1 only - save residue to local file
  python3 ecm_wrapper.py --composite "123456789012345" --stage1-only --b1 110000000 --curves 3000

  # Stage 2 only - load residue from local file
  python3 ecm_wrapper.py --stage2-only --residue-file /path/to/residue.txt --b2 11000000000000
"""
    )

    # Required composite argument
    parser.add_argument('--composite', type=str, required=True,
                       help='Composite number to factor (required)')

    # ECM parameters
    parser.add_argument('--b1', type=str,
                       help='Stage 1 bound (supports scientific notation: 26e7)')
    parser.add_argument('--b2', type=str,
                       help='Stage 2 bound (supports scientific notation: 4e11, -1 for GMP-ECM default)')
    parser.add_argument('--curves', type=int,
                       help='Number of curves to run')
    parser.add_argument('--sigma', type=str,
                       help='Sigma value(s) for ECM (comma-separated for multiple)')

    # Execution modes (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--two-stage', action='store_true',
                           help='Two-stage mode: GPU stage 1 + CPU stage 2')
    mode_group.add_argument('--multiprocess', action='store_true',
                           help='Multiprocess mode: parallel ECM execution')
    mode_group.add_argument('--tlevel', type=float,
                           help='Target t-level for progressive factorization')
    mode_group.add_argument('--stage1-only', action='store_true',
                           help='Stage 1 only: save residue to local file')
    mode_group.add_argument('--stage2-only', action='store_true',
                           help='Stage 2 only: load residue from local file')

    # Stage 2 only mode
    parser.add_argument('--residue-file', type=str,
                       help='Path to residue file for --stage2-only mode')

    # Two-stage parameters
    parser.add_argument('--stage1-curves', type=int,
                       help='Stage 1 curve count (two-stage mode)')
    parser.add_argument('--stage2-curves', type=int,
                       help='Stage 2 curves per residue (two-stage mode)')
    parser.add_argument('--save-residues', type=str,
                       help='Path to save residue file')

    # Multiprocess parameters
    parser.add_argument('--workers', type=int,
                       help='Number of worker processes (multiprocess mode)')

    # GPU parameters
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration')
    parser.add_argument('--gpu-device', type=int,
                       help='GPU device ID')
    parser.add_argument('--gpu-curves', type=int,
                       help='Curves per GPU batch')

    # Method selection
    parser.add_argument('--method', type=str, choices=['ecm', 'pm1', 'pp1'],
                       help='Factorization method (default: ecm)')
    parser.add_argument('--parametrization', type=int, choices=[0, 1, 2, 3],
                       help='ECM parametrization (0-3)')

    # Execution control
    parser.add_argument('--threads', type=int,
                       help='Number of threads for stage 2')
    parser.add_argument('--stage2-workers', type=int,
                       help='Number of stage 2 worker threads (two-stage mode)')
    parser.add_argument('--timeout', type=int,
                       help='Execution timeout in seconds')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--progress', type=int,
                       help='Progress reporting interval')

    # API submission
    parser.add_argument('--no-submit', action='store_true',
                       help='Skip API submission of results')
    parser.add_argument('--project', type=str,
                       help='Project name for result submission')

    # Configuration
    parser.add_argument('--config', type=str, default='client.yaml',
                       help='Path to configuration file')

    return parser


def main():
    """Main entry point for local/manual ECM factorization."""
    parser = create_wrapper_parser()
    args = parser.parse_args()

    # Apply config defaults
    args = apply_config_defaults(args)

    # Initialize wrapper
    wrapper = ECMWrapper(args.config)

    # Detect mode and execute
    result = None

    # Stage 2 Only Mode - load residue from local file
    if args.stage2_only:
        if not args.residue_file:
            print("Error: --stage2-only requires --residue-file")
            sys.exit(1)
        if not args.b2:
            print("Error: --stage2-only requires --b2")
            sys.exit(1)

        residue_path = Path(args.residue_file)
        if not residue_path.exists():
            print(f"Error: Residue file not found: {args.residue_file}")
            sys.exit(1)

        print(f"Stage 2 Only Mode: Loading residue from {args.residue_file}")
        print(f"Composite: {args.composite}")
        print(f"B2: {args.b2}")
        print(f"Threads: {args.stage2_workers or args.threads or 1}")

        # Run stage 2 using the executor's internal method
        from lib.ecm_config import FactorResult
        result = wrapper._run_stage2_from_file(
            composite=args.composite,
            residue_file=str(residue_path),
            b1=None,  # Will be parsed from residue file
            b2=args.b2,
            num_workers=args.stage2_workers or args.threads or 1,
            verbose=args.verbose or False,
            timeout=args.timeout or 7200
        )

    # Stage 1 Only Mode - save residue to local file
    elif args.stage1_only:
        if not args.b1:
            print("Error: --stage1-only requires --b1")
            sys.exit(1)

        save_path = args.save_residues or f"data/residues/residue_{hash(args.composite) % 100000}_{int(Path(__file__).stat().st_mtime)}.txt"

        print(f"Stage 1 Only Mode: Saving residue to {save_path}")
        print(f"Composite: {args.composite}")
        print(f"B1: {args.b1}")
        print(f"Curves: {args.curves or 1}")

        # Create config for stage 1 execution
        config = ECMConfig(
            composite=args.composite,
            b1=args.b1,
            b2=0,  # Stage 1 only
            curves=args.curves or 1,
            sigma=args.sigma,
            parametrization=args.parametrization or (3 if args.gpu else 1),
            threads=args.threads or 1,
            verbose=args.verbose or False,
            timeout=args.timeout or 3600,
            save_residues=save_path,
            use_gpu=args.gpu or False,
            gpu_device=args.gpu_device,
            gpu_curves=args.gpu_curves,
            method=args.method or 'ecm',
            progress_interval=args.progress or 0
        )

        result = wrapper.run_ecm_v2(config)

        if result.success or result.curves_run > 0:
            print(f"\nStage 1 complete:")
            print(f"  Residue saved: {save_path}")
            print(f"  Curves run: {result.curves_run}")
            if result.factors:
                print(f"  Factors found: {result.factors}")

    # T-level Mode
    elif args.tlevel:
        print(f"T-level Mode: Targeting t-level {args.tlevel}")
        print(f"Composite: {args.composite}")

        config = TLevelConfig(
            composite=args.composite,
            target_t_level=args.tlevel,
            b1_strategy='optimal',
            parametrization=args.parametrization or (3 if args.two_stage else 1),
            threads=args.threads or args.workers or 1,
            verbose=args.verbose or False,
            timeout=args.timeout or 7200,
            workers=args.workers or 1,
            use_two_stage=args.two_stage or False,
            project=args.project,
            no_submit=args.no_submit or False
        )

        result = wrapper.run_tlevel_v2(config)

    # Multiprocess Mode
    elif args.multiprocess:
        print(f"Multiprocess Mode: {args.workers or 'auto'} workers")
        print(f"Composite: {args.composite}")
        print(f"B1: {args.b1}, B2: {args.b2 or 'default'}")

        config = MultiprocessConfig(
            composite=args.composite,
            b1=args.b1,
            b2=args.b2,
            total_curves=args.curves or 1000,
            curves_per_process=100,
            num_processes=args.workers,
            parametrization=args.parametrization or 1,
            method=args.method or 'ecm',
            verbose=args.verbose or False,
            timeout=args.timeout or 3600,
            continue_after_factor=False
        )

        result = wrapper.run_multiprocess_v2(config)

    # Two-stage Mode
    elif args.two_stage:
        print(f"Two-stage Mode: GPU stage 1 + CPU stage 2")
        print(f"Composite: {args.composite}")
        print(f"B1: {args.b1}, B2: {args.b2 or 'default'}")

        config = TwoStageConfig(
            composite=args.composite,
            b1=args.b1,
            b2=args.b2,
            stage1_curves=args.stage1_curves or args.curves or 100,
            stage2_curves_per_residue=args.stage2_curves or 1000,
            stage1_device="GPU" if args.gpu else "CPU",
            stage2_device="CPU",
            stage1_parametrization=args.parametrization or 3,
            stage2_parametrization=1,
            threads=args.stage2_workers or args.threads or 1,
            verbose=args.verbose or False,
            save_residues=args.save_residues,
            timeout_stage1=args.timeout or 3600,
            timeout_stage2=args.timeout or 7200,
            gpu_device=args.gpu_device,
            gpu_curves=args.gpu_curves,
            continue_after_factor=False,
            project=args.project,
            no_submit=args.no_submit or False
        )

        result = wrapper.run_two_stage_v2(config)

    # Standard Mode
    else:
        print(f"Standard ECM Mode")
        print(f"Composite: {args.composite}")
        print(f"B1: {args.b1}, B2: {args.b2 or 'default'}")
        print(f"Curves: {args.curves or 1}")

        config = ECMConfig(
            composite=args.composite,
            b1=args.b1,
            b2=args.b2,
            curves=args.curves or 1,
            sigma=args.sigma,
            parametrization=args.parametrization or (3 if args.gpu else 1),
            threads=args.threads or 1,
            verbose=args.verbose or False,
            timeout=args.timeout or 3600,
            save_residues=args.save_residues,
            use_gpu=args.gpu or False,
            gpu_device=args.gpu_device,
            gpu_curves=args.gpu_curves,
            method=args.method or 'ecm',
            progress_interval=args.progress or 0
        )

        result = wrapper.run_ecm_v2(config)

    # Submit results if available
    if result and not args.no_submit:
        results_dict = result.to_dict(args.composite, args.method or 'ecm')

        # Add project if specified
        if args.project:
            results_dict['project'] = args.project

        # Submit via API
        if result.success and result.factors:
            print(f"\nSubmitting {len(result.factors)} factor(s) to API...")
            wrapper.process_and_submit_results(results_dict)
        elif result.curves_run > 0:
            print(f"\nSubmitting {result.curves_run} curves (no factors) to API...")
            wrapper.process_and_submit_results(results_dict)

    # Print summary
    if result:
        print(f"\nExecution Summary:")
        print(f"  Curves completed: {result.curves_run}")
        print(f"  Execution time: {result.execution_time:.2f}s")
        if result.factors:
            print(f"  Factors found: {', '.join(result.factors)}")
        else:
            print(f"  No factors found")

        sys.exit(0 if result.success else 1)
    else:
        print("Error: No result returned from execution")
        sys.exit(1)


if __name__ == '__main__':
    main()
