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

if sys.version_info < (3, 9):
    print(f"Error: Python 3.9+ is required (you have Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})")
    sys.exit(1)

from lib.ecm_executor import ECMWrapper
from lib.ecm_modes import (
    ResolvedParams,
    run_stage2_only_mode,
    run_stage1_only_mode,
    run_tlevel_mode,
    run_multiprocess_mode,
    run_two_stage_mode,
    run_standard_mode,
    submit_ecm_result,
)
from lib.arg_parser import create_ecm_parser, resolve_gpu_settings, get_workers_default, get_max_batch_default, parse_int_with_scientific, validate_ecm_args, load_b2_dictionary
from lib.user_output import UserOutput


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

    # Validate argument combinations before doing any real work
    validation_errors = validate_ecm_args(args)
    if validation_errors:
        for msg in validation_errors.values():
            output.error(msg)
        sys.exit(1)

    # Initialize wrapper (this loads and merges client.yaml + client.local.yaml)
    wrapper = ECMWrapper(args.config)

    # Resolve GPU settings from args + config (uses existing helper)
    use_gpu, gpu_device, gpu_curves = resolve_gpu_settings(args, wrapper.config)

    # Get workers default from config
    workers = args.workers if args.workers else get_workers_default(wrapper.config)

    # Get max_batch default from config (for two-stage GPU batching)
    max_batch = getattr(args, 'max_batch', None) or get_max_batch_default(wrapper.config)

    # Load B2 dictionary if specified
    b2_dictionary = None
    if getattr(args, 'b2_dictionary', None):
        b2_dictionary = load_b2_dictionary(args.b2_dictionary)

    # Resolve B1 from args or config based on method
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

    params = ResolvedParams(
        b1=b1,
        method=method,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
        gpu_curves=gpu_curves,
        workers=workers,
        max_batch=max_batch,
        b2_dictionary=b2_dictionary,
    )

    # Dispatch to mode handler
    result = None
    composite = args.composite

    if args.stage2_only:
        result, composite = run_stage2_only_mode(wrapper, args, output, params)
    elif args.stage1_only:
        result = run_stage1_only_mode(wrapper, args, output, params)
    elif args.tlevel is not None:
        result = run_tlevel_mode(wrapper, args, output, params)
    elif args.multiprocess:
        result = run_multiprocess_mode(wrapper, args, output, params)
    elif args.two_stage:
        result = run_two_stage_mode(wrapper, args, output, params)
    else:
        result = run_standard_mode(wrapper, args, output, params)

    # Submit results if available
    submit_ecm_result(wrapper, args, output, params, result, composite)

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
