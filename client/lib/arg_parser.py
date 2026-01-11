#!/usr/bin/env python3
"""
Shared argument parsing logic for ECM and YAFU wrappers.
"""
import argparse
import sys
import multiprocessing
from typing import Dict, Any, Optional


def parse_int_with_scientific(value: str) -> int:
    """
    Parse integer from string, supporting scientific notation.

    Examples:
        "1000000" -> 1000000
        "1e6" -> 1000000
        "26e7" -> 260000000
        "4e11" -> 400000000000
        "-1" -> -1 (special: GMP-ECM default for B2)

    Args:
        value: String representation of number

    Returns:
        Integer value

    Raises:
        argparse.ArgumentTypeError: If value cannot be parsed
    """
    try:
        # Convert through float to handle scientific notation, then to int
        result = int(float(value))
        # Allow -1 as a special sentinel value (GMP-ECM default for B2)
        if result < -1:
            raise argparse.ArgumentTypeError(f"Value must be -1 or positive: {value}")
        return result
    except (ValueError, OverflowError) as e:
        raise argparse.ArgumentTypeError(f"Invalid integer or scientific notation: {value}") from e


def create_ecm_parser() -> argparse.ArgumentParser:
    """Create argument parser for ECM wrapper."""
    parser = argparse.ArgumentParser(description='ECM Wrapper Client')

    # Configuration
    parser.add_argument('--config', default='client.yaml', help='Config file path')

    # Core parameters
    parser.add_argument('--composite', '-n', help='Number to factor (not required in --auto-work mode)')
    parser.add_argument('--b1', type=parse_int_with_scientific, help='B1 bound (supports scientific notation, e.g., 26e7)')
    parser.add_argument('--b2', type=parse_int_with_scientific, help='B2 bound (supports scientific notation, e.g., 4e11). Use -1 for GMP-ECM default, 0 for stage 1 only')
    parser.add_argument('--b2-multiplier', type=float, help='Dynamic B2 calculation: B2 = B1 * multiplier (e.g., 1000 for B2=1000*B1). Overridden by explicit --b2')
    parser.add_argument('--curves', '-c', type=int, help='Number of curves')
    parser.add_argument('--max-batch', type=int,
                       help='Max curves per GPU batch in two-stage t-level mode (enables chunking for earlier factor discovery)')
    parser.add_argument('--tlevel', '-t', type=float, nargs='?', const=-1.0,
                       help='Target t-level. If specified without a value, auto-calculates as 4/13 of digit length and runs progressively until factored.')
    parser.add_argument('--start-tlevel', type=float, help='Starting t-level (for resuming, requires --tlevel)')
    parser.add_argument('--project', '-p', help='Project name')
    parser.add_argument('--submit', action='store_true', help='Submit results to API')

    # Auto-work mode
    parser.add_argument('--auto-work', action='store_true',
                       help='Continuously request and process work assignments from server (uses server t-levels unless --b1/--b2 or --tlevel specified)')
    parser.add_argument('--work-count', type=int, help='Number of work assignments to complete before exiting (auto-work mode, default: unlimited)')
    parser.add_argument('--min-digits', type=int, help='Minimum composite digit length (auto-work mode)')
    parser.add_argument('--max-digits', type=int, help='Maximum composite digit length (auto-work mode)')
    parser.add_argument('--priority', type=int, help='Minimum priority filter (auto-work mode)')
    parser.add_argument('--work-type', choices=['standard', 'progressive'], default='standard',
                       help='Work assignment strategy: standard (smallest first) or progressive (least ECM done first, default: standard)')

    # Decoupled two-stage mode (stage 1 and stage 2 run separately)
    parser.add_argument('--stage1-only', action='store_true',
                       help='Run stage 1 only, submit results and upload residue file to server (GPU producer mode)')
    parser.add_argument('--upload', action='store_true',
                       help='Upload residue file to server after stage 1 (for --stage1-only mode)')

    # GPU options
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration (CGBN)')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--gpu-device', type=int, help='GPU device number to use')
    parser.add_argument('--gpu-curves', type=int, help='Number of curves to compute in parallel on GPU')

    # Sigma and parametrization for reproducibility
    parser.add_argument('--sigma', type=str, help='Specific sigma value to use (format: "N" or "3:N")')
    parser.add_argument('--param', type=int, choices=[0, 1, 2, 3], help='ECM parametrization (0-3)')

    # Method and verbosity
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose ECM output')
    parser.add_argument('--progress-interval', type=int, default=0,
                       help='Show progress updates every N completed curves (0 = disabled)')
    parser.add_argument('--method', choices=['ecm', 'pm1', 'pp1'], default='ecm',
                       help='Factorization method (ECM, P-1, P+1)')

    # Advanced modes
    parser.add_argument('--two-stage', action='store_true',
                       help='Use two-stage mode: GPU stage 1 + multi-threaded CPU stage 2')
    parser.add_argument('--multiprocess', action='store_true',
                       help='Use multi-process mode: parallel full ECM cycles (CPU-optimized)')
    parser.add_argument('--workers', type=int, default=0,
                       help='Number of parallel workers (processes for multiprocess, threads for stage2; default: CPU count)')

    # Residue file handling
    parser.add_argument('--save-residues', type=str, help='Save stage 1 residues with specified filename in configured residue_dir')
    parser.add_argument('--stage2-only', type=str, help='Run stage 2 only on residue file path')

    # Factor handling
    parser.add_argument('--continue-after-factor', action='store_true',
                       help='Continue processing all curves even after finding a factor')


    return parser


def create_client_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for ecm_client.py (server-coordinated modes).

    This parser is for server-coordinated work where composites and t-levels
    come from the server. For local/manual factorization, use create_ecm_parser().
    """
    parser = argparse.ArgumentParser(
        description='ECM Client - Server-coordinated factorization work',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-work with server defaults
  python3 ecm_client.py

  # Target a specific composite (server provides t-level info)
  python3 ecm_client.py --composite "123456789..."

  # Target composite with multiprocess
  python3 ecm_client.py --composite "123456789..." --multiprocess --workers 8

  # Process 10 work items with client-specified B1/B2
  python3 ecm_client.py --work-count 10 --b1 50000 --b2 5000000 --curves 100

  # Stage 1 only - upload residues to server
  python3 ecm_client.py --stage1-only --b1 110000000 --curves 3000

  # Stage 2 only - download and process residues
  python3 ecm_client.py --stage2-only --b2 11000000000000 --workers 8
"""
    )

    # Composite targeting
    parser.add_argument('--composite', type=str,
                       help='Target a specific composite (queries server for t-level status)')

    # Work filtering
    parser.add_argument('--work-count', type=int,
                       help='Number of work items to process (default: unlimited)')
    parser.add_argument('--min-target-tlevel', type=float,
                       help='Minimum target t-level (filter work by difficulty)')
    parser.add_argument('--max-target-tlevel', type=float,
                       help='Maximum target t-level (filter work by difficulty)')
    parser.add_argument('--priority', type=int,
                       help='Minimum priority level')
    parser.add_argument('--work-type', choices=['standard', 'progressive'], default='standard',
                       help='Work assignment strategy: standard (smallest first) or progressive (least ECM done first)')

    # Execution parameters (override server defaults)
    parser.add_argument('--tlevel', type=float,
                       help='Target t-level (overrides server t-level)')
    parser.add_argument('--b1', type=parse_int_with_scientific,
                       help='B1 parameter (overrides server default, supports scientific notation e.g., 52e6)')
    parser.add_argument('--b2', type=parse_int_with_scientific,
                       help='B2 parameter (overrides server default, -1 for GMP-ECM default, supports scientific notation)')
    parser.add_argument('--b2-multiplier', type=float,
                       help='Dynamic B2 = B1 * multiplier (for stage2-only mode)')
    parser.add_argument('--curves', type=int,
                       help='Curves per batch')
    parser.add_argument('--max-batch', type=int,
                       help='Max curves per GPU batch in two-stage t-level mode (enables chunking for earlier factor discovery)')
    parser.add_argument('--method', choices=['ecm', 'pm1', 'pp1'], default='ecm',
                       help='Factorization method (default: ecm)')

    # Execution modes
    parser.add_argument('--multiprocess', action='store_true',
                       help='Use multiprocess parallelization')
    parser.add_argument('--workers', type=int,
                       help='Number of parallel workers (processes for multiprocess, threads for stage2)')
    parser.add_argument('--two-stage', action='store_true',
                       help='Use two-stage GPU+CPU mode')

    # Decoupled two-stage modes (mutually exclusive)
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument('--stage1-only', action='store_true',
                            help='Stage 1 only: upload residue to server')
    stage_group.add_argument('--stage2-only', action='store_true',
                            help='Stage 2 only: download residue from server')

    # Stage 2 filtering (for --stage2-only mode)
    parser.add_argument('--min-b1', type=parse_int_with_scientific,
                       help='Minimum B1 filter for --stage2-only (supports scientific notation, e.g., 11e6)')
    parser.add_argument('--max-b1', type=parse_int_with_scientific,
                       help='Maximum B1 filter for --stage2-only (supports scientific notation, e.g., 26e7)')

    # GPU/compute
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU even if enabled in config')
    parser.add_argument('--gpu-device', type=int,
                       help='GPU device number')
    parser.add_argument('--gpu-curves', type=int,
                       help='Number of curves per GPU batch')
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


def create_yafu_parser() -> argparse.ArgumentParser:
    """Create argument parser for YAFU wrapper."""
    parser = argparse.ArgumentParser(description='YAFU Wrapper Client')

    # Configuration
    parser.add_argument('--config', default='client.yaml', help='Config file path')
    parser.add_argument('--composite', '-n', required=True, help='Number to factor')

    # Mode selection
    parser.add_argument('--mode', choices=['ecm', 'pm1', 'pp1', 'auto', 'siqs', 'nfs'],
                       default='ecm', help='Factorization mode')

    # ECM parameters
    parser.add_argument('--b1', type=parse_int_with_scientific, help='B1 bound for ECM (supports scientific notation, e.g., 26e7)')
    parser.add_argument('--b2', type=parse_int_with_scientific, help='B2 bound for ECM (supports scientific notation, e.g., 4e11)')
    parser.add_argument('--curves', '-c', type=int, default=100, help='Number of curves for ECM')

    # General parameters
    parser.add_argument('--project', '-p', help='Project name')
    parser.add_argument('--no-submit', action='store_true', help='Do not submit to API')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose YAFU output (stream in real-time)')

    return parser


def validate_ecm_args(args: argparse.Namespace, config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    Validate ECM arguments and return any validation errors.

    Args:
        args: Parsed command line arguments
        config: Configuration dictionary (optional, for B2 validation)

    Returns:
        Dictionary mapping argument names to error messages
    """
    errors = {}

    # Decoupled two-stage mode validation
    if hasattr(args, 'stage1_only') and args.stage1_only:
        if hasattr(args, 'tlevel') and args.tlevel is not None:
            errors['tlevel'] = "--stage1-only not compatible with --tlevel. Use --b1/--curves instead."
        if args.b2 is not None and args.b2 != 0:
            errors['b2'] = "--stage1-only runs stage 1 only. B2 should be 0 or omitted."

        # Auto-work mode: Only B1 is required (curves will default to config)
        if hasattr(args, 'auto_work') and args.auto_work:
            if args.b1 is None:
                errors['b1'] = "--stage1-only with --auto-work requires --b1 to be specified"
        # Manual mode: composite required, B1/curves use config defaults if not specified
        else:
            if not args.composite:
                errors['composite'] = "--stage1-only without --auto-work requires --composite to be specified"

    # Auto-work mode validation (check first, before other modes)
    if hasattr(args, 'auto_work') and args.auto_work:
        has_b1_b2 = args.b1 is not None and args.b2 is not None
        has_tlevel = hasattr(args, 'tlevel') and args.tlevel is not None

        # Parameters are now optional - can use server's t-level data
        # Three modes: server t-level (default), client B1/B2, or client t-level

        # Two-stage only compatible with B1/B2 mode (not t-level mode)
        if args.two_stage:
            if has_tlevel:
                errors['two_stage'] = "Two-stage mode not compatible with --tlevel. Use --b1/--b2 instead."
            elif not has_b1_b2:
                errors['two_stage'] = "Two-stage mode requires --b1 and --b2 to be specified"
            # Warn if using two-stage with curves > 1 (GPU batches automatically)
            if args.curves and args.curves > 1:
                errors['curves'] = "Two-stage mode: GPU batches curves automatically. Use --curves 1 or omit."

        # Multiprocess is allowed (works with t-level mode)
        # stage2-only not supported in auto-work
        if args.stage2_only:
            errors['stage2_only'] = "Auto-work mode not compatible with --stage2-only"

        # Composite should not be specified in auto-work mode
        if args.composite:
            errors['composite'] = "Auto-work mode gets composites from server. Do not specify --composite."

        # Return early to avoid conflicting validations
        return errors

    # Filter options only valid in auto-work mode
    if hasattr(args, 'work_count') and args.work_count is not None and not args.auto_work:
        errors['work_count'] = "--work-count only valid in --auto-work mode"
    if hasattr(args, 'min_target_tlevel') and args.min_target_tlevel is not None and not args.auto_work:
        errors['min_target_tlevel'] = "--min-target-tlevel only valid in --auto-work mode"
    if hasattr(args, 'max_target_tlevel') and args.max_target_tlevel is not None and not args.auto_work:
        errors['max_target_tlevel'] = "--max-target-tlevel only valid in --auto-work mode"
    if hasattr(args, 'priority') and args.priority is not None and not args.auto_work:
        errors['priority'] = "--priority only valid in --auto-work mode"

    # T-level mode validation
    if hasattr(args, 'tlevel') and args.tlevel is not None:
        if args.curves:
            errors['curves'] = "Cannot specify both --tlevel and --curves. Choose one."

        # Validate start-tlevel (only meaningful when explicit t-level given)
        if hasattr(args, 'start_tlevel') and args.start_tlevel is not None:
            if args.start_tlevel < 0:
                errors['start_tlevel'] = "--start-tlevel must be non-negative"
            # Only check start < target when explicit t-level given (not auto mode)
            elif args.tlevel > 0 and args.start_tlevel >= args.tlevel:
                errors['start_tlevel'] = f"--start-tlevel ({args.start_tlevel}) must be less than --tlevel ({args.tlevel})"

    # Validate start-tlevel requires tlevel
    if hasattr(args, 'start_tlevel') and args.start_tlevel is not None:
        if not hasattr(args, 'tlevel') or args.tlevel is None:
            errors['start_tlevel'] = "--start-tlevel requires --tlevel to be specified"
        if not args.composite:
            errors['composite'] = "T-level mode requires composite number. Use --composite argument."
        if args.b1:
            errors['b1'] = "T-level mode automatically selects B1. Remove --b1 argument."
        if args.stage2_only:
            errors['mode'] = "T-level mode not compatible with --stage2-only mode."

    # Mode compatibility checks
    if args.multiprocess and args.two_stage:
        errors['mode'] = "Cannot use both --multiprocess and --two-stage. Choose one mode."


    # Stage 2 only mode validation
    if args.stage2_only:
        if args.composite:
            errors['composite'] = "Stage 2 only mode - composite number not required"
        if not args.b2:
            errors['b2'] = "Stage 2 only mode requires B2 bound. Use --b2 argument."

    # Two-stage mode validation
    elif args.two_stage and args.method == 'ecm':
        if not args.composite:
            errors['composite'] = "Two-stage mode requires composite number. Use --composite argument."
        # Two-stage mode requires explicit B2 for Stage 2 coordination
        # Exception: B2=0 is allowed when saving residues (Stage 1 only)
        if args.b2 is None and config:
            _, b2_default = get_method_defaults(config, args.method)
            if not b2_default:
                errors['b2'] = "Two-stage mode requires B2 bound. Use --b2 argument or set default_b2 in config."
        elif args.b2 is None and not config:
            errors['b2'] = "Two-stage mode requires B2 bound. Use --b2 argument."

    # Multiprocess mode validation
    elif args.multiprocess:
        if not args.composite:
            errors['composite'] = "Multiprocess mode requires composite number. Use --composite argument."
        if args.save_residues:
            errors['residues'] = "--save-residues not applicable in multiprocess mode."


    # Standard mode validation
    else:
        if not args.composite:
            errors['composite'] = "Standard mode requires composite number. Use --composite argument."
        if args.two_stage and args.method != 'ecm':
            errors['method'] = "Two-stage mode only available for ECM method."
        if args.save_residues:
            errors['residues'] = "Save residues option only available in two-stage mode."

    # GPU validation
    if args.gpu and args.no_gpu:
        errors['gpu'] = "Cannot specify both --gpu and --no-gpu"

    return errors


def get_workers_default(config: Dict[str, Any]) -> int:
    """
    Get default workers value from config.

    Used for both multiprocess workers and stage2 threads.

    Args:
        config: Configuration dictionary

    Returns:
        Default number of workers
    """
    if config and 'programs' in config and 'gmp_ecm' in config['programs']:
        return config['programs']['gmp_ecm'].get('workers', 4)
    return 4


def get_max_batch_default(config: Dict[str, Any]) -> Optional[int]:
    """
    Get default max_batch value from config.

    Used for chunking large GPU batches in two-stage mode.

    Args:
        config: Configuration dictionary

    Returns:
        Default max_batch value or None if not set
    """
    if config and 'programs' in config and 'gmp_ecm' in config['programs']:
        return config['programs']['gmp_ecm'].get('max_batch')
    return None


# Backward compatibility alias
get_stage2_workers_default = get_workers_default


def get_method_defaults(config: Dict[str, Any], method: str) -> tuple[int, Optional[int]]:
    """
    Get default B1 and B2 values for the specified method.

    Args:
        config: Configuration dictionary
        method: Method name ('ecm', 'pm1', 'pp1')

    Returns:
        Tuple of (b1_default, b2_default)
    """
    gmp_config = config['programs']['gmp_ecm']

    if method == 'pm1':
        b1_default = gmp_config.get('pm1_b1', gmp_config['default_b1'])
        b2_default = gmp_config.get('pm1_b2', gmp_config.get('default_b2'))
    elif method == 'pp1':
        b1_default = gmp_config.get('pp1_b1', gmp_config['default_b1'])
        b2_default = gmp_config.get('pp1_b2', gmp_config.get('default_b2'))
    else:  # ecm
        b1_default = gmp_config['default_b1']
        b2_default = gmp_config.get('default_b2')

    return b1_default, b2_default


def resolve_gpu_settings(args: argparse.Namespace, config: Dict[str, Any]) -> tuple[bool, Optional[int], Optional[int]]:
    """
    Resolve GPU settings from arguments and configuration.

    Returns:
        Tuple of (use_gpu, gpu_device, gpu_curves)
    """
    # GPU settings: command line overrides config defaults
    if args.no_gpu:
        use_gpu = False
    elif args.gpu:
        use_gpu = True
    else:
        use_gpu = config['programs']['gmp_ecm'].get('gpu_enabled', False)

    gpu_device = (args.gpu_device if args.gpu_device is not None
                  else config['programs']['gmp_ecm'].get('gpu_device'))
    gpu_curves = (args.gpu_curves if args.gpu_curves is not None
                  else config['programs']['gmp_ecm'].get('gpu_curves'))

    return use_gpu, gpu_device, gpu_curves


def resolve_worker_count(args: argparse.Namespace) -> int:
    """Resolve number of workers for multiprocess mode."""
    if args.multiprocess and args.workers <= 0:
        return multiprocessing.cpu_count()
    return args.workers


def print_validation_errors(errors: Dict[str, str]) -> None:
    """Print validation errors and exit."""
    if errors:
        print("Argument validation errors:")
        for field, message in errors.items():
            print(f"  {field}: {message}")
        sys.exit(1)
