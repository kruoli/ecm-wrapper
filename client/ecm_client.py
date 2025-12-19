#!/usr/bin/env python3
"""
ECM Client - Server-coordinated factorization work

This entry point handles all server-coordinated ECM modes:
- Auto-work with server-provided composites
- Target a specific composite (--composite flag)
- Stage 1 only (GPU producer): Upload residues to server
- Stage 2 only (CPU consumer): Download and process residues from server

The client queries the server for t-level status and runs optimal curves.
"""

import sys

from lib.ecm_executor import ECMWrapper
from lib.work_modes import WorkLoopContext, get_work_mode
from lib.arg_parser import parse_int_with_scientific


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
    parser.add_argument('--min-digits', type=int,
                       help='Minimum composite size (digits)')
    parser.add_argument('--max-digits', type=int,
                       help='Maximum composite size (digits)')
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


def main():
    """Main entry point for ECM client."""
    parser = create_client_parser()
    args = parser.parse_args()

    # ecm_client.py always operates in auto-work mode (implied)
    args.auto_work = True

    # Initialize wrapper
    wrapper = ECMWrapper('client.yaml')

    # Get client ID from wrapper (uses same format as abandon_work: username-cpu_name)
    client_id = wrapper.client_id

    # Determine work count limit
    work_count_limit = args.work_count if hasattr(args, 'work_count') and args.work_count else None

    # Create work loop context
    ctx = WorkLoopContext(
        wrapper=wrapper,
        client_id=client_id,
        args=args,
        work_count_limit=work_count_limit
    )

    # Get appropriate work mode based on args
    mode = get_work_mode(ctx)

    # Run the work loop
    completed = mode.run()

    # Exit with success
    sys.exit(0)


if __name__ == '__main__':
    main()
