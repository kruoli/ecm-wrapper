#!/usr/bin/env python3
"""
ECM Client - Server-coordinated factorization work

This entry point handles all server-coordinated ECM modes:
- Adaptive CPU mode (default): stage 2 priority, falls back to ECM
- GPU producer (--stage1-only): Upload residues to server
- Target a specific composite (--composite flag)
- Stage 2 only (--stage2-only): Download and process residues from server
- P-1/P+1 sweep (--pm1/--pp1/--p1)
- Standard auto-work (--standard): Legacy t-level or B1/B2 mode

The client queries the server for t-level status and runs optimal curves.

For local/manual factorization (with a specific composite and residue files),
use ecm_wrapper.py instead.
"""

import sys

if sys.version_info < (3, 9):
    print(f"Error: Python 3.9+ is required (you have Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})")
    sys.exit(1)

from pathlib import Path

from lib.ecm_executor import ECMWrapper
from lib.work_modes import WorkLoopContext, get_work_mode
from lib.arg_parser import create_client_parser


def check_setup_complete() -> bool:
    """Check if client.local.yaml exists and warn if not."""
    config_path = Path("client.local.yaml")
    if not config_path.exists():
        print()
        print("!" * 70)
        print("!!" + " " * 66 + "!!")
        print("!!  WARNING: client.local.yaml not found!" + " " * 25 + "!!")
        print("!!" + " " * 66 + "!!")
        print("!!  You are using default settings (username: 'default_user')." + " " * 5 + "!!")
        print("!!  Your contributions will not be properly tracked." + " " * 15 + "!!")
        print("!!" + " " * 66 + "!!")
        print("!!  Please run the setup wizard first:" + " " * 28 + "!!")
        print("!!" + " " * 66 + "!!")
        print("!!      python3 setup.py" + " " * 42 + "!!")
        print("!!" + " " * 66 + "!!")
        print("!" * 70)
        print()

        # Ask if they want to continue anyway
        try:
            response = input("Continue with default settings? [y/N]: ").strip().lower()
            if response not in ('y', 'yes'):
                print("\nExiting. Please run 'python3 setup.py' to configure the client.")
                return False
            print()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return False

    return True


def _has_explicit_mode(args) -> bool:
    """Check if user specified an explicit work mode via flags."""
    # Direct mode flags
    mode_flags = [
        'composite', 'pm1', 'pp1', 'p1',
        'stage1_only', 'stage2_only', 'standard', 'adaptive',
    ]
    for flag in mode_flags:
        if getattr(args, flag, None):
            return True
    # Implicit standard mode: user specified execution parameters
    if getattr(args, 'b1', None) is not None:
        return True
    if getattr(args, 'tlevel', None) is not None:
        return True
    if getattr(args, 'two_stage', False):
        return True
    if getattr(args, 'multiprocess', False):
        return True
    return False


def _prompt_work_mode(gpu_enabled: bool) -> str:
    """
    Prompt user to select work mode interactively.

    Args:
        gpu_enabled: Whether GPU is configured in setup

    Returns:
        'gpu', 'cpu', or 'adaptive'
    """
    print()
    print("=" * 60)
    print("ECM Client - Work Mode Selection")
    print("=" * 60)
    print()

    if gpu_enabled:
        print("  [1] GPU Producer  - Run stage 1 on GPU, upload residues")
        print("  [2] CPU Worker    - Adaptive: stage 2 priority, then ECM")
        print()
        try:
            choice = input("Choice [2]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            sys.exit(0)

        if choice == '1':
            return 'gpu'
        else:
            return 'adaptive'
    else:
        # No GPU configured - go straight to adaptive CPU
        print("  Running in adaptive CPU mode (stage 2 priority, then ECM)")
        print("  Tip: Run setup.py to configure GPU for stage 1 production")
        print()
        return 'adaptive'


def main():
    """Main entry point for ECM client."""
    parser = create_client_parser()
    args = parser.parse_args()

    # Check for setup completion (unless --help was requested)
    if not check_setup_complete():
        sys.exit(1)

    # ecm_client.py always operates in auto-work mode (implied)
    args.auto_work = True

    # Initialize wrapper
    wrapper = ECMWrapper('client.yaml')

    # Interactive mode selection if no explicit flags given
    if not _has_explicit_mode(args):
        gpu_enabled = wrapper.config.get('programs', {}).get('gmp_ecm', {}).get('gpu_enabled', False)
        choice = _prompt_work_mode(gpu_enabled)

        if choice == 'gpu':
            args.stage1_only = True
        else:
            args.adaptive = True

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
