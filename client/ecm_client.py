#!/usr/bin/env python3
"""
ECM Client - Server-coordinated factorization work

This entry point handles all server-coordinated ECM modes:
- Auto-work with server-provided composites
- Target a specific composite (--composite flag)
- Stage 1 only (GPU producer): Upload residues to server
- Stage 2 only (CPU consumer): Download and process residues from server

The client queries the server for t-level status and runs optimal curves.

For local/manual factorization (with a specific composite and residue files),
use ecm_wrapper.py instead.
"""

import sys
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
