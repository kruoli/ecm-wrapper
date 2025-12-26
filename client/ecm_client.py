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

from lib.ecm_executor import ECMWrapper
from lib.work_modes import WorkLoopContext, get_work_mode
from lib.arg_parser import create_client_parser


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
