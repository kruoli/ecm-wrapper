#!/usr/bin/env python3
import subprocess
import sys
from typing import Optional, Dict, Any, List
from base_wrapper import BaseWrapper
from parsing_utils import parse_yafu_ecm_output, parse_yafu_auto_factors, Timeouts

class YAFUWrapper(BaseWrapper):
    def __init__(self, config_path: str):
        """Initialize YAFU wrapper with shared base functionality"""
        super().__init__(config_path)

    def _add_yafu_threading(self, cmd: List[str], threads: Optional[int] = None) -> None:
        """Add threading parameter to YAFU command if configured.

        Args:
            cmd: Command list to append to
            threads: Optional runtime thread override (uses config default if not specified)
        """
        if threads is not None:
            cmd.extend(['-threads', str(threads)])
        elif 'threads' in self.config.get('programs', {}).get('yafu', {}):
            cmd.extend(['-threads', str(self.config['programs']['yafu']['threads'])])

    def _build_yafu_ecm_cmd(self, method: str, b1: int, b2: Optional[int] = None,
                           curves: int = 100, composite: str = "") -> List[str]:
        """Build YAFU command for ECM/P-1/P+1 methods.

        Returns:
            Command list with expression as first positional argument
        """
        yafu_path = self.config['programs']['yafu']['path']

        # YAFU expects expression as first positional argument
        method_input = self._build_yafu_method_input(composite, method)
        cmd = [yafu_path, method_input]

        # Add method-specific parameters
        if method == "pm1":
            cmd.extend(['-B1pm1', str(b1)])
            if b2:
                cmd.extend(['-B2pm1', str(b2)])
        elif method == "pp1":
            cmd.extend(['-B1pp1', str(b1)])
            if b2:
                cmd.extend(['-B2pp1', str(b2)])
        else:  # ecm
            cmd.extend(['-B1ecm', str(b1)])
            if b2:
                cmd.extend(['-B2ecm', str(b2)])
            cmd.extend(['-curves', str(curves)])

        return cmd

    def _build_yafu_method_input(self, composite: str, method: str) -> str:
        """Build YAFU method input string.

        YAFU uses factor() for all methods - the actual method is controlled
        by flags like -B1ecm, -B1pm1, -B1pp1.
        """
        return f'factor({composite})'

    def _build_yafu_auto_cmd(self, method: Optional[str] = None, composite: str = "", threads: Optional[int] = None) -> List[str]:
        """Build YAFU command for automatic factorization.

        Args:
            method: Optional method to force (siqs, nfs, etc.)
            composite: Number to factor
            threads: Optional thread count override

        Returns:
            Command list with expression as first positional argument
        """
        yafu_path = self.config['programs']['yafu']['path']

        # YAFU expects expression as first positional argument
        auto_input = f'factor({composite})'
        cmd = [yafu_path, auto_input]

        if method:
            # Force specific method: -method siqs, -method nfs, etc
            cmd.extend(['-method', method])

        self._add_yafu_threading(cmd, threads)

        return cmd

    def run_yafu_ecm(self, composite: str, b1: int, b2: Optional[int] = None,
                     curves: int = 100, method: str = "ecm") -> Dict[str, Any]:
        """Run YAFU in ECM/P-1/P+1 mode using unified base infrastructure."""
        # Build command with expression as first positional argument
        cmd = self._build_yafu_ecm_cmd(method, b1, b2, curves, composite)
        self._add_yafu_threading(cmd)

        # Use unified subprocess execution with parsing (no stdin needed)
        results = self.run_subprocess_with_parsing(
            cmd=cmd,
            timeout=Timeouts.YAFU_ECM,
            composite=composite,
            method=method,
            parse_function=parse_yafu_ecm_output,
            curves=curves,
            b1=b1,
            b2=b2,
            track_curves=True  # Enable curves tracking for YAFU
        )

        # Store command for debugging
        results['cmd'] = cmd

        # Log found factors using base class functionality
        if results.get('factors_found'):
            for factor in results['factors_found']:
                self.log_factor_found(composite, factor, b1, b2, curves, method=method, program=f"YAFU ({method.upper()})")

        # Save raw output if configured
        if self.config['execution']['save_raw_output']:
            self.save_raw_output(results, f'yafu-{method}')

        return results

    def run_yafu_auto(self, composite: str, method: Optional[str] = None, threads: Optional[int] = None) -> Dict[str, Any]:
        """Run YAFU in automatic factorization mode using unified base infrastructure.

        Args:
            composite: Number to factor
            method: Optional method to force (siqs, nfs, etc.)
            threads: Optional thread count override

        Returns:
            Results dictionary
        """
        # Build command with expression as first positional argument
        cmd = self._build_yafu_auto_cmd(method, composite, threads)

        # Use unified subprocess execution with parsing (no stdin needed)
        results = self.run_subprocess_with_parsing(
            cmd=cmd,
            timeout=Timeouts.YAFU_AUTO,
            composite=composite,
            method=method or 'auto',
            parse_function=parse_yafu_auto_factors
        )

        # Store command for debugging
        results['cmd'] = cmd

        # Log found factors using base class functionality
        if results.get('factors_found'):
            for factor in results['factors_found']:
                self.log_factor_found(composite, factor, None, None, None,
                                    method=method or 'auto', program=f"YAFU ({(method or 'AUTO').upper()})")

        # Save raw output if configured
        if self.config['execution']['save_raw_output']:
            self.save_raw_output(results, f'yafu-{method or "auto"}')

        return results




    def get_program_version(self, program: str) -> str:
        """Override base class method to get YAFU version"""
        return self.get_yafu_version()

    def get_yafu_version(self) -> str:
        """Get YAFU version"""
        try:
            result = subprocess.run(
                [self.config['programs']['yafu']['path'], '-h'],
                capture_output=True,
                text=True,
                timeout=5
            )
            from parsing_utils import extract_program_version
            return extract_program_version(result.stdout, 'yafu')
        except:
            pass
        return "unknown"


def main():
    from arg_parser import create_yafu_parser

    parser = create_yafu_parser()
    args = parser.parse_args()

    wrapper = YAFUWrapper(args.config)

    if args.mode in ['ecm', 'pm1', 'pp1']:
        # Use ECM/P-1/P+1 mode
        b1 = args.b1 or 50000  # Default B1 if not specified
        results = wrapper.run_yafu_ecm(
            composite=args.composite,
            b1=b1,
            b2=args.b2,
            curves=args.curves,
            method=args.mode
        )
    else:
        # Use automatic or specific method
        method = None if args.mode == 'auto' else args.mode
        results = wrapper.run_yafu_auto(
            composite=args.composite,
            method=method
        )

    # Print results summary
    print(f"\n{'='*80}")
    print(f"YAFU {results.get('method', 'unknown').upper()} Results")
    print(f"{'='*80}")
    print(f"Composite: {args.composite} ({len(args.composite)} digits)")
    print(f"Curves completed: {results.get('curves_completed', 0)}")
    print(f"Execution time: {results.get('execution_time', 0):.2f}s")

    if results.get('factors_found'):
        print(f"\n✓ Found {len(results['factors_found'])} factor(s):")
        for factor in results['factors_found']:
            print(f"  - {factor}")
    else:
        print("\n✗ No factors found")
    print(f"{'='*80}")

    # Debug: print raw output if verbose or no factors
    if not results.get('factors_found') or (hasattr(args, 'verbose') and args.verbose):
        print("\n=== RAW YAFU OUTPUT ===")
        print(results.get('raw_output', '(no output)'))
        print("=== END ===\n")

        # Also print the command that was executed (for debugging)
        if 'cmd' in results:
            print("Command executed:", ' '.join(results['cmd']))

    # Submit results unless disabled or failed
    if not args.no_submit:
        # Only submit if we actually completed some curves (not a failure)
        if results.get('curves_completed', 0) > 0 or results.get('factors_found'):
            program_name = f'yafu-{results.get("method", "ecm")}'
            success = wrapper.submit_result(results, args.project, program_name)
            sys.exit(0 if success else 1)
        else:
            wrapper.logger.warning("Skipping result submission due to failure")
            sys.exit(1)

if __name__ == '__main__':
    main()