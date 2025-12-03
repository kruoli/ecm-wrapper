"""
ResultsBuilder - Fluent API for constructing ECM results dictionaries.

Standardizes the creation of results across all ECM execution modes:
- Standard ECM runs
- Two-stage pipeline
- Multiprocess execution
- Residue processing
- Batch pipeline workers

Design Decisions:
- OOP approach with fluent API
- Standardize on list accumulation for raw_output
- Merge _build_stage1_results() functionality
"""

from typing import Dict, Any, List, Optional, Tuple


class ResultsBuilder:
    """Builder for ECM results dictionaries with fluent API."""

    def __init__(self, composite: str, method: str = "ecm"):
        """Initialize with required fields."""
        self._data: Dict[str, Any] = {
            'composite': composite,
            'method': method,
            'factor_found': None,
            'factors_found': [],
            'curves_completed': 0,
            'curves_requested': 0,
            'execution_time': 0,
            'raw_output_lines': [],  # Internal list, converted to string on build()
            'b1': None,
            'b2': None,
            'sigma': None,
        }

    # Core parameter methods
    def with_b1(self, b1: int) -> 'ResultsBuilder':
        """Set B1 parameter."""
        self._data['b1'] = b1
        return self

    def with_b2(self, b2: Optional[int]) -> 'ResultsBuilder':
        """Set B2 parameter (None = default, 0 = stage1 only)."""
        self._data['b2'] = b2
        return self

    def with_sigma(self, sigma: Optional[str]) -> 'ResultsBuilder':
        """Set sigma value."""
        self._data['sigma'] = sigma
        return self

    def with_parametrization(self, param: int) -> 'ResultsBuilder':
        """Set ECM parametrization (0-3)."""
        self._data['parametrization'] = param
        return self

    # Curve tracking methods
    def with_curves(self, requested: int, completed: int = 0) -> 'ResultsBuilder':
        """Set curves requested and completed."""
        self._data['curves_requested'] = requested
        self._data['curves_completed'] = completed
        return self

    def increment_curves(self, count: int = 1) -> 'ResultsBuilder':
        """Increment curves_completed."""
        self._data['curves_completed'] += count
        return self

    # Factor methods (merged from _build_stage1_results)
    def with_factors(self, all_factors: List[Tuple[str, Optional[str]]]) -> 'ResultsBuilder':
        """
        Add factors from list of (factor, sigma) tuples.
        Automatically builds factors_found list and factor_sigmas dict.
        """
        if all_factors:
            self._data['factors_found'] = [f[0] for f in all_factors]
            factor_sigmas = {f: s for f, s in all_factors if s}
            if factor_sigmas:
                self._data['factor_sigmas'] = factor_sigmas
            self._data['factor_found'] = all_factors[0][0]
        return self

    def with_single_factor(self, factor: str, sigma: Optional[str] = None) -> 'ResultsBuilder':
        """Add a single factor."""
        return self.with_factors([(factor, sigma)])

    # Raw output methods (standardized on list accumulation)
    def add_raw_output(self, output: str) -> 'ResultsBuilder':
        """Add raw output line(s)."""
        if output:
            self._data['raw_output_lines'].append(output)
        return self

    def add_raw_outputs(self, outputs: List[str]) -> 'ResultsBuilder':
        """Add multiple raw output lines."""
        self._data['raw_output_lines'].extend(outputs)
        return self

    # Execution metadata
    def with_execution_time(self, seconds: float) -> 'ResultsBuilder':
        """Set execution time."""
        self._data['execution_time'] = seconds
        return self

    def with_work_id(self, work_id: str) -> 'ResultsBuilder':
        """Set work assignment ID."""
        self._data['work_id'] = work_id
        return self

    # Execution mode flags
    def as_two_stage(self) -> 'ResultsBuilder':
        """Mark as two-stage execution."""
        self._data['two_stage'] = True
        return self

    def as_multiprocess(self, workers: int) -> 'ResultsBuilder':
        """Mark as multiprocess execution."""
        self._data['multiprocess'] = True
        self._data['workers'] = workers
        return self

    def as_stage2_workers(self, workers: int) -> 'ResultsBuilder':
        """Mark as stage2 multiprocess execution."""
        self._data['stage2_workers'] = workers
        return self

    # Residue tracking
    def with_residue_file(self, residue_file: str) -> 'ResultsBuilder':
        """Track residue file source."""
        self._data['residue_file'] = residue_file
        return self

    # Special case: Stage1 only
    def as_stage1_only(self) -> 'ResultsBuilder':
        """Mark as stage1-only execution (sets b2=0)."""
        self._data['b2'] = 0
        return self

    # Build methods
    def build(self, truncate_output: int = 10000) -> Dict[str, Any]:
        """
        Build final results dictionary.

        Args:
            truncate_output: Max chars for raw_output (0 = no truncation)

        Returns:
            Complete results dictionary ready for API submission
        """
        # Convert raw_output_lines list to string
        raw_output = '\n\n'.join(self._data.pop('raw_output_lines', []))

        # Truncate if requested
        if truncate_output > 0 and len(raw_output) > truncate_output:
            raw_output = raw_output[:truncate_output] + f"\n... (truncated, {len(raw_output)} total chars)"

        self._data['raw_output'] = raw_output

        return self._data.copy()

    def build_no_truncate(self) -> Dict[str, Any]:
        """Build without truncating raw_output."""
        return self.build(truncate_output=0)


# Convenience factory functions
def results_for_ecm(composite: str, b1: int, curves: int, param: int = 1) -> ResultsBuilder:
    """Create builder pre-configured for standard ECM run."""
    return (ResultsBuilder(composite, method='ecm')
            .with_b1(b1)
            .with_curves(curves)
            .with_parametrization(param))


def results_for_stage1(composite: str, b1: int, curves: int, param: int = 1) -> ResultsBuilder:
    """Create builder pre-configured for stage1-only run."""
    return (ResultsBuilder(composite, method='ecm')
            .with_b1(b1)
            .as_stage1_only()
            .with_curves(curves)
            .with_parametrization(param))
