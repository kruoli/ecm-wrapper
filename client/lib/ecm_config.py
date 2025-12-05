"""
Configuration classes for ECM wrapper operations.

This module defines dataclasses that encapsulate ECM execution parameters,
reducing function argument counts and improving code maintainability.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class ECMConfig:
    """Configuration for standard ECM execution."""

    composite: str
    b1: int
    b2: Optional[int] = None
    curves: int = 1
    sigma: Optional[int] = None
    parametrization: int = 1  # Default to 1 (CPU/Montgomery), use 3 for GPU
    threads: int = 1
    verbose: bool = False
    timeout: int = 3600
    save_residues: Optional[str] = None

    # GPU support
    use_gpu: bool = False
    gpu_device: Optional[int] = None
    gpu_curves: Optional[int] = None

    # Method and progress
    method: str = "ecm"  # 'ecm', 'pm1', 'pp1'
    progress_interval: int = 0

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.b1 <= 0:
            raise ValueError(f"B1 must be positive, got {self.b1}")
        if self.curves <= 0:
            raise ValueError(f"Curves must be positive, got {self.curves}")
        if self.parametrization not in [0, 1, 2, 3]:
            raise ValueError(f"Parametrization must be 0-3, got {self.parametrization}")
        if self.method not in ['ecm', 'pm1', 'pp1']:
            raise ValueError(f"Method must be 'ecm', 'pm1', or 'pp1', got {self.method}")

        # Auto-set parametrization for GPU mode if not explicitly set
        if self.use_gpu and self.parametrization == 1:
            self.parametrization = 3  # GPU uses twisted Edwards curves


@dataclass
class TwoStageConfig:
    """Configuration for two-stage ECM pipeline (GPU + CPU)."""

    composite: str
    b1: int
    b2: Optional[int] = None
    stage1_curves: int = 100
    stage2_curves_per_residue: int = 1000
    stage1_device: str = "GPU"
    stage2_device: str = "CPU"
    stage1_parametrization: int = 3
    stage2_parametrization: int = 1
    threads: int = 1
    verbose: bool = False
    save_residues: Optional[str] = None
    timeout_stage1: int = 3600
    timeout_stage2: int = 7200
    resume_file: Optional[str] = None

    # GPU support
    gpu_device: Optional[int] = None
    gpu_curves: Optional[int] = None

    # Execution control
    continue_after_factor: bool = False

    # API submission
    project: Optional[str] = None
    no_submit: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if self.b1 <= 0:
            raise ValueError(f"B1 must be positive, got {self.b1}")
        if self.stage1_curves <= 0:
            raise ValueError(f"Stage1 curves must be positive")
        if self.stage2_curves_per_residue <= 0:
            raise ValueError(f"Stage2 curves must be positive")


@dataclass
class MultiprocessConfig:
    """Configuration for multiprocess ECM execution."""

    composite: str
    b1: int
    b2: Optional[int] = None
    total_curves: int = 1000
    curves_per_process: int = 100
    num_processes: Optional[int] = None  # None = auto-detect CPU count
    parametrization: int = 1  # Default to 1 (CPU/Montgomery)
    method: str = "ecm"  # 'ecm', 'pm1', 'pp1'
    verbose: bool = False
    timeout: int = 3600

    # Execution control
    continue_after_factor: bool = False
    method: str = "ecm"  # 'ecm', 'pm1', 'pp1'

    def __post_init__(self):
        """Validate and auto-configure."""
        if self.b1 <= 0:
            raise ValueError(f"B1 must be positive, got {self.b1}")
        if self.total_curves <= 0:
            raise ValueError(f"Total curves must be positive")
        if self.curves_per_process <= 0:
            raise ValueError(f"Curves per process must be positive")
        if self.method not in ['ecm', 'pm1', 'pp1']:
            raise ValueError(f"Method must be 'ecm', 'pm1', or 'pp1', got {self.method}")

        # Auto-detect number of processes if not specified
        if self.num_processes is None:
            import multiprocessing
            self.num_processes = multiprocessing.cpu_count()


@dataclass
class TLevelConfig:
    """Configuration for t-level targeting ECM."""

    composite: str
    target_t_level: float
    b1_strategy: str = "optimal"  # 'optimal', 'conservative', 'aggressive'
    parametrization: int = 1  # Default to 1 (CPU/Montgomery), use 3 for GPU/two-stage
    threads: int = 1
    verbose: bool = False
    timeout: int = 7200

    # Execution modes
    workers: int = 1  # Alias for threads (multiprocess support)
    use_two_stage: bool = False

    # API submission
    project: Optional[str] = None
    no_submit: bool = False
    auto_adjust_target: bool = False  # Adjust target when factors found
    work_id: Optional[int] = None  # For auto-work mode batch submissions

    def __post_init__(self):
        """Validate configuration."""
        if self.target_t_level <= 0:
            raise ValueError(f"Target t-level must be positive")
        if self.b1_strategy not in ['optimal', 'conservative', 'aggressive']:
            raise ValueError(f"Unknown B1 strategy: {self.b1_strategy}")

        # Sync workers and threads (they're the same thing)
        if self.workers != 1:
            self.threads = self.workers
        elif self.threads != 1:
            self.workers = self.threads

        # Auto-set parametrization for two-stage mode if not explicitly set
        if self.use_two_stage and self.parametrization == 1:
            self.parametrization = 3  # GPU uses twisted Edwards curves


@dataclass
class FactorResult:
    """Result from ECM factorization attempt."""

    factors: List[str] = field(default_factory=list)
    sigmas: List[Optional[str]] = field(default_factory=list)
    curves_run: int = 0
    execution_time: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    raw_output: Optional[str] = None

    def add_factor(self, factor: str, sigma: Optional[str] = None):
        """Add a discovered factor with its sigma."""
        self.factors.append(factor)
        self.sigmas.append(sigma)
        self.success = True

    @property
    def factor_sigma_pairs(self) -> List[tuple]:
        """Get list of (factor, sigma) tuples."""
        return list(zip(self.factors, self.sigmas))

    def to_dict(self, composite: Optional[str] = None, method: str = 'ecm') -> Dict[str, Any]:
        """
        Convert FactorResult to dictionary format (for backward compatibility).

        Args:
            composite: Optional composite number to include in result
            method: Method used ('ecm', 'pm1', 'pp1')

        Returns:
            Dictionary with v1-compatible format
        """
        result = {
            'success': self.success,
            'factors_found': self.factors,
            'curves_completed': self.curves_run,
            'execution_time': self.execution_time,
            'raw_output': self.raw_output,
            'method': method
        }
        if composite:
            result['composite'] = composite
        return result
