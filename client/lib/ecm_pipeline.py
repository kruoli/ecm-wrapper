"""
ECM Pipeline Orchestration

Handles complex multi-stage ECM workflows including:
- Two-stage GPU+CPU pipeline
- Multiprocess execution
- T-level targeting
"""

import logging
import multiprocessing
from typing import Dict, Any, Optional, List
from pathlib import Path

from .ecm_config import TwoStageConfig, MultiprocessConfig, TLevelConfig, FactorResult
from .ecm_math import calculate_tlevel, get_optimal_b1_for_tlevel


logger = logging.getLogger(__name__)


class TwoStagePipeline:
    """
    Orchestrates two-stage ECM: Stage 1 (GPU) + Stage 2 (CPU).

    Stage 1 generates residues using GPU-accelerated ECM.
    Stage 2 processes residues with CPU for deeper searches.
    """

    def __init__(self, config: TwoStageConfig, wrapper):
        """
        Initialize pipeline.

        Args:
            config: Two-stage configuration
            wrapper: ECMWrapper instance for execution
        """
        self.config = config
        self.wrapper = wrapper
        self.residue_file: Optional[Path] = None

    def execute(self) -> FactorResult:
        """
        Execute complete two-stage pipeline.

        Returns:
            FactorResult with all discovered factors
        """
        result = FactorResult()

        # Stage 1: GPU-accelerated residue generation
        logger.info("=" * 60)
        logger.info("STAGE 1: GPU residue generation")
        logger.info("=" * 60)

        stage1_result = self._execute_stage1()
        result.execution_time += stage1_result.execution_time

        # Check if factor found in stage 1
        if stage1_result.success:
            logger.info("Factor found in Stage 1, skipping Stage 2")
            for factor, sigma in stage1_result.factor_sigma_pairs:
                result.add_factor(factor, sigma)
            result.curves_run = stage1_result.curves_run
            return result

        # Stage 2: CPU processing of residues
        if self.residue_file and self.residue_file.exists():
            logger.info("=" * 60)
            logger.info("STAGE 2: CPU residue processing")
            logger.info("=" * 60)

            stage2_result = self._execute_stage2()
            result.execution_time += stage2_result.execution_time

            if stage2_result.success:
                for factor, sigma in stage2_result.factor_sigma_pairs:
                    result.add_factor(factor, sigma)

            result.curves_run = (
                stage1_result.curves_run +
                stage2_result.curves_run
            )
        else:
            logger.warning("No residue file found, skipping Stage 2")
            result.curves_run = stage1_result.curves_run

        return result

    def _execute_stage1(self) -> FactorResult:
        """Execute Stage 1: GPU residue generation."""
        from .ecm_config import ECMConfig

        # Configure Stage 1
        stage1_config = ECMConfig(
            composite=self.config.composite,
            b1=self.config.b1,
            b2=self.config.b2,
            curves=self.config.stage1_curves,
            parametrization=self.config.stage1_parametrization,
            verbose=self.config.verbose,
            timeout=self.config.timeout_stage1,
            save_residues=self.config.save_residues
        )

        # Execute using GPU if configured
        # Note: This would call the GPU execution method
        # For now, use standard execution
        result = self.wrapper.run_ecm_v2(stage1_config)

        # Store residue file path if generated
        if self.config.save_residues:
            self.residue_file = Path(self.config.save_residues)

        return result

    def _execute_stage2(self) -> FactorResult:
        """Execute Stage 2: CPU residue processing."""
        result = FactorResult()

        if not self.residue_file:
            return result

        # Stage 2 processes the residue file
        # This would call specialized Stage 2 execution
        logger.info("Processing residue file: %s", self.residue_file)
        logger.info("Stage 2 curves per residue: %d", self.config.stage2_curves_per_residue)

        # For now, log that we would process it
        # Actual implementation would call stage2_executor
        logger.warning("Stage 2 processing not yet implemented in pipeline")

        return result


class MultiprocessPipeline:
    """
    Orchestrates multiprocess ECM execution.

    Distributes curve workload across multiple CPU cores.
    """

    def __init__(self, config: MultiprocessConfig, wrapper):
        """
        Initialize multiprocess pipeline.

        Args:
            config: Multiprocess configuration
            wrapper: ECMWrapper instance
        """
        self.config = config
        self.wrapper = wrapper

    def execute(self) -> FactorResult:
        """
        Execute multiprocess ECM.

        Splits curves across processes and combines results.

        Returns:
            Combined FactorResult
        """
        combined_result = FactorResult()

        # Calculate curves per process
        num_processes = self.config.num_processes
        total_curves = self.config.total_curves
        curves_per_process = self.config.curves_per_process

        logger.info(
            "Multiprocess ECM: %d processes, %d total curves, %d curves/process",
            num_processes, total_curves, curves_per_process
        )

        # For now, execute sequentially
        # Full implementation would use multiprocessing.Pool
        from .ecm_config import ECMConfig

        remaining_curves = total_curves
        while remaining_curves > 0 and not combined_result.success:
            batch_curves = min(remaining_curves, curves_per_process)

            batch_config = ECMConfig(
                composite=self.config.composite,
                b1=self.config.b1,
                b2=self.config.b2,
                curves=batch_curves,
                parametrization=self.config.parametrization,
                verbose=self.config.verbose,
                timeout=self.config.timeout
            )

            batch_result = self.wrapper.run_ecm_v2(batch_config)

            # Combine results
            for factor, sigma in batch_result.factor_sigma_pairs:
                combined_result.add_factor(factor, sigma)

            combined_result.curves_run += batch_result.curves_run
            combined_result.execution_time += batch_result.execution_time

            # Stop if factor found
            if batch_result.success:
                logger.info("Factor found, stopping multiprocess execution")
                break

            remaining_curves -= batch_curves

        return combined_result


class TLevelPipeline:
    """
    Orchestrates T-level targeting workflow.

    Progressively runs ECM with optimized B1 values to reach target t-level.
    """

    def __init__(self, config: TLevelConfig, wrapper):
        """
        Initialize T-level pipeline.

        Args:
            config: T-level configuration
            wrapper: ECMWrapper instance
        """
        self.config = config
        self.wrapper = wrapper
        self.curve_history: List[str] = []

    def execute(self) -> FactorResult:
        """
        Execute T-level targeting.

        Returns:
            FactorResult with factors found during targeting
        """
        combined_result = FactorResult()
        current_tlevel = 0.0

        logger.info(
            "T-level targeting: composite=%s, target=t%.1f",
            self.config.composite[:20] + "...",
            self.config.target_t_level
        )

        while current_tlevel < self.config.target_t_level:
            # Determine B1 for next step
            remaining_tlevel = self.config.target_t_level - current_tlevel
            b1, estimated_curves = get_optimal_b1_for_tlevel(remaining_tlevel)

            # Limit curves per step
            curves_this_step = min(
                estimated_curves,
                self.config.max_curves_per_b1
            )

            logger.info(
                "T-level step: current=t%.1f, target=t%.1f, B1=%d, curves=%d",
                current_tlevel, self.config.target_t_level,
                b1, curves_this_step
            )

            # Execute ECM for this step
            from .ecm_config import ECMConfig
            step_config = ECMConfig(
                composite=self.config.composite,
                b1=b1,
                curves=curves_this_step,
                parametrization=self.config.parametrization,
                threads=self.config.threads,
                verbose=self.config.verbose,
                timeout=self.config.timeout
            )

            step_result = self.wrapper.run_ecm_v2(step_config)

            # Combine results
            for factor, sigma in step_result.factor_sigma_pairs:
                combined_result.add_factor(factor, sigma)

            combined_result.curves_run += step_result.curves_run
            combined_result.execution_time += step_result.execution_time

            # Update curve history
            self.curve_history.append(
                f"{curves_this_step}@{b1},p={self.config.parametrization}"
            )

            # Recalculate t-level
            current_tlevel = calculate_tlevel(self.curve_history)

            logger.info("Reached t%.3f after %d curves",
                       current_tlevel, combined_result.curves_run)

            # Stop if factor found
            if step_result.success:
                logger.info("Factor found, stopping T-level targeting")
                break

        logger.info(
            "T-level targeting complete: t%.3f in %d curves (%.2fs)",
            current_tlevel, combined_result.curves_run,
            combined_result.execution_time
        )

        return combined_result
