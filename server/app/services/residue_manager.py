"""
Residue file manager service for decoupled two-stage ECM.

Handles:
- Parsing residue file metadata
- Storing/retrieving residue files
- Work assignment for stage 2
- Lifecycle management (claim, complete, expire)
"""

import logging
import hashlib
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from ..models.residues import ECMResidue
from ..models.composites import Composite
from ..models.attempts import ECMAttempt
from ..config import get_settings
from .t_level_calculator import TLevelCalculator

logger = logging.getLogger(__name__)


class ResidueManager:
    """Manages ECM residue files for decoupled two-stage processing."""

    # Patterns for parsing residue file metadata
    PARAM_PATTERN = re.compile(r'PARAM=(\d+)')
    B1_PATTERN = re.compile(r'B1=(\d+)')
    N_PATTERN = re.compile(r'N=(\d+)')
    SIGMA_PATTERN = re.compile(r'SIGMA=(\d+)')

    def __init__(self):
        """Initialize the residue manager."""
        settings = get_settings()
        self.storage_dir = Path(settings.residue_storage_path)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.t_level_calculator = TLevelCalculator()

    def parse_residue_file(self, file_content: bytes) -> Dict[str, Any]:
        """
        Parse metadata from a residue file.

        Args:
            file_content: Raw bytes of the residue file

        Returns:
            Dict with keys: composite, b1, parametrization, curve_count

        Raises:
            ValueError: If file format is invalid or missing required fields
        """
        try:
            content = file_content.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Residue file is not valid UTF-8: {e}")

        lines = [line.strip() for line in content.strip().split('\n') if line.strip()]
        if not lines:
            raise ValueError("Residue file is empty")

        # Parse first line to get metadata
        first_line = lines[0]

        # Check if it's GPU format (single-line with METHOD=ECM; PARAM=...; etc)
        if 'METHOD=ECM' in first_line and 'SIGMA=' in first_line and ';' in first_line:
            # GPU format - each line is a complete curve
            param_match = self.PARAM_PATTERN.search(first_line)
            b1_match = self.B1_PATTERN.search(first_line)
            n_match = self.N_PATTERN.search(first_line)

            if not all([param_match, b1_match, n_match]):
                raise ValueError("GPU format residue file missing required fields (PARAM, B1, or N)")

            # Assertions for type narrowing after the above check
            assert n_match is not None
            assert b1_match is not None
            assert param_match is not None

            composite = n_match.group(1)
            b1 = int(b1_match.group(1))
            parametrization = int(param_match.group(1))

            # Count curves (each line with SIGMA= is a curve)
            curve_count = sum(1 for line in lines if self.SIGMA_PATTERN.search(line))

        else:
            # CPU format - multi-line with header blocks
            # Look for N=, B1=, PARAM= in the first few lines
            composite = None
            b1 = None
            parametrization = None

            for line in lines[:20]:  # Check first 20 lines for metadata
                if composite is None:
                    n_match = self.N_PATTERN.match(line)
                    if n_match:
                        composite = n_match.group(1)
                if b1 is None:
                    b1_match = self.B1_PATTERN.match(line)
                    if b1_match:
                        b1 = int(b1_match.group(1))
                if parametrization is None and 'PARAM=' in line:
                    param_match = self.PARAM_PATTERN.search(line)
                    if param_match:
                        parametrization = int(param_match.group(1))

            if not all([composite, b1, parametrization is not None]):
                raise ValueError("CPU format residue file missing required fields (N, B1, or PARAM)")

            # Count curves (SIGMA= lines indicate individual curves)
            curve_count = sum(1 for line in lines if line.startswith('SIGMA='))

        if curve_count == 0:
            raise ValueError("No curves found in residue file")

        return {
            'composite': composite,
            'b1': b1,
            'parametrization': parametrization,
            'curve_count': curve_count
        }

    def calculate_checksum(self, file_content: bytes) -> str:
        """Calculate SHA-256 checksum of file content."""
        return hashlib.sha256(file_content).hexdigest()

    def store_residue_file(
        self,
        db: Session,
        file_content: bytes,
        client_id: str,
        stage1_attempt_id: Optional[int] = None
    ) -> ECMResidue:
        """
        Store a residue file and create database record.

        Residues don't expire by time - they remain available until:
        - The composite is fully factored
        - A stage 2 worker completes processing them

        Args:
            db: Database session
            file_content: Raw residue file bytes
            client_id: ID of the uploading client
            stage1_attempt_id: Optional ID of the stage 1 attempt to link

        Returns:
            ECMResidue database record

        Raises:
            ValueError: If file format is invalid or composite not found
        """
        # Parse metadata from file
        metadata = self.parse_residue_file(file_content)
        composite_number = metadata['composite']

        # Look up composite in database
        composite = db.query(Composite).filter(
            Composite.current_composite == composite_number
        ).first()

        if not composite:
            # Try matching by number field as fallback
            composite = db.query(Composite).filter(
                Composite.number == composite_number
            ).first()

        if not composite:
            raise ValueError(f"Composite {composite_number[:50]}... not found in database")

        # Generate unique filename
        import uuid
        file_uuid = str(uuid.uuid4())
        composite_dir = self.storage_dir / str(composite.id)
        composite_dir.mkdir(parents=True, exist_ok=True)
        file_path = composite_dir / f"{file_uuid}.txt"

        # Calculate checksum
        checksum = self.calculate_checksum(file_content)

        # Check for duplicate by checksum
        existing = db.query(ECMResidue).filter(
            ECMResidue.checksum == checksum
        ).first()
        if existing:
            raise ValueError(f"Duplicate residue file (checksum matches residue ID {existing.id})")

        # Write file to storage
        file_path.write_bytes(file_content)
        logger.info(f"Stored residue file: {file_path} ({len(file_content)} bytes)")

        # Create database record
        # expires_at is None for available residues (no time-based expiration)
        # It will be set when claimed (claim timeout)
        residue = ECMResidue(
            composite_id=composite.id,
            client_id=client_id,
            stage1_attempt_id=stage1_attempt_id,
            b1=metadata['b1'],
            parametrization=metadata['parametrization'],
            curve_count=metadata['curve_count'],
            storage_path=str(file_path),
            file_size_bytes=len(file_content),
            checksum=checksum,
            status='available',
            expires_at=None
        )

        db.add(residue)
        db.flush()  # Get the ID

        logger.info(
            f"Created residue record ID {residue.id}: "
            f"composite={composite.id}, B1={metadata['b1']}, "
            f"curves={metadata['curve_count']}, param={metadata['parametrization']}"
        )

        return residue

    def get_available_work(
        self,
        db: Session,
        client_id: str,
        min_target_tlevel: Optional[float] = None,
        max_target_tlevel: Optional[float] = None,
        min_priority: Optional[int] = None,
        min_b1: Optional[int] = None,
        max_b1: Optional[int] = None
    ) -> Optional[ECMResidue]:
        """
        Find an available residue for stage 2 processing.

        Args:
            db: Database session
            client_id: ID of requesting client
            min_target_tlevel: Minimum target t-level
            max_target_tlevel: Maximum target t-level
            min_priority: Minimum composite priority
            min_b1: Minimum B1 bound of residue
            max_b1: Maximum B1 bound of residue

        Returns:
            ECMResidue if found, None otherwise
        """
        # Available residues don't have time-based expiration
        # Only filter by status (and exclude factored composites)
        query = db.query(ECMResidue).join(
            Composite, ECMResidue.composite_id == Composite.id
        ).filter(
            ECMResidue.status == 'available',
            Composite.is_fully_factored == False  # noqa: E712
        )

        # Apply filters
        if min_target_tlevel is not None:
            query = query.filter(Composite.target_t_level >= min_target_tlevel)
        if max_target_tlevel is not None:
            query = query.filter(Composite.target_t_level <= max_target_tlevel)
        if min_priority is not None:
            query = query.filter(Composite.priority >= min_priority)
        if min_b1 is not None:
            query = query.filter(ECMResidue.b1 >= min_b1)
        if max_b1 is not None:
            query = query.filter(ECMResidue.b1 <= max_b1)

        # Prioritize by composite priority (descending), then by creation time (oldest first)
        query = query.order_by(
            Composite.priority.desc(),
            ECMResidue.created_at.asc()
        )

        residue = query.first()
        if residue:
            logger.info(f"Found available residue ID {residue.id} for client {client_id}")
        else:
            logger.info(f"No available residues for client {client_id}")

        return residue

    def claim_residue(
        self,
        db: Session,
        residue_id: int,
        client_id: str,
        claim_timeout_hours: int = 72  # 3 days default for large stage 2 work
    ) -> ECMResidue:
        """
        Claim a residue for stage 2 processing.

        Args:
            db: Database session
            residue_id: ID of residue to claim
            client_id: ID of claiming client
            claim_timeout_hours: Hours until claim expires (default 72h/3 days)

        Returns:
            Updated ECMResidue record

        Raises:
            ValueError: If residue not found or not available
        """
        residue = db.query(ECMResidue).filter(ECMResidue.id == residue_id).first()

        if not residue:
            raise ValueError(f"Residue {residue_id} not found")

        if residue.status != 'available':
            raise ValueError(f"Residue {residue_id} is not available (status: {residue.status})")

        residue.status = 'claimed'
        residue.claimed_at = datetime.utcnow()
        residue.claimed_by = client_id
        # Update expiration to claim timeout
        residue.expires_at = datetime.utcnow() + timedelta(hours=claim_timeout_hours)

        logger.info(f"Residue {residue_id} claimed by {client_id}")
        return residue

    def release_claim(self, db: Session, residue_id: int, client_id: str) -> ECMResidue:
        """
        Release a claimed residue back to available pool.

        Args:
            db: Database session
            residue_id: ID of residue to release
            client_id: ID of client releasing (must match claimer)

        Returns:
            Updated ECMResidue record

        Raises:
            ValueError: If residue not found, not claimed, or wrong client
        """
        residue = db.query(ECMResidue).filter(ECMResidue.id == residue_id).first()

        if not residue:
            raise ValueError(f"Residue {residue_id} not found")

        if residue.status != 'claimed':
            raise ValueError(f"Residue {residue_id} is not claimed (status: {residue.status})")

        if residue.claimed_by != client_id:
            raise ValueError(f"Residue {residue_id} is claimed by {residue.claimed_by}, not {client_id}")

        residue.status = 'available'
        residue.claimed_at = None
        residue.claimed_by = None
        # Clear expiration - available residues don't expire by time
        residue.expires_at = None

        logger.info(f"Residue {residue_id} released by {client_id}")
        return residue

    def complete_residue(
        self,
        db: Session,
        residue_id: int,
        stage2_attempt_id: int
    ) -> Tuple[ECMResidue, Optional[float]]:
        """
        Mark residue as completed after stage 2 finishes.

        This supersedes the stage 1 attempt and deletes the residue file.

        Args:
            db: Database session
            residue_id: ID of the completed residue
            stage2_attempt_id: ID of the stage 2 ECM attempt

        Returns:
            Tuple of (residue, new_t_level)
            - new_t_level: Updated t-level after supersession (if applicable)

        Raises:
            ValueError: If residue or attempt not found
        """
        residue = db.query(ECMResidue).filter(ECMResidue.id == residue_id).first()
        if not residue:
            raise ValueError(f"Residue {residue_id} not found")

        # Get the stage 2 attempt
        stage2_attempt = db.query(ECMAttempt).filter(ECMAttempt.id == stage2_attempt_id).first()
        if not stage2_attempt:
            raise ValueError(f"Stage 2 attempt {stage2_attempt_id} not found")

        # Validate that this is a legitimate stage 2 completion:
        # Must either find a factor OR complete at least 75% of the assigned curves
        has_factor = stage2_attempt.factor_found is not None
        min_curves_required = int(0.75 * residue.curve_count)
        curves_completed = stage2_attempt.curves_completed

        if not has_factor and curves_completed < min_curves_required:
            # Reject this completion and release the residue back to available
            residue.status = 'available'
            residue.claimed_at = None
            residue.claimed_by = None
            residue.expires_at = None
            db.flush()

            logger.warning(
                f"Rejected residue {residue_id} completion: stage2_attempt {stage2_attempt_id} "
                f"has no factor and curves_completed={curves_completed} < {min_curves_required} "
                f"(75% of {residue.curve_count}). Residue released back to pool."
            )
            raise ValueError(
                f"Invalid stage 2 completion: no factor found and only {curves_completed} curves "
                f"completed out of {residue.curve_count} assigned (minimum required: {min_curves_required}, 75%). "
                f"Residue {residue_id} has been released back to the available pool."
            )

        # Mark stage 1 attempt as superseded (if linked)
        if residue.stage1_attempt_id:
            stage1_attempt = db.query(ECMAttempt).filter(
                ECMAttempt.id == residue.stage1_attempt_id
            ).first()
            if stage1_attempt:
                stage1_attempt.superseded_by = stage2_attempt_id
                db.flush()  # Ensure supersession is visible to subsequent queries
                logger.info(
                    f"Marked stage 1 attempt {residue.stage1_attempt_id} as superseded by {stage2_attempt_id}"
                )

        # Update residue status
        residue.status = 'completed'
        residue.completed_at = datetime.utcnow()

        # Delete the residue file
        try:
            file_path = Path(residue.storage_path)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted residue file: {file_path}")
            else:
                logger.warning(f"Residue file not found for deletion: {file_path}")
        except Exception as e:
            logger.error(f"Error deleting residue file {residue.storage_path}: {e}")

        # Mark orphaned attempts from the same residue as superseded
        # These are partial attempts that were submitted but failed to complete the residue
        # (e.g., client interrupted, then another client completed the same residue)
        orphaned_attempts = db.query(ECMAttempt).filter(
            ECMAttempt.residue_checksum == residue.checksum,
            ECMAttempt.id != stage2_attempt_id,
            ECMAttempt.superseded_by.is_(None)  # Not already superseded
        ).all()

        for orphan in orphaned_attempts:
            orphan.superseded_by = stage2_attempt_id
            logger.info(
                f"Marked orphaned attempt {orphan.id} as superseded by {stage2_attempt_id} "
                f"(same residue checksum {residue.checksum[:16]}...)"
            )

        if orphaned_attempts:
            db.flush()  # Ensure supersession is visible to t-level calculation

        # Recalculate t-level for composite (excluding superseded attempts)
        new_t_level = self._recalculate_composite_t_level(db, residue.composite_id)

        logger.info(
            f"Completed residue {residue_id}: stage2_attempt={stage2_attempt_id}, "
            f"new_t_level={new_t_level}"
        )

        return residue, new_t_level

    def _recalculate_composite_t_level(self, db: Session, composite_id: int) -> Optional[float]:
        """
        Recalculate t-level for a composite, excluding superseded attempts.

        Args:
            db: Database session
            composite_id: ID of composite to recalculate

        Returns:
            New t-level value, or None if calculation fails
        """
        composite = db.query(Composite).filter(Composite.id == composite_id).first()
        if not composite:
            return None

        # Use the t-level calculator's recalculate method which already
        # excludes superseded attempts
        old_t_level = composite.current_t_level or 0.0
        new_t_level = self.t_level_calculator.recalculate_composite_t_level(db, composite)

        logger.info(
            f"Recalculated t-level for composite {composite_id}: "
            f"{old_t_level:.2f} -> {new_t_level:.2f}"
        )

        return new_t_level

    def get_residue_file_path(self, db: Session, residue_id: int) -> Optional[Path]:
        """
        Get the filesystem path for a residue file.

        Args:
            db: Database session
            residue_id: ID of residue

        Returns:
            Path to file, or None if not found
        """
        residue = db.query(ECMResidue).filter(ECMResidue.id == residue_id).first()
        if not residue:
            return None

        file_path = Path(residue.storage_path)
        if file_path.exists():
            return file_path

        logger.warning(f"Residue file not found: {file_path}")
        return None

    def suggest_b2_for_residue(self, db: Session, residue_id: int) -> Optional[int]:
        """
        Suggest an appropriate B2 value for stage 2 based on B1.

        Args:
            db: Database session
            residue_id: ID of residue

        Returns:
            Suggested B2 value, or None if residue not found
        """
        residue = db.query(ECMResidue).filter(ECMResidue.id == residue_id).first()
        if not residue:
            return None

        # Standard B2 = 100 * B1 is a common heuristic
        # For GPU work, even larger ratios can be beneficial
        suggested_b2 = residue.b1 * 100

        # Cap at a reasonable maximum (e.g., 10 trillion)
        max_b2 = 10_000_000_000_000
        suggested_b2 = min(suggested_b2, max_b2)

        return suggested_b2

    def cleanup_expired_claims(self, db: Session) -> int:
        """
        Release claims that have timed out (claimed but not completed in time).

        Only claimed residues have expiration times. Available residues don't expire.
        This releases the claim so another worker can pick up the work.

        Args:
            db: Database session

        Returns:
            Number of claims released
        """
        expired_claims = db.query(ECMResidue).filter(
            ECMResidue.status == 'claimed',
            ECMResidue.expires_at < datetime.utcnow()
        ).all()

        count = 0
        for residue in expired_claims:
            try:
                # Release the claim back to available (don't delete file)
                old_claimer = residue.claimed_by
                residue.status = 'available'
                residue.claimed_at = None
                residue.claimed_by = None
                residue.expires_at = None
                count += 1
                logger.info(f"Released expired claim on residue {residue.id} (was claimed by {old_claimer})")
            except Exception as e:
                logger.error(f"Error releasing claim on residue {residue.id}: {e}")

        if count > 0:
            logger.info(f"Released {count} expired claims")

        return count

    def cleanup_factored_composites(self, db: Session) -> int:
        """
        Clean up residues for composites that have been fully factored.

        Residues are no longer useful once their composite is factored.

        Args:
            db: Database session

        Returns:
            Number of residues cleaned up
        """
        # Find residues for fully factored composites
        residues_to_cleanup = db.query(ECMResidue).join(
            Composite, ECMResidue.composite_id == Composite.id
        ).filter(
            Composite.is_fully_factored == True,  # noqa: E712
            ECMResidue.status.in_(['available', 'claimed'])
        ).all()

        count = 0
        for residue in residues_to_cleanup:
            try:
                file_path = Path(residue.storage_path)
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted residue file for factored composite: {file_path}")

                residue.status = 'expired'  # Mark as cleaned up
                count += 1
            except Exception as e:
                logger.error(f"Error cleaning up residue {residue.id}: {e}")

        if count > 0:
            logger.info(f"Cleaned up {count} residues for factored composites")

        return count

    def get_stats(self, db: Session) -> Dict[str, int]:
        """
        Get statistics about residues in the system.

        Args:
            db: Database session

        Returns:
            Dict with counts by status and total pending curves
        """
        stats = {
            'total_available': 0,
            'total_claimed': 0,
            'total_completed': 0,
            'total_expired': 0,
            'total_curves_pending': 0
        }

        # Count by status
        status_counts = db.query(
            ECMResidue.status,
            func.count(ECMResidue.id)
        ).group_by(ECMResidue.status).all()

        for status, count in status_counts:
            if status == 'available':
                stats['total_available'] = count
            elif status == 'claimed':
                stats['total_claimed'] = count
            elif status == 'completed':
                stats['total_completed'] = count
            elif status == 'expired':
                stats['total_expired'] = count

        # Sum curves in available residues
        curves_sum = db.query(func.sum(ECMResidue.curve_count)).filter(
            ECMResidue.status == 'available'
        ).scalar()
        stats['total_curves_pending'] = curves_sum or 0

        return stats
