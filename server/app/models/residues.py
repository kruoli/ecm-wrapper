from sqlalchemy import Column, Integer, String, BigInteger, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
from .base import Base, TimestampMixin


class ECMResidue(Base, TimestampMixin):
    """
    Tracks ECM stage 1 residue files for decoupled two-stage ECM processing.

    Workflow:
    1. GPU worker completes stage 1, submits results (gets immediate t-level credit)
    2. GPU worker uploads residue file, creating this record
    3. CPU worker requests stage 2 work, claims this residue
    4. CPU worker downloads file, runs stage 2, submits results
    5. CPU worker marks residue complete, file is deleted, stage 1 attempt is superseded
    """
    __tablename__ = "ecm_residues"

    id = Column(Integer, primary_key=True, index=True)
    composite_id = Column(Integer, ForeignKey("composites.id"), nullable=False)

    # Who uploaded this residue
    client_id = Column(String(255), nullable=False, index=True)

    # Link to the original stage 1 ECM attempt
    # This will be marked as superseded when stage 2 completes
    stage1_attempt_id = Column(Integer, ForeignKey("ecm_attempts.id"), nullable=True)

    # ECM parameters (parsed from residue file)
    b1 = Column(BigInteger, nullable=False)
    parametrization = Column(Integer, nullable=False)  # 0, 1, 2, or 3
    curve_count = Column(Integer, nullable=False)

    # File storage info
    storage_path = Column(String(512), nullable=False, unique=True)  # Filesystem path
    file_size_bytes = Column(Integer, nullable=False)
    checksum = Column(String(64), nullable=False)  # SHA-256 of file content

    # Lifecycle status
    # available: ready for stage 2 work
    # claimed: assigned to a client for stage 2
    # completed: stage 2 finished, file deleted
    # expired: unclaimed too long, cleaned up
    status = Column(String(20), default='available', nullable=False)

    # Timing
    expires_at = Column(DateTime, nullable=False)  # Auto-cleanup if not consumed
    claimed_at = Column(DateTime, nullable=True)
    claimed_by = Column(String(255), nullable=True)  # Client ID of stage 2 worker
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    composite = relationship("Composite")
    stage1_attempt = relationship("ECMAttempt", foreign_keys=[stage1_attempt_id])

    @classmethod
    def default_expiry(cls) -> datetime:
        """Default expiration: 7 days from now."""
        return datetime.utcnow() + timedelta(days=7)

    # Indexes for common queries
    __table_args__ = (
        Index('ix_ecm_residues_status', 'status'),
        Index('ix_ecm_residues_composite_status', 'composite_id', 'status'),
        Index('ix_ecm_residues_expires_at', 'expires_at'),
    )
