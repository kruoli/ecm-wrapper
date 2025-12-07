"""
Admin-specific residue management endpoints.
Provides functionality to delete residues and trigger cleanup of expired entries.
"""
import logging
import os
from fastapi import APIRouter, Depends, Header
from sqlalchemy.orm import Session

from ....database import get_db
from ....dependencies import verify_admin_key, get_residue_manager
from ....models.residues import ECMResidue
from ....services.residue_manager import ResidueManager
from ....utils.errors import get_or_404
from ....utils.transactions import transaction_scope

router = APIRouter()
logger = logging.getLogger(__name__)


@router.delete("/residues/{residue_id}")
async def delete_residue(
    residue_id: int,
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key),
    residue_manager: ResidueManager = Depends(get_residue_manager)
):
    """
    Admin endpoint to delete a residue (even if claimed/completed).

    This forcibly removes a residue record and its associated file
    regardless of status. Use with caution.

    Args:
        residue_id: ID of the residue to delete
        db: Database session
        _admin: Admin authentication check
        residue_manager: Residue manager service

    Returns:
        Success message with deleted residue info

    Raises:
        HTTPException: If residue not found
    """
    # Get residue
    residue = get_or_404(
        db.query(ECMResidue).filter(ECMResidue.id == residue_id).first(),
        "Residue",
        str(residue_id)
    )

    # Delete file if exists
    if residue.storage_path and os.path.exists(residue.storage_path):
        try:
            os.remove(residue.storage_path)
        except Exception as e:
            logger.warning(f"Failed to delete residue file {residue.storage_path}: {e}")

    # Store info for response
    composite_id = residue.composite_id
    status = residue.status

    # Delete database record within transaction
    with transaction_scope(db, "delete_residue"):
        db.delete(residue)

    return {
        "success": True,
        "message": f"Residue {residue_id} deleted",
        "residue_id": residue_id,
        "composite_id": composite_id,
        "previous_status": status
    }


@router.post("/residues/cleanup")
async def cleanup_expired_residues(
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key),
    residue_manager: ResidueManager = Depends(get_residue_manager)
):
    """
    Manually trigger cleanup of expired residues.

    This removes residues that have expired (expires_at < now)
    and deletes their associated files.

    Args:
        db: Database session
        _admin: Admin authentication check
        residue_manager: Residue manager service

    Returns:
        Cleanup summary with count of removed residues
    """
    # Use ResidueManager's cleanup method
    cleaned_count = residue_manager.cleanup_expired_residues(db)

    return {
        "success": True,
        "message": f"Cleaned up {cleaned_count} expired residue(s)",
        "cleaned_count": cleaned_count
    }
