"""
Admin-specific residue management endpoints.
Provides functionality to delete residues and trigger cleanup of expired entries.
"""
import secrets
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ....config import get_settings
from ....database import get_db
from ....models.residues import ECMResidue
from ....services.residue_manager import ResidueManager

router = APIRouter()


def verify_admin_key(x_admin_key: Optional[str] = None) -> bool:
    """
    Verify admin API key with constant-time comparison.

    Args:
        x_admin_key: API key from X-Admin-Key header

    Returns:
        True if valid

    Raises:
        HTTPException: If key is invalid
    """
    settings = get_settings()
    if not x_admin_key or not secrets.compare_digest(x_admin_key, settings.admin_api_key):
        raise HTTPException(status_code=401, detail="Invalid admin key")
    return True


@router.delete("/residues/{residue_id}")
async def delete_residue(
    residue_id: int,
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_key)
):
    """
    Admin endpoint to delete a residue (even if claimed/completed).

    This forcibly removes a residue record and its associated file
    regardless of status. Use with caution.

    Args:
        residue_id: ID of the residue to delete
        db: Database session
        _admin: Admin authentication check

    Returns:
        Success message with deleted residue info

    Raises:
        HTTPException: If residue not found
    """
    # Get residue
    residue = db.query(ECMResidue).filter(ECMResidue.id == residue_id).first()
    if not residue:
        raise HTTPException(status_code=404, detail=f"Residue {residue_id} not found")

    residue_manager = ResidueManager()

    # Delete file if exists
    if residue.storage_path:
        import os
        if os.path.exists(residue.storage_path):
            try:
                os.remove(residue.storage_path)
            except Exception as e:
                # Log error but continue with database deletion
                print(f"Warning: Failed to delete residue file {residue.storage_path}: {e}")

    # Store info for response
    composite_id = residue.composite_id
    status = residue.status

    # Delete database record
    db.delete(residue)
    db.commit()

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
    _admin: bool = Depends(verify_admin_key)
):
    """
    Manually trigger cleanup of expired residues.

    This removes residues that have expired (expires_at < now)
    and deletes their associated files.

    Args:
        db: Database session
        _admin: Admin authentication check

    Returns:
        Cleanup summary with count of removed residues
    """
    residue_manager = ResidueManager()

    # Use ResidueManager's cleanup method
    cleaned_count = residue_manager.cleanup_expired_residues(db)

    return {
        "success": True,
        "message": f"Cleaned up {cleaned_count} expired residue(s)",
        "cleaned_count": cleaned_count
    }
