"""
API endpoints for ECM residue management (decoupled two-stage ECM).

Provides endpoints for:
- Uploading stage 1 residue files
- Requesting stage 2 work
- Downloading residue files
- Completing stage 2 work
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Header, Query
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime
import logging

from ...database import get_db
from ...models.residues import ECMResidue
from ...models.composites import Composite
from ...services.residue_manager import ResidueManager
from ...schemas.residues import (
    ResidueUploadResponse,
    ResidueWorkResponse,
    ResidueCompleteRequest,
    ResidueCompleteResponse,
    ResidueInfoResponse,
    ResidueStatsResponse
)
from ...utils.transactions import transaction_scope

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize residue manager
residue_manager = ResidueManager()


@router.post("/upload", response_model=ResidueUploadResponse)
async def upload_residue(
    file: UploadFile = File(..., description="ECM residue file from stage 1"),
    client_id: str = Header(..., alias="X-Client-ID", description="Client identifier"),
    stage1_attempt_id: Optional[int] = Query(None, description="ID of stage 1 ECM attempt to link"),
    expiry_days: int = Query(7, ge=1, le=30, description="Days until residue expires"),
    db: Session = Depends(get_db)
):
    """
    Upload a residue file after completing stage 1 ECM.

    The server parses the file to extract:
    - Composite number (N=)
    - B1 parameter
    - Parametrization (PARAM=)
    - Curve count

    Args:
        file: The residue file content
        client_id: ID of the uploading client (header)
        stage1_attempt_id: Optional ID of the stage 1 attempt for supersession tracking
        expiry_days: Days until residue expires if not consumed
        db: Database session

    Returns:
        Parsed metadata and residue ID
    """
    with transaction_scope(db, "upload_residue"):
        # Read file content
        content = await file.read()

        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded"
            )

        # Limit file size (50 MB max)
        max_size = 50 * 1024 * 1024
        if len(content) > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size is {max_size // (1024*1024)} MB"
            )

        try:
            # Store residue and create database record
            residue = residue_manager.store_residue_file(
                db=db,
                file_content=content,
                client_id=client_id,
                stage1_attempt_id=stage1_attempt_id,
                expiry_days=expiry_days
            )

            # Get composite for response
            composite = db.query(Composite).filter(
                Composite.id == residue.composite_id
            ).first()

            logger.info(
                f"Client {client_id} uploaded residue ID {residue.id} "
                f"for composite {residue.composite_id}"
            )

            return ResidueUploadResponse(
                residue_id=residue.id,
                composite_id=residue.composite_id,
                composite=composite.current_composite if composite else "",
                b1=residue.b1,
                parametrization=residue.parametrization,
                curve_count=residue.curve_count,
                file_size_bytes=residue.file_size_bytes,
                expires_at=residue.expires_at,
                message=f"Residue uploaded successfully. {residue.curve_count} curves ready for stage 2."
            )

        except ValueError as e:
            logger.warning(f"Failed to process residue upload from {client_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Error uploading residue from {client_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store residue file"
            )


@router.get("/work", response_model=ResidueWorkResponse)
async def get_residue_work(
    client_id: str = Header(..., alias="X-Client-ID", description="Client identifier"),
    min_digits: Optional[int] = Query(None, ge=1, description="Minimum composite digit size"),
    max_digits: Optional[int] = Query(None, ge=1, description="Maximum composite digit size"),
    min_priority: Optional[int] = Query(None, description="Minimum composite priority"),
    claim_timeout_hours: int = Query(24, ge=1, le=168, description="Hours until claim expires"),
    db: Session = Depends(get_db)
):
    """
    Request stage 2 work (an available residue file).

    Finds an available residue, claims it for this client, and returns
    the information needed to process stage 2.

    Args:
        client_id: ID of the requesting client
        min_digits: Minimum composite digit size filter
        max_digits: Maximum composite digit size filter
        min_priority: Minimum composite priority filter
        claim_timeout_hours: Hours until claim expires
        db: Database session

    Returns:
        Residue details and download URL, or message if none available
    """
    with transaction_scope(db, "get_residue_work"):
        # Find available residue
        residue = residue_manager.get_available_work(
            db=db,
            client_id=client_id,
            min_digits=min_digits,
            max_digits=max_digits,
            min_priority=min_priority
        )

        if not residue:
            return ResidueWorkResponse(
                message="No residues available for stage 2 processing"
            )

        # Claim the residue
        try:
            residue = residue_manager.claim_residue(
                db=db,
                residue_id=residue.id,
                client_id=client_id,
                claim_timeout_hours=claim_timeout_hours
            )
        except ValueError as e:
            logger.warning(f"Failed to claim residue {residue.id}: {e}")
            return ResidueWorkResponse(
                message=f"Failed to claim residue: {str(e)}"
            )

        # Get composite details
        composite = db.query(Composite).filter(
            Composite.id == residue.composite_id
        ).first()

        # Suggest B2
        suggested_b2 = residue_manager.suggest_b2_for_residue(db, residue.id)

        logger.info(
            f"Client {client_id} claimed residue ID {residue.id} "
            f"for composite {residue.composite_id}"
        )

        return ResidueWorkResponse(
            residue_id=residue.id,
            composite_id=residue.composite_id,
            composite=composite.current_composite if composite else "",
            digit_length=composite.digit_length if composite else 0,
            b1=residue.b1,
            parametrization=residue.parametrization,
            curve_count=residue.curve_count,
            stage1_attempt_id=residue.stage1_attempt_id,
            download_url=f"/api/v1/residues/{residue.id}/download",
            suggested_b2=suggested_b2,
            expires_at=residue.expires_at,
            message=f"Claimed {residue.curve_count} curves for stage 2 (B1={residue.b1})"
        )


@router.get("/{residue_id}/download")
async def download_residue(
    residue_id: int,
    client_id: str = Header(..., alias="X-Client-ID", description="Client identifier"),
    db: Session = Depends(get_db)
):
    """
    Download a residue file for stage 2 processing.

    Only the client who claimed the residue can download it.

    Args:
        residue_id: ID of the residue to download
        client_id: ID of the requesting client (must match claimer)
        db: Database session

    Returns:
        File content as response
    """
    residue = db.query(ECMResidue).filter(ECMResidue.id == residue_id).first()

    if not residue:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Residue {residue_id} not found"
        )

    # Verify client has claimed this residue
    if residue.claimed_by != client_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Residue {residue_id} is not claimed by client {client_id}"
        )

    # Get file path
    file_path = residue_manager.get_residue_file_path(db, residue_id)
    if not file_path or not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Residue file not found on disk"
        )

    logger.info(f"Client {client_id} downloading residue {residue_id}")

    return FileResponse(
        path=str(file_path),
        media_type="text/plain",
        filename=f"residue_{residue_id}.txt"
    )


@router.post("/{residue_id}/complete", response_model=ResidueCompleteResponse)
async def complete_residue(
    residue_id: int,
    request: ResidueCompleteRequest,
    client_id: str = Header(..., alias="X-Client-ID", description="Client identifier"),
    db: Session = Depends(get_db)
):
    """
    Mark a residue as completed after stage 2 finishes.

    This:
    1. Links the stage 2 attempt to supersede the stage 1 attempt
    2. Recalculates the composite's t-level (excluding superseded S1)
    3. Deletes the residue file
    4. Updates residue status to 'completed'

    Args:
        residue_id: ID of the completed residue
        request: Contains the stage 2 attempt ID
        client_id: ID of the client completing the work
        db: Database session

    Returns:
        Completion confirmation with updated t-level
    """
    with transaction_scope(db, "complete_residue"):
        residue = db.query(ECMResidue).filter(ECMResidue.id == residue_id).first()

        if not residue:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Residue {residue_id} not found"
            )

        # Verify client has claimed this residue
        if residue.claimed_by != client_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Residue {residue_id} is not claimed by client {client_id}"
            )

        try:
            completed_residue, new_t_level = residue_manager.complete_residue(
                db=db,
                residue_id=residue_id,
                stage2_attempt_id=request.stage2_attempt_id
            )

            logger.info(
                f"Client {client_id} completed residue {residue_id} "
                f"with stage2_attempt {request.stage2_attempt_id}"
            )

            return ResidueCompleteResponse(
                residue_id=completed_residue.id,
                stage1_attempt_id=completed_residue.stage1_attempt_id,
                stage2_attempt_id=request.stage2_attempt_id,
                composite_id=completed_residue.composite_id,
                new_t_level=new_t_level,
                message=f"Stage 2 complete. Residue file deleted. T-level updated to {new_t_level:.2f}" if new_t_level else "Stage 2 complete. Residue file deleted."
            )

        except ValueError as e:
            logger.warning(f"Failed to complete residue {residue_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )


@router.delete("/{residue_id}/claim")
async def abandon_residue_claim(
    residue_id: int,
    client_id: str = Header(..., alias="X-Client-ID", description="Client identifier"),
    db: Session = Depends(get_db)
):
    """
    Release a claimed residue back to the available pool.

    Args:
        residue_id: ID of the residue to release
        client_id: ID of the client releasing (must match claimer)
        db: Database session

    Returns:
        Confirmation message
    """
    with transaction_scope(db, "abandon_residue_claim"):
        try:
            residue = residue_manager.release_claim(db, residue_id, client_id)

            logger.info(f"Client {client_id} released claim on residue {residue_id}")

            return {
                "residue_id": residue_id,
                "status": "available",
                "message": "Residue claim released successfully"
            }

        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )


@router.get("/{residue_id}", response_model=ResidueInfoResponse)
async def get_residue_info(
    residue_id: int,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a residue.

    Args:
        residue_id: ID of the residue
        db: Database session

    Returns:
        Residue details
    """
    residue = db.query(ECMResidue).filter(ECMResidue.id == residue_id).first()

    if not residue:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Residue {residue_id} not found"
        )

    composite = db.query(Composite).filter(
        Composite.id == residue.composite_id
    ).first()

    return ResidueInfoResponse(
        residue_id=residue.id,
        composite_id=residue.composite_id,
        composite=composite.current_composite if composite else "",
        client_id=residue.client_id,
        stage1_attempt_id=residue.stage1_attempt_id,
        b1=residue.b1,
        parametrization=residue.parametrization,
        curve_count=residue.curve_count,
        file_size_bytes=residue.file_size_bytes,
        status=residue.status,
        created_at=residue.created_at,
        expires_at=residue.expires_at,
        claimed_at=residue.claimed_at,
        claimed_by=residue.claimed_by,
        completed_at=residue.completed_at
    )


@router.get("/stats/summary", response_model=ResidueStatsResponse)
async def get_residue_stats(db: Session = Depends(get_db)):
    """
    Get statistics about residues in the system.

    Returns:
        Counts by status and total pending curves
    """
    stats = residue_manager.get_stats(db)
    return ResidueStatsResponse(**stats)
