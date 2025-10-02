from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List

from ...database import get_db
from ...schemas.composites import (
    CompositeStats, EffortLevel, ECMWorkSummary,
    BatchStatusRequest, BatchStatusResponse, CompositeBatchStatus
)
from ...models import Composite, ECMAttempt, Factor, ProjectComposite, Project
from ...services.composites import CompositeService

router = APIRouter()

@router.get("/stats/{composite}", response_model=CompositeStats)
async def get_composite_stats(
    composite: str = Path(..., description="The composite number to get stats for"),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive statistics for a composite number.
    
    Returns information about:
    - Composite properties (bit/digit length, factorization status)
    - All known factors
    - Summary of factorization work performed
    - Associated projects
    """
    # Get composite from database
    comp = CompositeService.get_composite_by_number(db, composite)
    if not comp:
        raise HTTPException(status_code=404, detail="Composite not found in database")
    
    # Get all factors
    factors = db.query(Factor).filter(Factor.composite_id == comp.id).all()
    factors_list = [f.factor for f in factors]
    
    # Determine status
    if comp.is_prime:
        status = "prime"
    elif comp.is_fully_factored:
        status = "fully_factored"
    else:
        status = "composite"
    
    # Get ECM work summary
    attempts = db.query(ECMAttempt).filter(ECMAttempt.composite_id == comp.id).all()
    
    total_attempts = len(attempts)
    total_curves = sum(attempt.curves_completed for attempt in attempts)
    last_attempt = max((attempt.created_at for attempt in attempts), default=None)
    
    # Group efforts by B1 level
    effort_groups = {}
    for attempt in attempts:
        b1 = attempt.b1
        if b1 not in effort_groups:
            effort_groups[b1] = 0
        effort_groups[b1] += attempt.curves_completed
    
    effort_by_level = [
        EffortLevel(b1=b1, curves=curves) 
        for b1, curves in sorted(effort_groups.items())
    ]
    
    ecm_work = ECMWorkSummary(
        total_attempts=total_attempts,
        total_curves=total_curves,
        effort_by_level=effort_by_level,
        last_attempt=last_attempt
    )
    
    # Get associated projects
    project_links = db.query(ProjectComposite).filter(
        ProjectComposite.composite_id == comp.id
    ).all()
    
    project_names = []
    for link in project_links:
        project = db.query(Project).filter(Project.id == link.project_id).first()
        if project:
            project_names.append(project.name)
    
    return CompositeStats(
        composite=comp.number,
        current_composite=comp.current_composite,
        digit_length=comp.digit_length,
        has_snfs_form=comp.has_snfs_form,
        snfs_difficulty=comp.snfs_difficulty,
        target_t_level=comp.target_t_level,
        current_t_level=comp.current_t_level,
        priority=comp.priority,
        status=status,
        factors_found=factors_list,
        ecm_work=ecm_work,
        projects=project_names
    )


@router.post("/composites/batch-status", response_model=BatchStatusResponse)
async def get_batch_composite_status(
    request: BatchStatusRequest,
    db: Session = Depends(get_db)
):
    """
    Get t-level status for multiple composites in a single request.

    Returns current and target t-levels for each composite number.
    If a composite is not found in the database, returns found=False.
    """
    results = []

    for number in request.numbers:
        # Try to find composite by number
        comp = db.query(Composite).filter(Composite.number == number).first()

        if comp:
            results.append(CompositeBatchStatus(
                number=number,
                target_t_level=comp.target_t_level,
                current_t_level=comp.current_t_level,
                digit_length=comp.digit_length,
                has_snfs_form=comp.has_snfs_form,
                snfs_difficulty=comp.snfs_difficulty,
                found=True
            ))
        else:
            results.append(CompositeBatchStatus(
                number=number,
                target_t_level=None,
                current_t_level=None,
                digit_length=None,
                has_snfs_form=None,
                snfs_difficulty=None,
                found=False
            ))

    return BatchStatusResponse(composites=results)