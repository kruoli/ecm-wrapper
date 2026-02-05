from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import uuid
import logging
import json

from ...database import get_db
from ...dependencies import get_t_level_calculator
from ...models.composites import Composite
from ...models.attempts import ECMAttempt
from ...models.work_assignments import WorkAssignment
from ...models.residues import ECMResidue
from ...services.t_level_calculator import TLevelCalculator
from ...utils.transactions import transaction_scope
from ...config import get_settings
from ...constants import ECM_BOUNDS, OPTIMAL_B1_TABLE, get_b1_above_tlevel

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


@router.get("/ecm-work")
async def get_ecm_work(
    client_id: str,
    priority: Optional[int] = None,
    min_target_tlevel: Optional[float] = None,
    max_target_tlevel: Optional[float] = None,
    timeout_days: int = 1,
    work_type: str = "standard",
    db: Session = Depends(get_db),
    t_level_calc: TLevelCalculator = Depends(get_t_level_calculator)
):
    """
    Get ECM work assignment with t-level targeting.

    This endpoint returns an incomplete composite (current_t < target_t)
    that matches the filter criteria, sorted by work_type strategy.

    Exclusions:
    - Composites with active work assignments
    - Composites with pending residues (status='available' or 'claimed')
      to prevent duplicate stage 1 work while waiting for stage 2 processing

    Args:
        client_id: Unique identifier for the requesting client
        priority: Minimum priority level (filters for priority >= this value)
        min_target_tlevel: Minimum target t-level (filters for target_t_level >= this value)
        max_target_tlevel: Maximum target t-level (filters for target_t_level <= this value)
        timeout_days: Work assignment expiration in days (default: 1)
        work_type: Work assignment strategy - "standard" (easiest/lowest target t-level first) or "progressive" (least ECM done first)
        db: Database session

    Returns:
        JSON response with work assignment details or explanation if no work available
    """
    with transaction_scope(db, "get_ecm_work"):
        # Validate work_type parameter
        if work_type not in ["standard", "progressive"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid work_type: {work_type}. Must be 'standard' or 'progressive'"
            )

        # Check if client has too much active work (exclude expired assignments)
        active_work_count = db.query(WorkAssignment).filter(
            and_(
                WorkAssignment.client_id == client_id,
                WorkAssignment.status.in_(['assigned', 'claimed', 'running']),
                WorkAssignment.expires_at > datetime.utcnow()
            )
        ).count()

        if active_work_count >= settings.max_work_items_per_client:
            response_data: Dict[str, Any] = {
                "work_id": None,
                "composite_id": None,
                "composite": None,
                "digit_length": None,
                "current_t_level": None,
                "target_t_level": None,
                "expires_at": None,
                "message": f"Client has {active_work_count} active work assignments (max: {settings.max_work_items_per_client})"
            }
            content = json.dumps(response_data, default=str) + "\n"
            return Response(content=content, media_type="application/json")

        # Build query for suitable composites
        # current_t_level now includes prior_t_level (calculated using -w flag)
        # Use ecm_progress < 1.0 which leverages the indexed generated column
        query = db.query(Composite).filter(
            and_(
                Composite.is_active == True,  # Only assign active composites
                Composite.is_fully_factored == False,
                or_(Composite.is_complete.is_(None), Composite.is_complete == False),
                Composite.ecm_progress.isnot(None),  # Has a target set
                Composite.ecm_progress < 1.0  # Not yet complete (indexed)
            )
        )

        # Apply priority filter
        if priority is not None:
            query = query.filter(Composite.priority >= priority)

        # Apply target t-level filters
        if min_target_tlevel is not None:
            query = query.filter(Composite.target_t_level >= min_target_tlevel)
        if max_target_tlevel is not None:
            query = query.filter(Composite.target_t_level <= max_target_tlevel)

        # Exclude composites with active work assignments
        active_work_composites = db.query(WorkAssignment.composite_id).filter(
            WorkAssignment.status.in_(['assigned', 'claimed', 'running'])
        ).subquery()

        query = query.filter(~Composite.id.in_(active_work_composites))  # type: ignore[arg-type]

        # Exclude composites with pending residues (stage 1 done, stage 2 not yet completed)
        # This prevents duplicate stage 1 work when residues are waiting to be processed
        pending_residue_composites = db.query(ECMResidue.composite_id).filter(
            ECMResidue.status.in_(['available', 'claimed'])
        ).subquery()

        query = query.filter(~Composite.id.in_(pending_residue_composites))  # type: ignore[arg-type]

        # Apply sorting strategy based on work_type
        if work_type == "progressive":
            # Progressive: prioritize composites with least ECM work done
            composite = query.order_by(
                Composite.current_t_level.asc(),
                Composite.target_t_level.asc(),
                Composite.digit_length.asc()
            ).first()
        else:  # "standard"
            # Standard: prioritize easiest composites first (by target t-level, which accounts for SNFS)
            composite = query.order_by(
                Composite.target_t_level.asc(),
                Composite.created_at.asc()
            ).first()

        # No work available
        if not composite:
            response_data = {
                "work_id": None,
                "composite_id": None,
                "composite": None,
                "digit_length": None,
                "current_t_level": None,
                "target_t_level": None,
                "expires_at": None,
                "message": "No suitable work available matching criteria"
            }
            content = json.dumps(response_data, default=str) + "\n"
            return Response(content=content, media_type="application/json")

        # Get previous ECM attempts for parameter calculation
        previous_attempts = db.query(ECMAttempt).filter(
            ECMAttempt.composite_id == composite.id
        ).all()

        # Calculate suggested ECM parameters using t-level targeting
        # Note: target_t_level is guaranteed non-None by the filter above
        # current_t_level now includes prior_t_level
        try:
            suggestion = t_level_calc.suggest_next_ecm_parameters(
                composite.target_t_level or 0.0,  # Default to 0 if None (shouldn't happen due to filter)
                composite.current_t_level,  # Includes prior_t_level
                composite.digit_length
            )

            if suggestion['status'] == 'target_reached':
                # Use escalated parameters
                b1, b2, curves = _get_escalated_parameters(composite.digit_length, previous_attempts)
            else:
                b1, b2, curves = suggestion['b1'], suggestion['b2'], suggestion['curves']

        except Exception as e:
            logger.warning(f"T-level calculation failed for composite {composite.id}: {e}")
            # Fallback to basic parameters
            b1, b2, curves = 50000, 12500000, 100

        # Create work assignment
        work_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(days=timeout_days)

        work_assignment = WorkAssignment(
            id=work_id,
            composite_id=composite.id,
            client_id=client_id,
            method='ecm',
            b1=b1,
            b2=b2,
            curves_requested=curves,
            expires_at=expires_at,
            status='assigned'
        )

        db.add(work_assignment)
        db.flush()

        prior_t = composite.prior_t_level
        current_t = composite.current_t_level

        if prior_t:
            logger.info(f"Created ECM work assignment {work_id} for client {client_id}: "
                       f"{composite.digit_length}-digit composite, "
                       f"t{current_t:.1f} → t{composite.target_t_level:.1f} "
                       f"(includes prior: t{prior_t:.1f})")
        else:
            logger.info(f"Created ECM work assignment {work_id} for client {client_id}: "
                       f"{composite.digit_length}-digit composite, "
                       f"t{current_t:.1f} → t{composite.target_t_level:.1f}")

        # Build message based on work type strategy
        if work_type == "progressive":
            message = f"Assigned composite with least ECM work (t{current_t:.1f})"
        else:
            message = f"Assigned easiest incomplete composite (target: t{composite.target_t_level:.1f})"

        response_data = {
            "work_id": work_id,
            "composite_id": composite.id,
            "composite": composite.current_composite,
            "digit_length": composite.digit_length,
            "current_t_level": current_t,
            "prior_t_level": prior_t,
            "target_t_level": composite.target_t_level,
            "expires_at": expires_at.isoformat() if expires_at else None,
            "message": message
        }
        content = json.dumps(response_data, default=str) + "\n"
        return Response(content=content, media_type="application/json")


def _get_escalated_parameters(digit_length: int, previous_attempts: list) -> tuple:
    """Get escalated ECM parameters when target t-level is reached."""
    max_b1_attempted = max((attempt.b1 for attempt in previous_attempts if attempt.method == 'ecm'), default=0)
    escalated_b1 = max_b1_attempted * 3

    # Find next level beyond what's been tried
    for max_digits, b1, b2, curves in ECM_BOUNDS:
        if digit_length <= max_digits and b1 > escalated_b1:
            return b1, b2, min(curves // 5, 200)

    # Fallback to highest available
    return ECM_BOUNDS[-1][1], ECM_BOUNDS[-1][2], 100


@router.get("/p1-work")
async def get_p1_work(
    client_id: str,
    method: str = "p1",
    priority: Optional[int] = None,
    min_target_tlevel: Optional[float] = None,
    max_target_tlevel: Optional[float] = None,
    timeout_days: int = 1,
    work_type: str = "standard",
    db: Session = Depends(get_db),
):
    """
    Get P-1/P+1 work assignment.

    Assigns composites that haven't had PM1/PP1 done at the required B1 level.
    B1 is calculated as one step above the composite's target t-level in the
    optimal B1 table.

    Unlike /ecm-work, this endpoint does NOT filter by ecm_progress < 1.0,
    since PM1/PP1 sweeps are valuable even on composites that have reached
    their ECM target.

    Args:
        client_id: Unique identifier for the requesting client
        method: Which methods to check - "pm1" (P-1 only), "pp1" (P+1 only),
                or "p1" (both P-1 and P+1, default)
        priority: Minimum priority level
        min_target_tlevel: Minimum target t-level filter
        max_target_tlevel: Maximum target t-level filter
        timeout_days: Work assignment expiration in days (default: 1)
        work_type: "standard" (easiest first) or "progressive" (least work first)

    Returns:
        JSON response with work assignment details including pm1_b1/pp1_b1
    """
    with transaction_scope(db, "get_p1_work"):
        # Validate parameters
        if method not in ("pm1", "pp1", "p1"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid method: {method}. Must be 'pm1', 'pp1', or 'p1'"
            )
        if work_type not in ("standard", "progressive"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid work_type: {work_type}. Must be 'standard' or 'progressive'"
            )

        check_pm1 = method in ("pm1", "p1")
        check_pp1 = method in ("pp1", "p1")

        # Check active work limit
        active_work_count = db.query(WorkAssignment).filter(
            and_(
                WorkAssignment.client_id == client_id,
                WorkAssignment.status.in_(['assigned', 'claimed', 'running']),
                WorkAssignment.expires_at > datetime.utcnow()
            )
        ).count()

        if active_work_count >= settings.max_work_items_per_client:
            response_data: Dict[str, Any] = {
                "work_id": None,
                "composite_id": None,
                "composite": None,
                "digit_length": None,
                "current_t_level": None,
                "target_t_level": None,
                "pm1_b1": None,
                "pp1_b1": None,
                "expires_at": None,
                "message": f"Client has {active_work_count} active work assignments (max: {settings.max_work_items_per_client})"
            }
            content = json.dumps(response_data, default=str) + "\n"
            return Response(content=content, media_type="application/json")

        # Build candidate query - no ecm_progress filter (PM1/PP1 valuable regardless)
        query = db.query(Composite).filter(
            and_(
                Composite.is_active == True,
                Composite.is_fully_factored == False,
                or_(Composite.is_complete.is_(None), Composite.is_complete == False),
                Composite.target_t_level.isnot(None),
            )
        )

        if priority is not None:
            query = query.filter(Composite.priority >= priority)
        if min_target_tlevel is not None:
            query = query.filter(Composite.target_t_level >= min_target_tlevel)
        if max_target_tlevel is not None:
            query = query.filter(Composite.target_t_level <= max_target_tlevel)

        # Exclude composites with active work assignments
        active_work_composites = db.query(WorkAssignment.composite_id).filter(
            WorkAssignment.status.in_(['assigned', 'claimed', 'running'])
        ).subquery()
        query = query.filter(~Composite.id.in_(active_work_composites))  # type: ignore[arg-type]

        # Apply sorting strategy
        if work_type == "progressive":
            query = query.order_by(
                Composite.current_t_level.asc(),
                Composite.target_t_level.asc(),
                Composite.digit_length.asc()
            )
        else:
            query = query.order_by(
                Composite.target_t_level.asc(),
                Composite.created_at.asc()
            )

        # Fetch a batch of candidates and check each for PM1/PP1 coverage
        candidates = query.limit(50).all()

        assigned_composite = None
        computed_b1 = 0

        for composite in candidates:
            b1 = get_b1_above_tlevel(composite.target_t_level or 35.0)
            needs_work = False

            if check_pm1:
                # Check if there's a PM1 attempt with b1 >= computed b1
                has_pm1 = db.query(
                    db.query(ECMAttempt).filter(
                        and_(
                            ECMAttempt.composite_id == composite.id,
                            ECMAttempt.method == 'pm1',
                            ECMAttempt.b1 >= b1,
                        )
                    ).exists()
                ).scalar()
                if not has_pm1:
                    needs_work = True

            if check_pp1 and not needs_work:
                # Check if there's a PP1 attempt with b1 >= computed b1
                has_pp1 = db.query(
                    db.query(ECMAttempt).filter(
                        and_(
                            ECMAttempt.composite_id == composite.id,
                            ECMAttempt.method == 'pp1',
                            ECMAttempt.b1 >= b1,
                        )
                    ).exists()
                ).scalar()
                if not has_pp1:
                    needs_work = True

            if needs_work:
                assigned_composite = composite
                computed_b1 = b1
                break

        if not assigned_composite:
            response_data = {
                "work_id": None,
                "composite_id": None,
                "composite": None,
                "digit_length": None,
                "current_t_level": None,
                "target_t_level": None,
                "pm1_b1": None,
                "pp1_b1": None,
                "expires_at": None,
                "message": "No composites need P-1/P+1 work matching criteria"
            }
            content = json.dumps(response_data, default=str) + "\n"
            return Response(content=content, media_type="application/json")

        # Create work assignment
        work_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(days=timeout_days)

        work_assignment = WorkAssignment(
            id=work_id,
            composite_id=assigned_composite.id,
            client_id=client_id,
            method=method,
            b1=computed_b1,
            b2=0,  # PM1/PP1 uses GMP-ECM default B2
            curves_requested=1,  # Client decides actual curve count
            expires_at=expires_at,
            status='assigned'
        )

        db.add(work_assignment)
        db.flush()

        logger.info(
            f"Created P1 work assignment {work_id} for client {client_id}: "
            f"{assigned_composite.digit_length}-digit composite, "
            f"method={method}, B1={computed_b1}, "
            f"target_t={assigned_composite.target_t_level:.1f}"
        )

        response_data = {
            "work_id": work_id,
            "composite_id": assigned_composite.id,
            "composite": assigned_composite.current_composite,
            "digit_length": assigned_composite.digit_length,
            "current_t_level": assigned_composite.current_t_level,
            "target_t_level": assigned_composite.target_t_level,
            "pm1_b1": computed_b1 if check_pm1 else None,
            "pp1_b1": computed_b1 if check_pp1 else None,
            "expires_at": expires_at.isoformat() if expires_at else None,
            "message": f"Assigned {assigned_composite.digit_length}-digit composite for {method.upper()} sweep (B1={computed_b1})"
        }
        content = json.dumps(response_data, default=str) + "\n"
        return Response(content=content, media_type="application/json")
