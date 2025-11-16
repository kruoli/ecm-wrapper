from fastapi import APIRouter, Depends, HTTPException, Request, status, Query
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, case
from typing import List, Optional

from ...database import get_db
from ...dependencies import get_composite_service
from ...models import Composite, ECMAttempt, Factor
from ...services.composites import CompositeService
from ...templates import templates
from ...utils.query_helpers import get_aggregated_attempts

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    """
    Simple dashboard showing all factorization results.
    """

    # Get recent composites with their attempts and factors
    composites = db.query(Composite).order_by(desc(Composite.created_at)).limit(50).all()

    # Get recent attempts (aggregated by composite)
    attempts = get_aggregated_attempts(db, limit=50)

    # Get all factors
    factors = db.query(Factor).order_by(desc(Factor.created_at)).all()

    # Build summary stats
    total_composites = db.query(Composite).count()
    total_attempts = db.query(ECMAttempt).count()
    total_factors = db.query(Factor).count()
    fully_factored = db.query(Composite).filter(Composite.is_fully_factored == True).count()

    return templates.TemplateResponse("public/dashboard.html", {
        "request": request,
        "composites": composites,
        "attempts": attempts,
        "factors": factors,
        "total_composites": total_composites,
        "total_attempts": total_attempts,
        "total_factors": total_factors,
        "fully_factored": fully_factored,
        "db": db,
        "Composite": Composite,
        "ECMAttempt": ECMAttempt,
        "Factor": Factor
    })


@router.get("/testing-status", response_class=HTMLResponse)
async def testing_status(
    request: Request,
    db: Session = Depends(get_db),
    limit: int = Query(200, ge=1, le=200, description="Composites per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    min_t_level: Optional[float] = Query(None, ge=0, description="Minimum current t-level"),
    max_t_level: Optional[float] = Query(None, ge=0, description="Maximum current t-level"),
    min_priority: Optional[int] = Query(None, ge=0, description="Minimum priority"),
    snfs_difficulty: Optional[int] = Query(None, ge=0, description="SNFS difficulty filter")
):
    """
    Public testing status dashboard showing t-level progress and method coverage.

    Now supports pagination and filtering for efficient handling of large datasets.
    """
    # Get milestone groups (not paginated, shows overall progress)
    from ...services.composites import CompositeService
    composite_service = CompositeService()
    milestone_groups = composite_service.get_milestone_groups(db)

    # Build query with filters
    query = db.query(Composite).filter(Composite.is_fully_factored == False)

    # Apply filters
    if min_t_level is not None:
        query = query.filter(Composite.current_t_level >= min_t_level)
    if max_t_level is not None:
        query = query.filter(Composite.current_t_level <= max_t_level)
    if min_priority is not None:
        query = query.filter(Composite.priority >= min_priority)
    if snfs_difficulty is not None:
        query = query.filter(Composite.snfs_difficulty == snfs_difficulty)

    # Count total before pagination
    total = query.count()

    # Apply ordering and pagination
    query = query.order_by(Composite.priority.desc(), Composite.digit_length.asc())
    composites = query.offset(offset).limit(limit).all()

    # Get method counts for all composites in this page using aggregation (fixes N+1 query problem)
    composite_ids = [c.id for c in composites]

    method_counts = {}
    if composite_ids:
        # Aggregate method counts in a single query
        counts_query = db.query(
            ECMAttempt.composite_id,
            ECMAttempt.method,
            func.count(ECMAttempt.id).label('attempt_count'),
            func.sum(
                case(
                    (ECMAttempt.method == 'ecm', ECMAttempt.curves_completed),
                    else_=0
                )
            ).label('ecm_curves')
        ).filter(
            ECMAttempt.composite_id.in_(composite_ids)
        ).group_by(
            ECMAttempt.composite_id,
            ECMAttempt.method
        ).all()

        # Build method counts dictionary
        for composite_id, method, attempt_count, ecm_curves in counts_query:
            if composite_id not in method_counts:
                method_counts[composite_id] = {
                    'ecm_curves': 0,
                    'pm1_count': 0,
                    'pp1_count': 0
                }

            if method == 'ecm':
                method_counts[composite_id]['ecm_curves'] = int(ecm_curves or 0)
            elif method == 'pm1':
                method_counts[composite_id]['pm1_count'] = attempt_count
            elif method == 'pp1':
                method_counts[composite_id]['pp1_count'] = attempt_count

    # Calculate status for each composite
    composite_data = []
    for comp in composites:
        # Determine status category
        t = comp.current_t_level if comp.current_t_level is not None else 0
        if t == 0:
            status = "not_started"
            status_label = "Not Started"
            status_color = "red"
        elif t < 30:
            status = "initial"
            status_label = "Initial"
            status_color = "yellow"
        elif t < 40:
            status = "standard"
            status_label = "Standard"
            status_color = "orange"
        elif t < 50:
            status = "advanced"
            status_label = "Advanced"
            status_color = "blue"
        else:
            status = "deep"
            status_label = "Deep"
            status_color = "green"

        # Get method counts from aggregated results
        counts = method_counts.get(comp.id, {'ecm_curves': 0, 'pm1_count': 0, 'pp1_count': 0})
        ecm_curves = counts['ecm_curves']
        pm1_count = counts['pm1_count']
        pp1_count = counts['pp1_count']

        # Calculate progress percentage safely
        current_t = comp.current_t_level if comp.current_t_level is not None else 0
        target_t = comp.target_t_level if comp.target_t_level is not None else 0
        progress_pct = (current_t / target_t * 100) if target_t > 0 else 0

        composite_data.append({
            'id': comp.id,
            'digit_length': comp.digit_length,
            'priority': comp.priority,
            'current_t_level': comp.current_t_level,
            'target_t_level': comp.target_t_level,
            'progress_pct': progress_pct,
            'status': status,
            'status_label': status_label,
            'status_color': status_color,
            'ecm_curves': ecm_curves,
            'pm1_count': pm1_count,
            'pp1_count': pp1_count,
            'methods_used': sum([ecm_curves > 0, pm1_count > 0, pp1_count > 0])
        })

    # Calculate pagination metadata
    page = (offset // limit) + 1 if limit > 0 else 1
    total_pages = (total + limit - 1) // limit if limit > 0 else 1
    showing_from = offset + 1 if total > 0 else 0
    showing_to = min(offset + limit, total)

    return templates.TemplateResponse("public/testing_status.html", {
        "request": request,
        "milestone_groups": milestone_groups,
        "composites": composite_data,
        # Pagination metadata
        "total": total,
        "page": page,
        "total_pages": total_pages,
        "limit": limit,
        "offset": offset,
        "showing_from": showing_from,
        "showing_to": showing_to,
        "has_prev": offset > 0,
        "has_next": offset + limit < total,
        # Filter values (for form state)
        "min_t_level": min_t_level,
        "max_t_level": max_t_level,
        "min_priority": min_priority,
        "snfs_difficulty": snfs_difficulty
    })


@router.get("/composites/find")
async def find_composite_public(
    q: str,
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service)
):
    """Find composite by ID, number (formula), or current_composite value.

    Args:
        q: Search query - can be composite ID, number (e.g., "2^1223-1"),
           or current_composite value

    Returns:
        Redirect to the composite's details page
    """
    from fastapi.responses import RedirectResponse

    composite = composite_service.find_composite_by_identifier(db, q)
    if not composite:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Composite not found: {q}"
        )

    # Redirect to the canonical details page URL
    return RedirectResponse(
        url=f"/api/v1/dashboard/composites/{composite.id}/details",
        status_code=status.HTTP_302_FOUND
    )


@router.get("/composites/{composite_id}/details", response_class=HTMLResponse)
async def get_composite_details_public(
    composite_id: int,
    request: Request,
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service)
):
    """
    Public web page showing detailed information about a specific composite.
    """
    details = composite_service.get_composite_details(db, composite_id)

    if not details:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Composite Not Found</title>
        </head>
        <body>
            <h1>Composite Not Found</h1>
            <p><a href="/api/v1/dashboard/">Back to Dashboard</a></p>
        </body>
        </html>
        """

    # Get method breakdown for tabbed interface
    from ...services.composites import CompositeService
    composite_service = CompositeService()
    method_breakdown = composite_service.get_method_breakdown(composite_id, db)

    return templates.TemplateResponse("public/composite_details.html", {
        "request": request,
        "composite": details['composite'],
        "progress": details['progress'],
        "recent_attempts": details['recent_attempts'],
        "active_work": details['active_work'],
        "factors_with_group_orders": details['factors_with_group_orders'],
        "method_breakdown": method_breakdown,
        "db": db,
        "Factor": Factor
    })