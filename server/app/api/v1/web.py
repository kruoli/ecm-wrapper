from fastapi import APIRouter, Depends, HTTPException, Request, status, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, case, distinct
from typing import List, Optional
from datetime import datetime, timedelta

from ...database import get_db
from ...dependencies import get_composite_service
from ...models import Composite, ECMAttempt, Factor
from ...services.composites import CompositeService
from ...templates import templates
from ...utils.query_helpers import get_aggregated_attempts

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    db: Session = Depends(get_db),
    priority: Optional[int] = Query(None, ge=1, description="Filter by priority level")
):
    """
    Simple dashboard showing all factorization results.
    Optionally filter by priority level.
    """

    # Build query for composites with optional priority filter
    composites_query = db.query(Composite)
    if priority is not None:
        composites_query = composites_query.filter(Composite.priority == priority)
    composites = composites_query.order_by(desc(Composite.created_at)).limit(50).all()

    # Get recent attempts (aggregated by composite), filtered by priority
    attempts = get_aggregated_attempts(db, limit=50, priority=priority)

    # Get recent factors (limited for main page)
    factors = db.query(Factor).order_by(desc(Factor.created_at)).limit(25).all()

    # Build summary stats (filtered by priority if specified)
    stats_query = db.query(Composite)
    if priority is not None:
        stats_query = stats_query.filter(Composite.priority == priority)

    total_composites = stats_query.count()
    fully_factored = stats_query.filter(Composite.is_fully_factored == True).count()

    # Total attempts - need to join through composites if filtering by priority
    attempts_query = db.query(ECMAttempt).filter(ECMAttempt.superseded_by.is_(None))
    if priority is not None:
        attempts_query = attempts_query.join(Composite).filter(Composite.priority == priority)
    total_attempts = attempts_query.count()

    # Total factors count (all priorities)
    total_factors = db.query(Factor).count()

    return templates.TemplateResponse("public/dashboard.html", {
        "request": request,
        "composites": composites,
        "attempts": attempts,
        "factors": factors,
        "total_composites": total_composites,
        "total_attempts": total_attempts,
        "total_factors": total_factors,
        "fully_factored": fully_factored,
        "priority": priority,
        "db": db,
        "Composite": Composite,
        "ECMAttempt": ECMAttempt,
        "Factor": Factor
    })


@router.get("/testing-status", response_class=HTMLResponse)
async def testing_status(
    request: Request,
    db: Session = Depends(get_db),
    composite_service: CompositeService = Depends(get_composite_service),
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
        # current_t_level now includes prior_t_level
        current_t = comp.current_t_level
        if current_t == 0:
            status = "not_started"
            status_label = "Not Started"
            status_color = "red"
        elif current_t < 30:
            status = "initial"
            status_label = "Initial"
            status_color = "yellow"
        elif current_t < 40:
            status = "standard"
            status_label = "Standard"
            status_color = "orange"
        elif current_t < 50:
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

        # Calculate progress percentage
        target_t = comp.target_t_level if comp.target_t_level is not None else 0
        progress_pct = (current_t / target_t * 100) if target_t > 0 else 0

        composite_data.append({
            'id': comp.id,
            'digit_length': comp.digit_length,
            'priority': comp.priority,
            'current_t_level': current_t,
            'prior_t_level': comp.prior_t_level,
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


@router.get("/curves", response_class=HTMLResponse)
async def recent_curves(
    request: Request,
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=200, description="Results per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    client_id: Optional[str] = Query(None, description="Filter by client ID"),
    group_by_composite: bool = Query(True, description="Group curves by composite")
):
    """
    Public page showing recent ECM curves with filtering and pagination.
    """
    if group_by_composite:
        # Get aggregated attempts - fetch all for proper pagination
        all_aggregated = get_aggregated_attempts(db, limit=500)  # Higher limit for aggregation

        # Filter by client if specified (filter the individual attempts)
        if client_id:
            filtered_attempts = []
            for agg in all_aggregated:
                client_attempts = [a for a in agg['attempts'] if a.client_id == client_id]
                if client_attempts:
                    # Recalculate aggregates for filtered attempts
                    total_curves = sum(a.curves_completed or 0 for a in client_attempts)
                    total_time = sum(a.execution_time_seconds or 0 for a in client_attempts)
                    filtered_attempts.append({
                        **agg,
                        'attempts': client_attempts,
                        'attempt_count': len(client_attempts),
                        'total_curves': total_curves,
                        'total_time': total_time,
                    })
            all_aggregated = filtered_attempts

        # Apply pagination to aggregated results
        total = len(all_aggregated)
        attempts = all_aggregated[offset:offset + limit]
    else:
        # Get individual attempts
        query = db.query(ECMAttempt).filter(ECMAttempt.superseded_by.is_(None))

        if client_id:
            query = query.filter(ECMAttempt.client_id == client_id)

        total = query.count()
        attempts = query.order_by(desc(ECMAttempt.created_at)).offset(offset).limit(limit).all()

    # Get list of active clients for filter dropdown
    clients_rows = db.query(distinct(ECMAttempt.client_id)).filter(
        ECMAttempt.client_id.isnot(None)
    ).order_by(ECMAttempt.client_id).limit(100).all()
    clients: List[str] = [c[0] for c in clients_rows if c[0]]

    # Pagination metadata
    page = (offset // limit) + 1 if limit > 0 else 1
    total_pages = (total + limit - 1) // limit if limit > 0 else 1

    return templates.TemplateResponse("public/recent_curves.html", {
        "request": request,
        "attempts": attempts,
        "group_by_composite": group_by_composite,
        "total": total,
        "page": page,
        "total_pages": total_pages,
        "limit": limit,
        "offset": offset,
        "has_prev": offset > 0,
        "has_next": offset + limit < total,
        "client_id": client_id,
        "clients": clients,
        "db": db,
        "Composite": Composite,
        "Factor": Factor
    })


@router.get("/factors", response_class=HTMLResponse)
async def recent_factors(
    request: Request,
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=200, description="Results per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    client_id: Optional[str] = Query(None, description="Filter by client who found factor"),
    min_digits: Optional[int] = Query(None, ge=1, description="Minimum factor digits"),
    max_digits: Optional[int] = Query(None, ge=1, description="Maximum factor digits")
):
    """
    Public page showing discovered factors with filtering and pagination.
    """
    # Build query
    query = db.query(Factor)

    # Filter by client who found the factor (via the attempt)
    if client_id:
        query = query.join(ECMAttempt, Factor.found_by_attempt_id == ECMAttempt.id).filter(
            ECMAttempt.client_id == client_id
        )

    # Apply digit filters using length of factor string
    if min_digits:
        query = query.filter(func.length(Factor.factor) >= min_digits)
    if max_digits:
        query = query.filter(func.length(Factor.factor) <= max_digits)

    total = query.count()
    factors = query.order_by(desc(Factor.created_at)).offset(offset).limit(limit).all()

    # Get list of clients who have found factors for filter dropdown
    clients_rows = db.query(distinct(ECMAttempt.client_id)).join(
        Factor, Factor.found_by_attempt_id == ECMAttempt.id
    ).filter(ECMAttempt.client_id.isnot(None)).order_by(ECMAttempt.client_id).limit(100).all()
    clients: List[str] = [c[0] for c in clients_rows if c[0]]

    # Pagination metadata
    page = (offset // limit) + 1 if limit > 0 else 1
    total_pages = (total + limit - 1) // limit if limit > 0 else 1

    return templates.TemplateResponse("public/recent_factors.html", {
        "request": request,
        "factors": factors,
        "total": total,
        "page": page,
        "total_pages": total_pages,
        "limit": limit,
        "offset": offset,
        "has_prev": offset > 0,
        "has_next": offset + limit < total,
        "client_id": client_id,
        "min_digits": min_digits,
        "max_digits": max_digits,
        "clients": clients,
        "db": db,
        "Composite": Composite,
        "ECMAttempt": ECMAttempt
    })


@router.get("/leaderboard", response_class=HTMLResponse)
async def leaderboard(
    request: Request,
    db: Session = Depends(get_db),
    days: int = Query(30, ge=1, le=365, description="Days to look back")
):
    """
    Public leaderboard showing top contributors by curves and factors.
    """
    since = datetime.utcnow() - timedelta(days=days)

    # Top contributors by curves
    top_by_curves = db.query(
        ECMAttempt.client_id,
        func.sum(ECMAttempt.curves_completed).label('total_curves'),
        func.count(ECMAttempt.id).label('attempt_count'),
        func.max(ECMAttempt.created_at).label('last_active')
    ).filter(
        ECMAttempt.created_at >= since,
        ECMAttempt.client_id.isnot(None),
        ECMAttempt.superseded_by.is_(None)
    ).group_by(ECMAttempt.client_id).order_by(
        desc('total_curves')
    ).limit(25).all()

    # Top contributors by factors found
    top_by_factors = db.query(
        ECMAttempt.client_id,
        func.count(Factor.id).label('factor_count'),
        func.max(Factor.created_at).label('last_factor')
    ).join(
        Factor, Factor.found_by_attempt_id == ECMAttempt.id
    ).filter(
        Factor.created_at >= since,
        ECMAttempt.client_id.isnot(None)
    ).group_by(ECMAttempt.client_id).order_by(
        desc('factor_count')
    ).limit(25).all()

    # Recent activity stats
    total_curves_period = db.query(func.sum(ECMAttempt.curves_completed)).filter(
        ECMAttempt.created_at >= since,
        ECMAttempt.superseded_by.is_(None)
    ).scalar() or 0

    total_factors_period = db.query(func.count(Factor.id)).filter(
        Factor.created_at >= since
    ).scalar() or 0

    active_clients = db.query(func.count(distinct(ECMAttempt.client_id))).filter(
        ECMAttempt.created_at >= since,
        ECMAttempt.superseded_by.is_(None)
    ).scalar() or 0

    # Activity by day (last 14 days for chart) - 2 queries instead of 28
    lookback_days = min(14, days)
    chart_since = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=lookback_days - 1)

    # Get curves by day in single query
    curves_by_day = db.query(
        func.date(ECMAttempt.created_at).label('day'),
        func.sum(ECMAttempt.curves_completed).label('curves')
    ).filter(
        ECMAttempt.created_at >= chart_since,
        ECMAttempt.superseded_by.is_(None)
    ).group_by(func.date(ECMAttempt.created_at)).all()

    # Get factors by day in single query
    factors_by_day = db.query(
        func.date(Factor.created_at).label('day'),
        func.count(Factor.id).label('factors')
    ).filter(
        Factor.created_at >= chart_since
    ).group_by(func.date(Factor.created_at)).all()

    # Build lookup dicts
    curves_lookup = {str(row.day): int(row.curves or 0) for row in curves_by_day}
    factors_lookup = {str(row.day): int(row.factors or 0) for row in factors_by_day}

    # Build daily activity list (oldest first)
    daily_activity = []
    for i in range(lookback_days - 1, -1, -1):
        day = (datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=i)).strftime('%Y-%m-%d')
        daily_activity.append({
            'date': day,
            'curves': curves_lookup.get(day, 0),
            'factors': factors_lookup.get(day, 0)
        })

    return templates.TemplateResponse("public/leaderboard.html", {
        "request": request,
        "top_by_curves": top_by_curves,
        "top_by_factors": top_by_factors,
        "total_curves_period": total_curves_period,
        "total_factors_period": total_factors_period,
        "active_clients": active_clients,
        "daily_activity": daily_activity,
        "days": days
    })