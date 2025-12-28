"""
Common database query patterns to reduce duplication across routes.
Centralizes frequently-used queries for composites, work assignments, and related entities.
"""
from datetime import datetime, timedelta
from typing import Optional, Any, List

from sqlalchemy import and_, desc, func, distinct
from sqlalchemy.orm import Session

from .calculations import CompositeCalculations


def get_recent_work_assignments(
    db: Session,
    limit: int = 20,
    status_filter: Optional[str] = None,
    client_id: Optional[str] = None
):
    """
    Get recent work assignments with optional filtering.

    Args:
        db: Database session
        limit: Maximum number of assignments to return
        status_filter: Filter by status (assigned, claimed, running, etc.)
        client_id: Filter by specific client

    Returns:
        List of WorkAssignment models
    """
    from ..models.work_assignments import WorkAssignment

    query = db.query(WorkAssignment)

    filters = []
    if status_filter:
        filters.append(WorkAssignment.status == status_filter)
    if client_id:
        filters.append(WorkAssignment.client_id == client_id)

    if filters:
        query = query.filter(and_(*filters))

    return query.order_by(desc(WorkAssignment.created_at)).limit(limit).all()


def get_active_work_assignments(db: Session, limit: int = 100):
    """
    Get currently active work assignments.

    Args:
        db: Database session
        limit: Maximum number to return

    Returns:
        List of active WorkAssignment models
    """
    from ..models.work_assignments import WorkAssignment

    return db.query(WorkAssignment).filter(
        WorkAssignment.status.in_(['assigned', 'claimed', 'running'])
    ).order_by(desc(WorkAssignment.created_at)).limit(limit).all()


def get_composites_by_completion(
    db: Session,
    limit: int = 50,
    include_factored: bool = False
):
    """
    Get composites sorted by ECM completion percentage.

    Args:
        db: Database session
        limit: Maximum number to return
        include_factored: If True, includes fully factored composites

    Returns:
        List of Composite models sorted by completion (high to low)
    """
    from ..models.composites import Composite

    query = db.query(Composite)

    filters: List[Any] = [Composite.target_t_level.isnot(None)]
    if not include_factored:
        filters.append(Composite.is_fully_factored == False)

    composites = query.filter(and_(*filters)).all()

    # Sort by completion percentage using centralized calculation
    composites = CompositeCalculations.sort_composites_by_progress(composites, reverse=True)
    return composites[:limit]


def get_recent_clients(db: Session, limit: int = 10, days: int = 7):
    """
    Get recently active clients with work statistics.

    Args:
        db: Database session
        limit: Maximum number of clients to return
        days: Look back this many days for activity

    Returns:
        List of tuples (client_id, work_count, last_seen)
    """
    from ..models.work_assignments import WorkAssignment

    since = datetime.utcnow() - timedelta(days=days)

    return db.query(
        WorkAssignment.client_id,
        func.count(WorkAssignment.id).label('work_count'),
        func.max(WorkAssignment.created_at).label('last_seen')
    ).filter(
        WorkAssignment.created_at >= since
    ).group_by(
        WorkAssignment.client_id
    ).order_by(
        desc('last_seen')
    ).limit(limit).all()


def get_recent_factors(db: Session, limit: int = 10):
    """
    Get recently discovered factors.

    Args:
        db: Database session
        limit: Maximum number to return

    Returns:
        List of Factor models with composite relationship loaded
    """
    from ..models.factors import Factor

    return db.query(Factor).order_by(desc(Factor.created_at)).limit(limit).all()


def get_recent_attempts(db: Session, limit: int = 100, method: Optional[str] = None):
    """
    Get recent ECM attempts with optional method filtering.

    Args:
        db: Database session
        limit: Maximum number to return
        method: Filter by method (ecm, pm1, pp1, etc.)

    Returns:
        List of ECMAttempt models
    """
    from ..models.attempts import ECMAttempt

    query = db.query(ECMAttempt)

    if method:
        query = query.filter(ECMAttempt.method == method)

    return query.order_by(desc(ECMAttempt.created_at)).limit(limit).all()


def get_aggregated_attempts(db: Session, limit: int = 50, method: Optional[str] = None, priority: Optional[int] = None):
    """
    Get recent ECM attempts aggregated by composite.

    For each composite with recent work, returns:
    - Composite information
    - Total curves run
    - Number of attempts (batches)
    - Date range
    - Factors found
    - List of individual attempts for expansion

    Args:
        db: Database session
        limit: Maximum number of composites to return
        method: Filter by method (ecm, pm1, pp1, etc.)
        priority: Filter by composite priority level

    Returns:
        List of dicts with aggregated attempt data per composite
    """
    from ..models.attempts import ECMAttempt
    from ..models.composites import Composite
    from ..models.factors import Factor

    # Get composites with recent attempts, ordered by most recent activity
    # Use GROUP BY with MAX to get the latest attempt time per composite
    query = db.query(
        ECMAttempt.composite_id,
        func.max(ECMAttempt.created_at).label('latest_attempt')
    )

    # Join with Composite table if filtering by priority
    if priority is not None:
        query = query.join(Composite).filter(Composite.priority == priority)

    query = query.group_by(ECMAttempt.composite_id)

    if method:
        query = query.filter(ECMAttempt.method == method)

    # Order by most recent activity
    composite_ids = query.order_by(desc('latest_attempt')).limit(limit).all()
    composite_ids = [cid[0] for cid in composite_ids]  # Extract just the composite_id

    aggregated = []
    for composite_id in composite_ids:
        composite = db.query(Composite).filter(Composite.id == composite_id).first()
        if not composite:
            continue

        # Get all attempts for this composite
        attempts_query = db.query(ECMAttempt).filter(
            ECMAttempt.composite_id == composite_id
        )
        if method:
            attempts_query = attempts_query.filter(ECMAttempt.method == method)

        attempts = attempts_query.order_by(desc(ECMAttempt.created_at)).all()

        if not attempts:
            continue

        # Aggregate statistics
        total_curves = sum(a.curves_completed or 0 for a in attempts)
        total_time = sum(a.execution_time_seconds or 0 for a in attempts)
        earliest = min(a.created_at for a in attempts if a.created_at)
        latest = max(a.created_at for a in attempts if a.created_at)

        # Get unique factors found across all attempts
        factors = db.query(Factor).filter(
            Factor.found_by_attempt_id.in_([a.id for a in attempts])
        ).all()

        aggregated.append({
            'composite': composite,
            'attempt_count': len(attempts),
            'total_curves': total_curves,
            'total_time': total_time,
            'earliest_attempt': earliest,
            'latest_attempt': latest,
            'factors': factors,
            'attempts': attempts  # Full list for expansion
        })

    return aggregated


def get_expired_work_assignments(db: Session):
    """
    Get work assignments that have expired but are still marked as active.

    Args:
        db: Database session

    Returns:
        List of expired WorkAssignment models
    """
    from ..models.work_assignments import WorkAssignment

    return db.query(WorkAssignment).filter(
        and_(
            WorkAssignment.expires_at < datetime.utcnow(),
            WorkAssignment.status.in_(['assigned', 'claimed', 'running'])
        )
    ).all()


def get_composite_with_details(db: Session, composite_id: int):
    """
    Get a composite with all related data (attempts, factors, work).

    Args:
        db: Database session
        composite_id: ID of the composite

    Returns:
        Tuple of (composite, attempts, factors, active_work) or None if not found
    """
    from ..models.composites import Composite
    from ..models.attempts import ECMAttempt
    from ..models.factors import Factor
    from ..models.work_assignments import WorkAssignment

    composite = db.query(Composite).filter(Composite.id == composite_id).first()
    if not composite:
        return None

    attempts = db.query(ECMAttempt).filter(
        ECMAttempt.composite_id == composite_id
    ).order_by(desc(ECMAttempt.created_at)).limit(50).all()

    factors = db.query(Factor).filter(
        Factor.composite_id == composite_id
    ).all()

    active_work = db.query(WorkAssignment).filter(
        and_(
            WorkAssignment.composite_id == composite_id,
            WorkAssignment.status.in_(['assigned', 'claimed', 'running'])
        )
    ).all()

    return composite, attempts, factors, active_work


def count_active_clients(db: Session, days: int = 7) -> int:
    """
    Count distinct clients active in the last N days.

    Args:
        db: Database session
        days: Look back this many days

    Returns:
        Count of unique active clients
    """
    from ..models.work_assignments import WorkAssignment

    since = datetime.utcnow() - timedelta(days=days)

    return db.query(distinct(WorkAssignment.client_id)).filter(
        and_(
            WorkAssignment.status.in_(['assigned', 'claimed', 'running']),
            WorkAssignment.created_at >= since
        )
    ).count()


def get_summary_statistics(db: Session) -> dict:
    """
    Get high-level summary statistics for the system.

    Args:
        db: Database session

    Returns:
        Dictionary with various counts and metrics
    """
    from ..models.composites import Composite
    from ..models.work_assignments import WorkAssignment
    from ..models.attempts import ECMAttempt
    from ..models.factors import Factor

    last_24h = datetime.utcnow() - timedelta(hours=24)

    return {
        "total_composites": db.query(Composite).count(),
        "factored_composites": db.query(Composite).filter(
            Composite.is_fully_factored
        ).count(),
        "active_work": db.query(WorkAssignment).filter(
            WorkAssignment.status.in_(['assigned', 'claimed', 'running'])
        ).count(),
        "recent_attempts_24h": db.query(ECMAttempt).filter(
            ECMAttempt.created_at >= last_24h
        ).count(),
        "recent_factors_24h": db.query(Factor).filter(
            Factor.created_at >= last_24h
        ).count(),
        "active_clients": count_active_clients(db, days=7)
    }


def get_inactive_composites(db: Session, limit: int = 100, offset: int = 0):
    """
    Get inactive composites that are not fully factored.

    Args:
        db: Database session
        limit: Maximum number to return
        offset: Pagination offset

    Returns:
        Tuple of (composites list, total count)
    """
    from ..models.composites import Composite

    query = db.query(Composite).filter(
        and_(
            Composite.is_active == False,
            Composite.is_fully_factored == False
        )
    )

    total = query.count()
    composites = query.order_by(
        desc(Composite.priority),
        desc(Composite.created_at)
    ).offset(offset).limit(limit).all()

    return composites, total


def get_outstanding_work_assignments(
    db: Session,
    limit: int = 100,
    offset: int = 0,
    client_id: Optional[str] = None,
    status_filter: Optional[str] = None,
    method_filter: Optional[str] = None
):
    """
    Get work assignments with status in ['assigned', 'claimed', 'running'].

    Args:
        db: Database session
        limit: Maximum number to return
        offset: Pagination offset
        client_id: Optional filter by client ID
        status_filter: Optional filter by status
        method_filter: Optional filter by method

    Returns:
        Tuple of (assignments list, total count)
    """
    from ..models.work_assignments import WorkAssignment

    query = db.query(WorkAssignment).filter(
        WorkAssignment.status.in_(['assigned', 'claimed', 'running'])
    )

    # Apply optional filters
    filters = []
    if client_id:
        filters.append(WorkAssignment.client_id == client_id)
    if status_filter:
        filters.append(WorkAssignment.status == status_filter)
    if method_filter:
        filters.append(WorkAssignment.method == method_filter)

    if filters:
        query = query.filter(and_(*filters))

    total = query.count()
    assignments = query.order_by(
        desc(WorkAssignment.created_at)
    ).offset(offset).limit(limit).all()

    return assignments, total


def get_recently_added_composites(
    db: Session,
    days: int = 30,
    limit: int = 100,
    offset: int = 0
):
    """
    Get composites created in the last N days.

    Args:
        db: Database session
        days: Number of days to look back
        limit: Maximum number to return
        offset: Pagination offset

    Returns:
        Tuple of (composites list, total count)
    """
    from ..models.composites import Composite

    since = datetime.utcnow() - timedelta(days=days)

    query = db.query(Composite).filter(
        Composite.created_at >= since
    )

    total = query.count()
    composites = query.order_by(
        desc(Composite.created_at)
    ).offset(offset).limit(limit).all()

    return composites, total


def get_residues_filtered(
    db: Session,
    limit: int = 100,
    offset: int = 0,
    status_filter: Optional[str] = None,
    client_id: Optional[str] = None,
    composite_id: Optional[int] = None,
    expiring_soon: bool = False
):
    """
    Get residues with comprehensive filtering.

    Args:
        db: Database session
        limit: Maximum number to return
        offset: Pagination offset
        status_filter: Filter by status (available/claimed/completed/expired)
        client_id: Filter by client that uploaded
        composite_id: Filter by specific composite
        expiring_soon: If True, show only residues expiring in < 24 hours

    Returns:
        Tuple of (residues list, total count)
    """
    from ..models.residues import ECMResidue

    query = db.query(ECMResidue)

    # Apply filters
    filters = []
    if status_filter:
        filters.append(ECMResidue.status == status_filter)
    if client_id:
        filters.append(ECMResidue.client_id == client_id)
    if composite_id:
        filters.append(ECMResidue.composite_id == composite_id)
    if expiring_soon:
        # Only claimed residues have expires_at set (claim timeout)
        # Available residues don't expire by time
        expiry_threshold = datetime.utcnow() + timedelta(hours=24)
        filters.append(ECMResidue.expires_at < expiry_threshold)
        filters.append(ECMResidue.status == 'claimed')

    if filters:
        query = query.filter(and_(*filters))

    total = query.count()
    residues = query.order_by(
        desc(ECMResidue.created_at)
    ).offset(offset).limit(limit).all()

    return residues, total
