"""
Specialized admin dashboard routes for composite and work management.
"""
import secrets
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from ....config import get_settings
from ....database import get_db
from ....templates import templates
from ....utils.html_helpers import get_unauthorized_redirect_html
from ....utils.query_helpers import (
    get_inactive_composites,
    get_outstanding_work_assignments,
    get_recently_added_composites,
    get_residues_filtered,
    calculate_pagination
)

router = APIRouter()


@router.get("/inactive-composites", response_class=HTMLResponse)
async def inactive_composites_dashboard(
    request: Request,
    db: Session = Depends(get_db),
    x_admin_key: str = Header(None),
    limit: int = Query(100, ge=1, le=200, description="Items per page"),
    offset: int = Query(0, ge=0, description="Pagination offset")
):
    """
    Dashboard showing inactive composites that are not fully factored.
    Allows bulk activation.
    """
    settings = get_settings()
    # Security: Constant-time comparison to prevent timing attacks
    if not x_admin_key or not secrets.compare_digest(x_admin_key, settings.admin_api_key):
        return get_unauthorized_redirect_html()

    composites, total = get_inactive_composites(db, limit=limit, offset=offset)
    pagination = calculate_pagination(offset, limit, total)

    return templates.TemplateResponse("admin/inactive_composites.html", {
        "request": request,
        "composites": composites,
        **pagination.to_dict()
    })


@router.get("/outstanding-work", response_class=HTMLResponse)
async def outstanding_work_dashboard(
    request: Request,
    db: Session = Depends(get_db),
    x_admin_key: str = Header(None),
    limit: int = Query(100, ge=1, le=200, description="Items per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    client_id: Optional[str] = Query(None, description="Filter by client ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    method: Optional[str] = Query(None, description="Filter by method")
):
    """
    Dashboard showing outstanding work assignments with filtering.
    Allows expire and cancel actions.
    """
    settings = get_settings()
    # Security: Constant-time comparison to prevent timing attacks
    if not x_admin_key or not secrets.compare_digest(x_admin_key, settings.admin_api_key):
        return get_unauthorized_redirect_html()

    assignments, total = get_outstanding_work_assignments(
        db,
        limit=limit,
        offset=offset,
        client_id=client_id,
        status_filter=status,
        method_filter=method
    )
    pagination = calculate_pagination(offset, limit, total)

    return templates.TemplateResponse("admin/outstanding_work.html", {
        "request": request,
        "assignments": assignments,
        **pagination.to_dict(),
        # Filter values for form state
        "filter_client_id": client_id,
        "filter_status": status,
        "filter_method": method,
        "now": datetime.utcnow()
    })


@router.get("/recent-composites", response_class=HTMLResponse)
async def recent_composites_dashboard(
    request: Request,
    db: Session = Depends(get_db),
    x_admin_key: str = Header(None),
    limit: int = Query(100, ge=1, le=200, description="Items per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    days: int = Query(30, ge=1, le=90, description="Days to look back")
):
    """
    Dashboard showing recently added composites.
    Allows activate/deactivate, priority changes, and deletion.
    """
    settings = get_settings()
    # Security: Constant-time comparison to prevent timing attacks
    if not x_admin_key or not secrets.compare_digest(x_admin_key, settings.admin_api_key):
        return get_unauthorized_redirect_html()

    composites, total = get_recently_added_composites(db, days=days, limit=limit, offset=offset)
    pagination = calculate_pagination(offset, limit, total)

    return templates.TemplateResponse("admin/recent_composites.html", {
        "request": request,
        "composites": composites,
        **pagination.to_dict(),
        "days": days
    })


@router.get("/residue-status", response_class=HTMLResponse)
async def residue_status_dashboard(
    request: Request,
    db: Session = Depends(get_db),
    x_admin_key: str = Header(None),
    limit: int = Query(100, ge=1, le=200, description="Items per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    status: Optional[str] = Query(None, description="Filter by status"),
    client_id: Optional[str] = Query(None, description="Filter by client ID"),
    composite_id: Optional[int] = Query(None, description="Filter by composite ID"),
    expiring_soon: bool = Query(False, description="Show only expiring soon")
):
    """
    Dashboard showing residue status with comprehensive filtering.
    Allows release, download, delete actions and cleanup of expired residues.
    """
    settings = get_settings()
    # Security: Constant-time comparison to prevent timing attacks
    if not x_admin_key or not secrets.compare_digest(x_admin_key, settings.admin_api_key):
        return get_unauthorized_redirect_html()

    from ....services.residue_manager import ResidueManager
    residue_manager = ResidueManager()

    # Get filtered residues
    residues, total = get_residues_filtered(
        db,
        limit=limit,
        offset=offset,
        status_filter=status,
        client_id=client_id,
        composite_id=composite_id,
        expiring_soon=expiring_soon
    )

    # Get summary statistics
    stats = residue_manager.get_stats(db)
    pagination = calculate_pagination(offset, limit, total)

    return templates.TemplateResponse("admin/residue_status.html", {
        "request": request,
        "residues": residues,
        **pagination.to_dict(),
        # Filter values for form state
        "filter_status": status,
        "filter_client_id": client_id,
        "filter_composite_id": composite_id,
        "filter_expiring_soon": expiring_soon,
        # Statistics
        "stats": stats,
        "now": datetime.utcnow()
    })
