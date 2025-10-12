from fastapi import Header, HTTPException, status
from .config import get_settings

settings = get_settings()

async def verify_admin_key(x_admin_key: str = Header(None)):
    """
    Dependency to verify admin API key from header.

    Requires X-Admin-Key header to match ADMIN_API_KEY environment variable.

    Raises:
        HTTPException: 401 if key missing or invalid
    """
    if not x_admin_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin API key required. Provide X-Admin-Key header."
        )

    if x_admin_key != settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin API key"
        )

    return True
