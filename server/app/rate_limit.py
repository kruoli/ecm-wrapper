"""Rate limiting utilities for use behind Cloudflare/reverse proxies."""

from fastapi import Request
from slowapi.util import get_remote_address


def get_real_client_ip(request: Request) -> str:
    """Extract real client IP from proxy headers, falling back to remote address."""
    # Cloudflare sets CF-Connecting-IP to the real client
    cf_ip = request.headers.get("CF-Connecting-IP")
    if cf_ip:
        return cf_ip
    # Standard proxy header
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)
