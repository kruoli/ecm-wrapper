"""Rate limiting utilities for use behind Cloudflare/reverse proxies."""

from fastapi import Request
from slowapi.util import get_remote_address

from .config import get_settings


def get_real_client_ip(request: Request) -> str:
    """Extract real client IP, honoring proxy headers only from trusted proxies.

    If the immediate caller (request.client.host) isn't in trusted_proxies,
    CF-Connecting-IP and X-Forwarded-For are ignored — otherwise an attacker
    hitting uvicorn directly could spoof their rate-limit identity.
    """
    trusted = get_settings().trusted_proxies_set
    immediate = request.client.host if request.client else None

    if immediate in trusted:
        # Cloudflare sets CF-Connecting-IP to the real client
        cf_ip = request.headers.get("CF-Connecting-IP")
        if cf_ip:
            return cf_ip
        # Standard proxy header — leftmost entry is the original client
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

    return get_remote_address(request)
