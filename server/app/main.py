import asyncio
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware

from .config import get_settings
from .database import engine
from .models.base import Base
from .api.v1.router import v1_router


class TimeoutMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add request timeouts and prevent slow requests from accumulating.

    For memory-constrained environments, this prevents a few slow requests
    from tying up all available connections/memory.

    Admin endpoints get a longer timeout since some operations (like t-level
    recalculation) can take longer.
    """

    def __init__(self, app, timeout_seconds: int = 60, admin_timeout_seconds: int = 300):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds
        self.admin_timeout_seconds = admin_timeout_seconds

    async def dispatch(self, request: Request, call_next):
        # Admin endpoints get longer timeout (5 min vs 60s)
        path = request.url.path
        if "/admin/" in path:
            timeout = self.admin_timeout_seconds
        else:
            timeout = self.timeout_seconds

        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logging.warning(f"Request timeout ({timeout}s): {request.method} {path}")
            return JSONResponse(
                status_code=504,
                content={"detail": "Request timeout - please try again"}
            )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create database tables
Base.metadata.create_all(bind=engine)

# Get settings
settings = get_settings()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware for web client access
# Note: allow_credentials=False because we accept requests from any origin
# and don't use cookie-based authentication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin (public API)
    allow_credentials=False,  # No cookie-based auth used
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add timeout middleware to prevent slow requests from tying up resources
# 60 second timeout is generous but prevents runaway requests
app.add_middleware(TimeoutMiddleware, timeout_seconds=60)

# Include API routers
app.include_router(v1_router, prefix="/api")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ecm-distributed-api"}

@app.get("/")
async def root():
    return {
        "service": "ECM Distributed Factorization API",
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/favicon.ico")
async def favicon():
    """Return a simple SVG favicon"""
    # Simple "E" icon for ECM
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <rect width="100" height="100" fill="#667eea"/>
        <text x="50" y="75" font-family="Arial" font-size="70" font-weight="bold" fill="white" text-anchor="middle">E</text>
    </svg>"""
    return Response(content=svg, media_type="image/svg+xml")
