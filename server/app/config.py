import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings

def read_secret_file(file_path: str) -> Optional[str]:
    """Read secret from file if it exists."""
    try:
        path = Path(file_path)
        if path.exists():
            return path.read_text(encoding='utf-8').strip()
    except Exception:
        pass
    return None

def get_database_url() -> str:
    """Construct database URL from environment or secret files."""
    # Check if full DATABASE_URL is provided
    if "DATABASE_URL" in os.environ:
        return os.getenv("DATABASE_URL")

    # Build from components (for Docker secrets)
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "ecm_distributed")
    user = os.getenv("POSTGRES_USER", "ecm_user")

    # Try to read password from secret file, fallback to env var
    password_file = os.getenv("POSTGRES_PASSWORD_FILE")
    password = read_secret_file(password_file) if password_file else os.getenv("POSTGRES_PASSWORD", "ecm_password")

    return f"postgresql://{user}:{password}@{host}:{port}/{db}"

class Settings(BaseSettings):
    # Database
    database_url: str = Field(
        default_factory=get_database_url,
        description="PostgreSQL connection string"
    )

    # API
    api_title: str = "ECM Distributed Factorization API"
    api_version: str = "1.0.0"
    api_description: str = "API for coordinating distributed integer factorization"

    # Server
    host: str = Field(default="0.0.0.0", description="Server bind address")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    reload: bool = False

    # Work assignment
    default_work_timeout_minutes: int = Field(default=60, ge=1, le=1440, description="Work timeout in minutes")
    max_work_items_per_client: int = Field(default=5, ge=1, le=100, description="Max work items per client")

    # T-level calculation
    t_level_binary_path: str = Field(
        default=os.getenv("T_LEVEL_BINARY_PATH", "/app/bin/t-level"),
        description="Path to t-level executable binary"
    )

    # Security
    secret_key: str = Field(
        default_factory=lambda: (
            read_secret_file(os.getenv("SECRET_KEY_FILE")) if os.getenv("SECRET_KEY_FILE")
            else os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
        ),
        min_length=16,
        description="Secret key for cryptographic operations"
    )

    admin_api_key: str = Field(
        default_factory=lambda: (
            read_secret_file(os.getenv("ADMIN_API_KEY_FILE")) if os.getenv("ADMIN_API_KEY_FILE")
            else os.getenv("ADMIN_API_KEY", "dev-admin-key-change-in-production")
        ),
        min_length=16,
        description="API key for admin endpoints"
    )

    @validator("database_url")
    def validate_database_url(cls, v):
        if not v.startswith("postgresql://") and not v.startswith("postgresql+psycopg2://"):
            raise ValueError("database_url must be a PostgreSQL connection string")
        return v

    @validator("secret_key")
    def validate_secret_key(cls, v):
        if v == "dev-secret-key-change-in-production":
            import warnings
            warnings.warn("Using default secret key - change for production!", UserWarning)
        return v

    @validator("admin_api_key")
    def validate_admin_api_key(cls, v):
        if v == "dev-admin-key-change-in-production":
            import warnings
            warnings.warn("Using default admin API key - change for production!", UserWarning)
        return v

    class Config:
        env_file = ".env"
        validate_assignment = True

@lru_cache()
def get_settings():
    return Settings()
