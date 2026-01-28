from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from .config import get_settings

settings = get_settings()

# Create SQLAlchemy engine with conservative pool settings for low-memory environments
# Default pool_size=5, max_overflow=10 is too aggressive for 1GB RAM droplet
engine = create_engine(
    settings.database_url,
    poolclass=QueuePool,
    pool_size=2,          # Keep only 2 persistent connections
    max_overflow=3,       # Allow up to 3 temporary connections under load
    pool_timeout=30,      # Wait up to 30s for a connection
    pool_recycle=1800,    # Recycle connections every 30 min to prevent stale connections
    pool_pre_ping=True,   # Verify connections are alive before using
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for models
Base = declarative_base()

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
