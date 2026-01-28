FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including PARI/GP
RUN apt-get update && apt-get install -y \
    gcc \
    pari-gp \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
# requirements.txt should be in the build context root (server/)
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/
COPY alembic.ini .
COPY migrations/ migrations/

# Create bin directory and copy binaries and scripts
RUN mkdir -p /app/bin
COPY bin/t-level /app/bin/t-level
COPY bin/group.gp /app/bin/group.gp
RUN chmod +x /app/bin/t-level

# Run database migrations on startup
# Memory-optimized settings for 1GB RAM droplet:
# - Single worker (multiple workers caused OOM/swapping)
# - limit-concurrency: Max simultaneous requests to prevent memory exhaustion
# - limit-max-requests: Restart worker after N requests to prevent memory leaks
# - timeout-keep-alive: Shorter keepalive to free connections faster
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --limit-concurrency 20 --limit-max-requests 5000 --timeout-keep-alive 5"]