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
# Use 4 workers for better connection handling under load
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4"]