#!/bin/bash

# ECM Distributed Server Deployment Script
# Usage: ./deploy.sh [production|staging]

set -e

ENVIRONMENT=${1:-production}
DOMAIN=${2:-your-domain.com}

echo "🚀 Deploying ECM Distributed Server to $ENVIRONMENT"

# Detect docker-compose command (plugin vs standalone)
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
elif docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    echo "❌ Error: Neither 'docker-compose' nor 'docker compose' found."
    exit 1
fi

echo "📦 Using: $DOCKER_COMPOSE"

# Check if running as root or with docker permissions
if ! docker ps >/dev/null 2>&1; then
    echo "❌ Error: Cannot access Docker. Run with sudo or add user to docker group."
    exit 1
fi

# Create secrets directory if it doesn't exist
mkdir -p secrets

# Generate secrets if they don't exist
if [ ! -f secrets/postgres_password.txt ]; then
    echo "🔐 Generating PostgreSQL password..."
    openssl rand -base64 32 > secrets/postgres_password.txt
    chmod 600 secrets/postgres_password.txt
fi

if [ ! -f secrets/api_secret_key.txt ]; then
    echo "🔐 Generating API secret key..."
    openssl rand -base64 64 > secrets/api_secret_key.txt
    chmod 600 secrets/api_secret_key.txt
fi

# Create SSL directory and generate self-signed cert if needed
mkdir -p ssl
if [ ! -f ssl/cert.pem ] || [ ! -f ssl/key.pem ]; then
    echo "🔐 Generating self-signed SSL certificate..."
    echo "⚠️  For production, replace with real certificates (Let's Encrypt recommended)"
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout ssl/key.pem -out ssl/cert.pem \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=$DOMAIN"
fi

# Update domain in nginx config (create temp file to avoid modifying source)
echo "🌐 Configuring domain: $DOMAIN"
sed "s/your-domain.com/$DOMAIN/g" nginx.conf > nginx.conf.prod
sed "s/your-domain.com/$DOMAIN/g" docker-compose.prod.yml > docker-compose.prod.tmp
mv docker-compose.prod.tmp docker-compose.prod.yml.active

# Pull latest images
echo "📦 Pulling latest Docker images..."
$DOCKER_COMPOSE -f docker-compose.prod.yml pull

# Stop existing services
echo "🛑 Stopping existing services..."
$DOCKER_COMPOSE -f docker-compose.prod.yml.active down 2>/dev/null || true

# Build and start services
echo "🏗️  Building and starting services..."
$DOCKER_COMPOSE -f docker-compose.prod.yml.active up -d --build

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🩺 Checking service health..."
if curl -f http://localhost:8000/health >/dev/null 2>&1; then
    echo "✅ API server is healthy"
else
    echo "❌ API server health check failed"
    $DOCKER_COMPOSE -f docker-compose.prod.yml.active logs api
    exit 1
fi

# Display status
echo "📊 Service status:"
$DOCKER_COMPOSE -f docker-compose.prod.yml.active ps

echo "🎉 Deployment complete!"
echo "📱 Dashboard: https://$DOMAIN/api/v1/dashboard/"
echo "📚 API Docs: https://$DOMAIN/docs"
echo "🔍 Health Check: https://$DOMAIN/health"

# Show logs
echo "📝 Recent logs:"
$DOCKER_COMPOSE -f docker-compose.prod.yml.active logs --tail=50