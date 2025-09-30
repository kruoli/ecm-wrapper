# ECM Coordination Middleware

A lightweight, distributed ECM factorization coordination system with client wrappers for GMP-ECM and YAFU.

## Quick Start

### Run Server with Docker (5 minutes)

```bash
cd server/
cp .env.example .env

# Generate secure keys
echo "DB_PASSWORD=$(openssl rand -hex 32)" >> .env
echo "SECRET_KEY=$(openssl rand -hex 32)" >> .env
echo "ADMIN_API_KEY=$(openssl rand -hex 32)" >> .env

# Start services
docker-compose -f docker-compose.simple.yml up -d

# Access admin dashboard
# Open http://localhost:8000/api/v1/admin/login
# Enter your ADMIN_API_KEY from .env
```

### Run Client

```bash
cd client/

# Install dependencies
pip install requests pyyaml

# Configure client.yaml with your API endpoint and binary paths

# Run ECM factorization
python3 ecm-wrapper.py --composite "123456789012345" --curves 100 --b1 50000

# Run YAFU factorization
python3 yafu-wrapper.py --composite "123456789012345" --mode ecm --curves 100
```

## Project Structure

```
ecm-wrapper/
├── server/              # FastAPI coordination server
│   ├── app/             # Application code
│   ├── migrations/      # Database migrations
│   ├── docker-compose.simple.yml  # Easy Docker setup
│   ├── DEPLOYMENT.md    # Detailed deployment guide
│   └── .env.example     # Environment template
│
├── client/              # Python factorization clients
│   ├── ecm-wrapper.py   # GMP-ECM wrapper
│   ├── yafu-wrapper.py  # YAFU wrapper
│   ├── client.yaml      # Client configuration
│   └── scripts/         # Batch processing scripts
│
└── README.md            # This file
```

## Features

### Server
- **Work Coordination**: Assign ECM work to distributed clients
- **T-Level Tracking**: Monitor progress toward factorization goals
- **Admin Dashboard**: Web-based management interface with secure login
- **API Security**: Admin endpoints protected by API key authentication
- **REST API**: Full OpenAPI documentation at `/docs`

### Client
- **GMP-ECM Support**: Run elliptic curve method factorization
- **YAFU Support**: Run multiple factorization methods (ECM, P-1, P+1, SIQS, NFS)
- **Batch Processing**: Scripts for automated processing
- **Result Submission**: Automatic submission to coordination server

## Documentation

- **Server Deployment**: See [server/DEPLOYMENT.md](server/DEPLOYMENT.md)
- **Client Usage**: See [client/CLAUDE.md](client/CLAUDE.md)
- **API Documentation**: Available at `/docs` when server is running

## Security Features

✅ Admin authentication via API keys
✅ XSS protection with HTML escaping
✅ CORS configured for public API
✅ Environment-based secrets management
✅ Docker secrets support for production

## API Endpoints

### Public (No authentication)
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation
- `POST /api/v1/submit_result` - Submit factorization results
- `GET /api/v1/work` - Get work assignments

### Admin (Requires X-Admin-Key header)
- `GET /api/v1/admin/login` - Admin login page
- `GET /api/v1/admin/dashboard` - Admin dashboard
- `POST /api/v1/admin/composites/upload` - Upload composites
- All other `/api/v1/admin/*` endpoints

## Architecture

```
┌─────────────────────┐    HTTP/API     ┌─────────────────────┐
│   Client (Python)  │◄──────────────►│ ECM Middleware      │
│   • GMP-ECM        │                 │   • Work assignment │
│   • YAFU           │                 │   • T-level tracking│
│   • Batch scripts  │                 │   • Progress monitor│
└─────────────────────┘                 └─────────────────────┘
                                                 │
                                        ┌───────────────────┐
                                        │   PostgreSQL      │
                                        │   Database        │
                                        └───────────────────┘
```

## Requirements

### Server
- Docker & Docker Compose (recommended)
- OR: Python 3.11+ and PostgreSQL 15+

### Client
- Python 3.8+
- GMP-ECM binary (for ecm-wrapper.py)
- YAFU binary (for yafu-wrapper.py)

## Support

- GitHub Issues: Report bugs or request features
- API Docs: `/docs` endpoint for API reference
- Deployment Guide: See `server/DEPLOYMENT.md`