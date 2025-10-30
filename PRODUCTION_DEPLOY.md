# Production Deployment Checklist

## Prerequisites

### 1. GitHub Secrets Configuration

Configure these secrets in your GitHub repository (Settings → Secrets and variables → Actions):

| Secret Name | Description | How to Generate |
|-------------|-------------|-----------------|
| `DROPLET_SSH_KEY` | SSH private key for server access | Already configured (reused from other projects) |
| `DROPLET_HOST` | Server IP or hostname | Already configured (Your Digital Ocean droplet IP) |
| `DROPLET_USER` | SSH username | Already configured (Usually `deploy` or `root`) |
| `DOMAIN_NAME` | Your production domain | e.g., `ecm.example.com` |
| `POSTGRES_PASSWORD` | Database password | `openssl rand -hex 32` |
| `API_SECRET_KEY` | API cryptographic key | `openssl rand -hex 64` |
| `ADMIN_API_KEY` | Admin dashboard access key | `openssl rand -hex 32` |

**Note:** The first 3 secrets (`DROPLET_*`) are already configured if you have other projects deploying to the same Digital Ocean droplet.

### 2. Server Setup

Your GitHub Action will automatically:
- ✅ Create deployment directory (`~/ecm-distributed`)
- ✅ Clone/pull the repository
- ✅ Write secrets to `server/secrets/` directory
- ✅ Generate self-signed SSL certificates (if needed)
- ✅ Configure nginx with your domain
- ✅ Build and start Docker containers
- ✅ Run health checks

**Manual server prerequisites (one-time setup):**
```bash
# Install Docker and Docker Compose (if not already installed)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Give deploy user Docker permissions (IMPORTANT!)
sudo usermod -aG docker deploy

# Install Docker Compose plugin
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Log out and back in for group changes to take effect
# OR run: newgrp docker

# Verify deploy user can run Docker
docker ps

# Optional: Install OpenSSL for SSL certificate generation (usually pre-installed)
sudo apt-get install openssl
```

### 3. DNS Configuration

Point your domain to the server:
```
A Record: @ → YOUR_SERVER_IP
A Record: www → YOUR_SERVER_IP
```

### 4. SSL Certificates

**Option A: Self-Signed (Auto-generated - Development/Testing)**
- The deploy script automatically generates self-signed certificates
- ⚠️ Browsers will show security warnings
- Good for testing, but NOT for production

**Option B: Let's Encrypt (Recommended for Production)**
After first deployment with self-signed certs:
```bash
ssh deploy@server
cd ~/ecm-distributed/server

# Install certbot
sudo apt-get install certbot

# Get certificates
sudo certbot certonly --standalone -d YOUR_DOMAIN
# Note: Stop nginx first: docker-compose -f docker-compose.prod.yml.active stop nginx

# Copy certificates
sudo cp /etc/letsencrypt/live/YOUR_DOMAIN/fullchain.pem ssl/cert.pem
sudo cp /etc/letsencrypt/live/YOUR_DOMAIN/privkey.pem ssl/key.pem
sudo chown $USER:$USER ssl/*.pem

# Restart services
docker-compose -f docker-compose.prod.yml.active restart nginx
```

**Option C: Custom Certificates**
Place your certificates in `server/ssl/` before first deployment:
```bash
# On server
mkdir -p ~/ecm-distributed/server/ssl
# Upload your cert.pem and key.pem
```

## Deployment Process

### Option 1: Automatic Deployment (Recommended)

Push to `main` branch to automatically deploy:
```bash
git add .
git commit -m "Deploy to production"
git push origin main
```

Changes to `server/**` trigger automatic deployment.

### Option 2: Manual Deployment

Trigger deployment manually via GitHub Actions:
1. Go to Actions tab in GitHub
2. Select "Deploy to Digital Ocean" workflow
3. Click "Run workflow"
4. Select environment (production/staging)
5. Click "Run workflow"

## Post-Deployment Verification

### 1. Check Deployment Status

GitHub Actions will automatically verify:
- ✅ API health endpoint responding
- ✅ HTTPS certificate valid
- ✅ Services running

### 2. Manual Verification

```bash
# Check service health
curl https://YOUR_DOMAIN/health

# View API documentation
open https://YOUR_DOMAIN/docs

# Access admin dashboard
open https://YOUR_DOMAIN/api/v1/admin/login
# Use ADMIN_API_KEY from GitHub secrets
```

### 3. Check Logs

```bash
ssh deploy@server
cd ~/ecm-distributed/server
docker-compose -f docker-compose.prod.yml logs -f api
```

## Architecture

### Secret Management Flow

```
GitHub Secrets
     ↓
GitHub Actions (deploy.yml)
     ↓
SSH to Server
     ↓
Write to server/secrets/*.txt
     ↓
Docker Secrets Mount to /run/secrets/*
     ↓
app/config.py reads from secret files
```

### Files in Production

```
~/ecm-distributed/  (in deploy user's home directory)
├── server/
│   ├── secrets/                        # Created by GitHub Actions
│   │   ├── postgres_password.txt       # From POSTGRES_PASSWORD secret
│   │   ├── api_secret_key.txt          # From API_SECRET_KEY secret
│   │   └── admin_api_key.txt           # From ADMIN_API_KEY secret
│   ├── ssl/                            # Created by GitHub Actions
│   │   ├── cert.pem                    # SSL certificate (self-signed or real)
│   │   └── key.pem                     # SSL private key
│   ├── nginx.conf                      # Template (in git)
│   ├── nginx.conf.prod                 # Generated with domain (not in git)
│   ├── docker-compose.prod.yml         # Template (in git)
│   ├── docker-compose.prod.yml.active  # Generated with domain (not in git)
│   └── app/                            # Application code
├── .git/                               # Git repository
└── .github/
    └── workflows/
        └── deploy.yml                  # Deployment automation (all logic here)
```

## Security Checklist

- [x] Strong random passwords (32+ chars)
- [x] Secrets stored in GitHub Secrets (not in code)
- [x] Docker secrets used for sensitive data
- [x] HTTPS with valid SSL certificate
- [ ] Firewall configured (ports 80, 443, 22 only)
- [ ] Regular security updates scheduled
- [ ] Database backups configured
- [ ] Rate limiting at nginx level

## Troubleshooting

### Deployment fails with "Permission denied"

Check SSH key is correctly configured:
```bash
# On your local machine
ssh-keygen -t ed25519 -C "github-actions"
cat ~/.ssh/id_ed25519.pub

# On server
echo "YOUR_PUBLIC_KEY" >> ~/.ssh/authorized_keys
```

### Secrets not working

Verify secrets are written correctly:
```bash
ssh deploy@server
cd ~/ecm-distributed/server/secrets
ls -la
cat postgres_password.txt  # Should show the password
```

### Database connection errors

Check PostgreSQL is running:
```bash
cd ~/ecm-distributed/server
docker-compose -f docker-compose.prod.yml.active ps postgres
docker-compose -f docker-compose.prod.yml.active logs postgres
```

### Migration errors

Run migrations manually:
```bash
cd ~/ecm-distributed/server
docker-compose -f docker-compose.prod.yml.active exec api alembic upgrade head
```

## Rollback Procedure

### Option 1: Via GitHub Actions (Recommended)
```bash
# On your local machine, rollback the repository
git log --oneline  # Find previous working commit
git reset --hard COMMIT_HASH
git push origin main --force  # Triggers automatic deployment

# OR manually trigger deployment from GitHub Actions UI
# - Go to Actions tab → "Deploy to Digital Ocean"
# - Run workflow on specific commit
```

### Option 2: Manual Rollback on Server
```bash
ssh deploy@server
cd ~/ecm-distributed
git log --oneline  # Find previous commit
git reset --hard COMMIT_HASH

# Redeploy manually
cd server
docker-compose -f docker-compose.prod.yml.active down
docker-compose -f docker-compose.prod.yml.active up -d --build
```

## Monitoring

### Key Endpoints to Monitor

- `https://YOUR_DOMAIN/health` - Health check (200 OK expected)
- `https://YOUR_DOMAIN/api/v1/admin/composites/status` - Queue status
- `https://YOUR_DOMAIN/docs` - API documentation

### Logs

```bash
cd ~/ecm-distributed/server

# API logs
docker-compose -f docker-compose.prod.yml.active logs -f api

# Database logs
docker-compose -f docker-compose.prod.yml.active logs -f postgres

# Nginx logs
docker-compose -f docker-compose.prod.yml.active logs -f nginx
```

## Backup and Restore

### Backup Database

```bash
cd ~/ecm-distributed/server
docker-compose -f docker-compose.prod.yml.active exec postgres \
  pg_dump -U ecm_user ecm_distributed > backup_$(date +%Y%m%d).sql
```

### Restore Database

```bash
cd ~/ecm-distributed/server
docker-compose -f docker-compose.prod.yml.active exec -T postgres \
  psql -U ecm_user ecm_distributed < backup_20240101.sql
```

## Support

- GitHub Issues: Check existing issues or create new one
- Logs: Enable `DEBUG` level in production if needed (temporarily)
- Documentation: `/docs` endpoint has interactive API docs
