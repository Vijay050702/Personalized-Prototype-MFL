# PP-MFL

**Personalized Prototype-Based Multimodal Federated Learning via Client-Adaptive Cross-Modal Knowledge Transfer**

A full-stack application for federated learning with multimodal data, prototype-based personalization, and cross-modal knowledge transfer.

## Quick Start (Docker)

```bash
# Clone and enter the project
git clone <repo-url> pp-mfl
cd pp-mfl

# Configure environment (optional - defaults work out of the box)
cp .env.example .env

# Build and start all services
docker compose up -d

# Check health
docker compose ps
curl http://localhost:80/health

# View logs
docker compose logs -f
```

Services:
- **Frontend**: http://localhost:80
- **API**: http://localhost:80/api/v1
- **Swagger UI**: http://localhost:80/docs
- **ReDoc**: http://localhost:80/redoc

## Architecture

```
┌──────────────┐     ┌──────────────────────────────────────┐
│   Browser    │────▶│           Nginx (port 80)             │
└──────────────┘     │                                      │
                     │  / → serve SPA (index.html)          │
                     │  /api/* → proxy to backend           │
                     │  /docs → proxy to backend             │
                     │  /health → proxy to backend           │
                     └──────────┬───────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │  FastAPI Backend       │
                    │  (port 8000)           │
                    │                        │
                    │  uvicorn 4 workers     │
                    │  /api/v1/* endpoints   │
                    │  /health endpoint      │
                    └────────────────────────┘
```

## Project Structure

```
pp-mfl/
├── backend/               # FastAPI Python backend
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── api/           # Router registration
│   │   ├── core/          # Config, logging, error handlers
│   │   ├── datasets/      # Dataset adapters & registry
│   │   ├── data/          # Multimodal data layer
│   │   ├── models/        # PyTorch models (encoders, fusion, classifiers)
│   │   ├── prototypes/    # Prototype learning engine
│   │   ├── knowledge_transfer/  # Cross-modal transfer
│   │   ├── personalization/     # Client personalization
│   │   ├── evaluation/    # Research evaluation framework
│   │   ├── federated/     # Federated learning aggregation
│   │   ├── training/      # Training engine
│   │   ├── routers/       # REST API endpoints
│   │   ├── schemas/       # Pydantic schemas
│   │   └── services/      # Business logic
│   └── tests/             # 1766 tests
├── frontend/              # React TypeScript frontend
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── src/               # React components, API clients, pages
│   └── tests/             # 322 tests
├── docker-compose.yml
├── .env.example
├── Makefile
└── .github/workflows/ci.yml
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DEBUG` | `false` | Enable debug mode |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `CORS_ORIGINS` | `["*"]` | Allowed CORS origins (JSON array) |
| `SECRET_KEY` | `change-me-in-production` | JWT signing key (change in production!) |
| `ALGORITHM` | `HS256` | JWT algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | JWT token lifetime |
| `VITE_API_URL` | `http://localhost:80` | API URL for frontend |
| `BACKEND_PORT` | `8000` | Host port for backend |
| `FRONTEND_PORT` | `80` | Host port for frontend |

## Deployment

### Docker Compose (Production)

```bash
# Production-ready start
docker compose up -d

# With custom environment
FRONTEND_PORT=8080 BACKEND_PORT=8001 docker compose up -d

# Rebuild after changes
docker compose build --no-cache
docker compose up -d
```

### Manual Development

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
npm ci
npm run dev
```

### Verification

After starting with Docker Compose:

```bash
# All services running
docker compose ps

# Backend health
curl http://localhost:80/health
# → {"status":"ok","version":"0.1.0","service":"PP-MFL"}

# API response
curl http://localhost:80/api/v1/prototypes
# → {"status":"success","data":[...],"total":...}

# Frontend accessible
curl -s http://localhost:80 | head -5
# → <!doctype html>...
```

## Testing

```bash
# All tests
make test

# Backend only
cd backend && python -m pytest tests/ -v

# Frontend only
cd frontend && npm test

# In Docker (run tests inside container)
docker compose exec backend python -m pytest tests/ -v
docker compose exec frontend npm test
```

## Maintenance

```bash
# View logs
docker compose logs -f          # all services
docker compose logs -f backend   # backend only
docker compose logs -f frontend  # frontend only

# Stop services
docker compose down              # stop and remove containers
docker compose down -v           # stop and remove volumes (deletes data!)

# Update images
docker compose pull
docker compose up -d

# Clean build artifacts
make clean
```

## Security

- Backend runs as non-root user (`appuser`)
- Frontend runs as non-root user (`appuser`, uid 1000)
- Nginx adds security headers: X-Frame-Options, X-Content-Type-Options, X-XSS-Protection, Referrer-Policy, Permissions-Policy
- No hardcoded secrets — all config via environment variables
- No unnecessary packages in production images
- Multi-stage builds minimize image size
- Health checks ensure service availability

## Troubleshooting

| Problem | Solution |
|---|---|
| Backend fails to start | Check `docker compose logs backend` for error details |
| Frontend shows blank page | Ensure backend is healthy (`curl http://localhost:80/health`) |
| Port already in use | Set `FRONTEND_PORT=8080 BACKEND_PORT=8001` before `docker compose up` |
| Permission errors | Ensure Docker has permission to write to volumes |
| Slow first start | PyTorch download can take time on first `docker compose build` |
| CORS errors | In dev mode, ensure backend CORS allows frontend origin |
