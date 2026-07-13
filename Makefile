.PHONY: install test backend frontend docker-build docker-up docker-down lint clean help

help:
	@echo "PP-MFL Makefile"
	@echo "================"
	@echo "install       - Install backend and frontend dependencies"
	@echo "test          - Run all backend and frontend tests"
	@echo "backend       - Run backend tests only"
	@echo "frontend      - Run frontend tests and TypeScript check"
	@echo "docker-build  - Build Docker images"
	@echo "docker-up     - Start all services with Docker Compose"
	@echo "docker-down    - Stop all services"
	@echo "lint          - Run TypeScript check and (if available) Ruff lint"
	@echo "clean         - Remove build artifacts"

install:
	cd backend && pip install -r requirements.txt
	cd frontend && npm ci

test: backend frontend

backend:
	cd backend && python -m pytest tests/ -v --tb=short

frontend:
	cd frontend && npx tsc --noEmit && npm test

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

lint:
	cd frontend && npx tsc --noEmit

clean:
	cd frontend && rm -rf dist node_modules
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
