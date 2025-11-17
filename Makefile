.PHONY: help test build up down logs health clean

help:
	@echo "Available commands:"
	@echo "  make test      - Run all tests"
	@echo "  make build     - Build Docker image"
	@echo "  make up        - Start services"
	@echo "  make down      - Stop services"
	@echo "  make logs      - View logs"
	@echo "  make health    - Check API health"
	@echo "  make clean     - Clean up"

test:
	@echo "🧪 Running tests..."
	pytest tests/ -v

build:
	@echo "🐳 Building Docker image..."
	docker compose build

up:
	@echo "▶️  Starting services..."
	docker compose up -d
	@sleep 5
	@echo "✅ Services started!"
	@echo "   API: http://localhost:8000"
	@echo "   Docs: http://localhost:8000/docs"

down:
	@echo "⏹️  Stopping services..."
	docker compose down

logs:
	docker compose logs -f

health:
	@echo "🏥 Checking API health..."
	@curl -s http://localhost:8000/health | python -m json.tool

clean:
	@echo "🧹 Cleaning up..."
	docker compose down -v
	@echo "✅ Cleanup complete!"
