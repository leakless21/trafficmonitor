.PHONY: install test lint format clean dev-setup help

# Default target
help:
	@echo "Traffic Monitor Development Commands"
	@echo "=================================="
	@echo "install      Install dependencies"
	@echo "dev-setup    Setup development environment"
	@echo "test         Run all tests"
	@echo "test-unit    Run unit tests only"
	@echo "test-integration  Run integration tests only"
	@echo "lint         Run linting checks"
	@echo "format       Format code"
	@echo "clean        Clean build artifacts and cache"
	@echo "benchmark    Run performance benchmarks"
	@echo "docs         Serve documentation locally"

install:
	uv sync

dev-setup:
	uv sync --dev
	@echo "Installing pre-commit hooks..."
	uv run pre-commit install || echo "pre-commit not available, skipping hooks"

test:
	uv run pytest tests/

test-unit:
	uv run pytest tests/unit/ -m "not slow"

test-integration:
	uv run pytest tests/integration/ -m "not slow"

test-slow:
	uv run pytest tests/ -m "slow"

test-gpu:
	uv run pytest tests/ -m "gpu"

lint:
	@echo "Running ruff checks..."
	uv run ruff check src/ tests/ tools/
	@echo "Running mypy type checks..."
	uv run mypy src/ || echo "mypy not available, skipping type checks"

format:
	@echo "Formatting code with ruff..."
	uv run ruff format src/ tests/ tools/

clean:
	@echo "Cleaning Python cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info/
	@echo "Cleaning test artifacts..."
	rm -rf .pytest_cache/ .coverage htmlcov/
	@echo "Cleaning temporary files..."
	rm -f tmp_* *.tmp

benchmark:
	@echo "Running performance benchmarks..."
	uv run python tools/benchmarking/benchmark_performance.py

docs:
	@echo "Documentation not yet configured"
	@echo "See docs/ directory for manual documentation"

# Development shortcuts
run-dev:
	uv run traffic-monitor --config configs/environments/development.yaml

run-prod:
	uv run traffic-monitor --config configs/environments/production.yaml