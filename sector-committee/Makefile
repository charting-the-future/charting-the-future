# Makefile for sector-committee
.PHONY: test test-all test-unit test-integration clean help

# Default Python command
PYTHON := uv run
PYTEST := $(PYTHON) pytest

help:  ## Show this help
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

test: test-unit  ## Run unit tests (fast)

test-unit:  ## Run only unit tests
	@echo "Running unit tests..."
	$(PYTEST) tests/unit/ -v

test-integration:  ## Run only integration tests
	@echo "Running integration tests..."
	$(PYTEST) tests/integration/ -v

test-all: test-unit test-integration  ## Run all tests (unit + integration)

test-slow:  ## Run only slow/deep-research tests
	@echo "Running slow tests..."
	$(PYTEST) -m slow -v

test-fast:  ## Run only fast tests (excluding slow markers)
	@echo "Running fast tests..."
	$(PYTEST) -m "not slow" -v

coverage:  ## Run unit tests with coverage report
	@echo "Running unit tests with coverage..."
	$(PYTEST) tests/unit/ --cov=sector_committee --cov-report=term-missing --cov-report=html

coverage-html:  ## Generate and open HTML coverage report
	@echo "Generating HTML coverage report..."
	$(PYTEST) tests/unit/ --cov=sector_committee --cov-report=html
	@echo "Opening coverage report..."
	@open htmlcov/index.html 2>/dev/null || xdg-open htmlcov/index.html 2>/dev/null || echo "Coverage report saved to htmlcov/index.html"

coverage-all:  ## Run all tests with coverage report
	@echo "Running all tests with coverage..."
	$(PYTEST) tests/ --cov=sector_committee --cov-report=term-missing --cov-report=html

clean:  ## Clean up cache and temporary files
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true

lint:  ## Run linting with ruff
	$(PYTHON) ruff check .

lint-fix:  ## Run linting with ruff and auto-fix issues
	$(PYTHON) ruff check . --fix
	$(PYTHON) ruff format .

format:  ## Format code with ruff
	$(PYTHON) ruff format .

install:  ## Install dependencies
	uv sync

dev-install:  ## Install development dependencies
	uv sync --all-extras
