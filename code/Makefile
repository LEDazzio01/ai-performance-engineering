# AI Performance Engineering - Development Makefile
#
# Common development tasks for the codebase.
# Run `make help` to see available targets.

.PHONY: help test lint validate coverage check all clean

# Default target
help:
	@echo "AI Performance Engineering - Development Tasks"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Testing:"
	@echo "  test              Run all pytest tests"
	@echo "  test-fast         Run tests without slow markers"
	@echo "  test-cov          Run tests with coverage report"
	@echo ""
	@echo "Validation:"
	@echo "  validate          Validate all benchmark imports"
	@echo "  validate-ch7      Validate chapter 7 imports only"
	@echo "  lint              Run linters and static analysis"
	@echo "  metrics           Analyze get_custom_metrics() coverage"
	@echo "  metrics-validate  Validate metric quality"
	@echo ""
	@echo "Reports:"
	@echo "  coverage          Generate benchmark coverage report"
	@echo "  coverage-md       Generate coverage report as Markdown"
	@echo "  coverage-json     Generate coverage report as JSON"
	@echo ""
	@echo "Maintenance:"
	@echo "  audit             Audit for silent fallback patterns"
	@echo "  audit-fix         Fix silent fallback patterns"
	@echo "  clean             Remove __pycache__ and .pyc files"
	@echo ""
	@echo "All-in-one:"
	@echo "  check             Run all validation checks (no fixes)"
	@echo "  all               Run tests + validation + coverage"

# =============================================================================
# Testing
# =============================================================================

test:
	python -m pytest tests/ -v --tb=short

test-fast:
	python -m pytest tests/ -v --tb=short -m "not slow"

test-cov:
	python -m pytest tests/ -v --tb=short --cov=common/python --cov-report=term-missing --cov-report=html

# =============================================================================
# Validation
# =============================================================================

validate:
	python scripts/validate_imports.py

validate-ch%:
	python scripts/validate_imports.py --chapter $*

lint:
	@echo "Running flake8..."
	-flake8 common/python/ scripts/ --max-line-length=120 --ignore=E501,W503
	@echo ""
	@echo "Running mypy..."
	-mypy common/python/benchmark_metrics.py common/python/profiler_config.py --ignore-missing-imports

metrics:
	python scripts/update_custom_metrics.py --analyze

metrics-validate:
	python scripts/update_custom_metrics.py --validate

metrics-apply:
	python scripts/update_custom_metrics.py --apply

# =============================================================================
# Reports
# =============================================================================

coverage:
	python scripts/benchmark_coverage.py

coverage-md:
	python scripts/benchmark_coverage.py --markdown

coverage-json:
	python scripts/benchmark_coverage.py --json

# =============================================================================
# Maintenance
# =============================================================================

audit:
	python scripts/audit_silent_fallbacks.py

audit-fix:
	python scripts/audit_silent_fallbacks.py --fix

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage 2>/dev/null || true

# =============================================================================
# Composite Targets
# =============================================================================

check: validate metrics lint
	@echo ""
	@echo "✅ All checks passed!"

all: test validate metrics coverage
	@echo ""
	@echo "✅ All tasks completed!"

