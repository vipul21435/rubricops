.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help install lint format typecheck test cov check demo clean

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

install: ## Create the virtualenv from uv.lock and install git hooks
	$(UV) sync --frozen
	$(UV) run pre-commit install

lint: ## Ruff lint and format check
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

format: ## Apply ruff fixes and formatting
	$(UV) run ruff check --fix src tests
	$(UV) run ruff format src tests

typecheck: ## mypy --strict over src/
	$(UV) run mypy

test: ## Run the test suite
	$(UV) run pytest -q

cov: ## Run tests with branch coverage (fails under 85%)
	$(UV) run pytest -q --cov --cov-report=term-missing --cov-report=xml

check: lint typecheck cov ## Everything CI runs

demo: ## End-to-end demo (grows with each slice; see PLAN.md)
	$(UV) run rubricops --version
	$(UV) run rubricops config

clean: ## Remove caches and build output
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov dist build
