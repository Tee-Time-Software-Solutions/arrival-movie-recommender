.DEFAULT_GOAL := help
SHELL := /bin/bash


# VARIABLES
RED = \033[31m
GREEN = \033[32m
YELLOW = \033[33m
BLUE = \033[34m
RESET = \033[0m

VENV = ./.venv
VENV_ACTIVATE = source $(VENV)/bin/activate
PORT = 8000
APP_PATH = src.movie_recommender.main:app


# 0) Set-up
install: ## Install packages with uv
	uv venv $(VENV) --python 3.11
	uv sync --group dev

help:
	@echo "$(BLUE)Available commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# 1) Boot-up
dev: ## Start backend in dev
	$(VENV_ACTIVATE) && uvicorn $(APP_PATH) --reload --host 0.0.0.0 --port $(PORT)



# 2) DATABASE
db-create-migration: ## Generate migration files. Usage: make db-create-migration MESSAGE="your description"
	@if [ -z "$(MESSAGE)" ]; then \
		echo "$(RED)❌ Please provide MESSAGE='your description'$(RESET)"; exit 1; \
	fi
	$(VENV_ACTIVATE) && DB_HOST=localhost alembic revision --autogenerate -m "$(MESSAGE)"

db-migrate: ## Apply all pending migrations (dev)
	$(VENV_ACTIVATE) && alembic upgrade head
	@echo "$(GREEN)✅ Migrations applied$(RESET)"

db-migrate-safe: ## Pre-flight checks then migrate (use for prod/staging)
	@echo "$(YELLOW)🔍 Pre-deployment checks...$(RESET)"
	$(VENV_ACTIVATE) && alembic check || (echo "$(RED)❌ Validation failed$(RESET)"; exit 1)
	@echo "Current revision:"
	$(VENV_ACTIVATE) && alembic current
	@echo "Pending migrations:"
	$(VENV_ACTIVATE) && alembic history --verbose | head -10
	$(VENV_ACTIVATE) && alembic upgrade head
	@echo "$(GREEN)✅ Migration complete$(RESET)"

db-downgrade: ## Roll back one migration
	$(VENV_ACTIVATE) && alembic downgrade -1

db-downgrade-to: ## Roll back to specific revision. Usage: make db-downgrade-to MIGRATION_ID=abc123
	@if [ -z "$(MIGRATION_ID)" ]; then \
		echo "$(RED)❌ Please provide MIGRATION_ID=<revision>$(RESET)"; exit 1; \
	fi
	$(VENV_ACTIVATE) && alembic downgrade $(MIGRATION_ID)

db-status: ## Show current migration revision and history
	$(VENV_ACTIVATE) && alembic current && alembic history --verbose


# 3) CI
unit-test: ## Run unit tests
	$(VENV_ACTIVATE) && python -m pytest tests/unit/ -v

format-check: ## Formats
	@echo "$(YELLOW)Running CI formatting$(RESET)"
	@echo "$(BLUE)Checking code formatting with ruff...$(RESET)"
	@$(VENV_ACTIVATE) && ruff format --check src/

format-fix: ## Format code (for pre-commit)
	@echo "$(YELLOW)Formatting code with ruff...$(RESET)"
	$(VENV_ACTIVATE) && ruff format src/
