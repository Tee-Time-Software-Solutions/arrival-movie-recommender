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
	uv pip install -e ".[dev,tests]"

help:
	@echo "$(BLUE)Available commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# 1) Boot-up
dev: ## Start backend in dev
	$(VENV_ACTIVATE) && uvicorn $(APP_PATH) --reload --host 0.0.0.0 --port $(PORT)



# 2) CI
format-check: ## Formats
	@echo "$(YELLOW)Running CI formatting$(RESET)"
	@echo "$(BLUE)Checking code formatting with ruff...$(RESET)"
	@$(VENV_ACTIVATE) && ruff format --check src/

format-fix: ## Format code (for pre-commit)
	@echo "$(YELLOW)Formatting code with ruff...$(RESET)"
	$(VENV_ACTIVATE) && ruff format src/

test: ## Run unit tests with pytest
	@echo "$(YELLOW)Running tests...$(RESET)"
	$(VENV_ACTIVATE) && pytest tests/ -v
