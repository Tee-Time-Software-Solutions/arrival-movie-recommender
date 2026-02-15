.DEFAULT_GOAL := help

# VARIABLES
RED = \033[31m
GREEN = \033[32m
YELLOW = \033[33m
BLUE = \033[34m
RESET = \033[0m

install:
	uv sync

help:
	@echo "$(BLUE)Available commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

format: ## Formats
	@echo "$(YELLOW)Running CI formatting$(RESET)"
	@echo "$(BLUE)Checking code formatting with ruff...$(RESET)"
	@$(VENV_ACTIVATE) && ruff format --check src/
