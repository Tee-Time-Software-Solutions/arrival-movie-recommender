
.DEFAULT_GOAL := help

# VARIABLES
RED = \033[31m
GREEN = \033[32m
YELLOW = \033[33m
BLUE = \033[34m
RESET = \033[0m

PROJECT_NAME = movie_recommender
BACKEND_ENV_FILE_SYNCED_PATH = ../backend/env_config/synced/.env.$(ENVIRONMENT)



help: ## Help
	@echo "$(BLUE)Available commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install projet-wide dependencies
	$(MAKE) -C backend install

dev-start: ## Start dev (no rebuild, fast). Use dev-rebuild if deps changed
	$(MAKE) check-enviroment-variables
	BACKEND_ENV_FILE=$(BACKEND_ENV_FILE_SYNCED_PATH) docker compose -f deployment/docker-compose.yml -f deployment/docker-compose.dev.yml -p $(PROJECT_NAME) up

dev-rebuild: ## Rebuild images then start dev (use when deps change)
	$(MAKE) check-enviroment-variables
	BACKEND_ENV_FILE=$(BACKEND_ENV_FILE_SYNCED_PATH) docker compose -f deployment/docker-compose.yml -f deployment/docker-compose.dev.yml -p $(PROJECT_NAME) up --build

check-enviroment-variables:
	@if [ -z "$$ENVIRONMENT" ]; then \
		echo "Error: ENVIRONMENT  must be defined. Do `export ENVIRONMENT={dev|production}`"; \
		exit 1; \
	fi
	echo "Environment is: $(ENVIRONMENT)"
