.DEFAULT_GOAL := help
SHELL := /bin/bash

PY ?= python3.11
VENV := .venv
BIN := $(VENV)/bin
COMPOSE := docker compose

.PHONY: help
help: ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(BIN)/activate: pyproject.toml
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q -e '.[dev]'
	@touch $@

.PHONY: install
install: $(BIN)/activate ## Create the virtualenv and install dev dependencies

.PHONY: up
up: ## Build and start the full stack (gateway, mocks, redis)
	@test -f .env || cp .env.example .env
	$(COMPOSE) up -d --build --wait
	@echo "gateway: http://localhost:8000"

.PHONY: down
down: ## Stop the stack
	$(COMPOSE) down

.PHONY: logs
logs: ## Tail gateway logs
	$(COMPOSE) logs -f gateway

.PHONY: test
test: install ## Unit + component tests with coverage
	$(BIN)/pytest -m "not integration" --cov --cov-report=term-missing

.PHONY: test-integration
test-integration: install ## Tests against the running compose stack (run `make up` first)
	$(BIN)/pytest -m integration -v

.PHONY: lint
lint: install ## ruff lint + format check
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

.PHONY: fmt
fmt: install ## Auto-format
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

.PHONY: typecheck
typecheck: install ## mypy --strict
	$(BIN)/mypy

.PHONY: check
check: lint typecheck test ## Everything CI runs
