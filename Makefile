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
up: ## Build and start the stack; `make up SEMANTIC=1` adds the semantic-cache models
	@test -f .env || cp .env.example .env
	SEMANTIC=$(if $(SEMANTIC),true,false) $(COMPOSE) up -d --build --wait
	@echo "gateway:    http://localhost:8000"
	@echo "grafana:    http://localhost:3000  (dashboard: Switchyard gateway)"
	@echo "prometheus: http://localhost:9090"

.PHONY: down
down: ## Stop the stack
	$(COMPOSE) down

.PHONY: key
key: ## Create an API key: make key NAME=me [RPM=600] [TPM=1000000]
	@$(COMPOSE) exec -T gateway python -m switchyard.cli keys create --name "$(or $(NAME),dev)" --rpm $(or $(RPM),600) --tpm $(or $(TPM),1000000)

.PHONY: logs
logs: ## Tail gateway logs
	$(COMPOSE) logs -f gateway

.PHONY: redis
redis: ## Start only Redis (what `make test` needs)
	$(COMPOSE) up -d --wait redis

.PHONY: test
test: install redis ## Unit + component tests with coverage (starts Redis if needed)
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

.PHONY: loadtest
loadtest: install ## Run every load-test scenario against the mocks (~1 h), then refresh README tables
	@curl -sf localhost:8000/readyz >/dev/null || $(MAKE) up
	$(BIN)/python loadtest/run.py all
	$(BIN)/python scripts/render_readme.py

.PHONY: dashboard
dashboard: ## Regenerate the Grafana dashboard JSON from scripts/build_dashboard.py
	$(BIN)/python scripts/build_dashboard.py

.PHONY: readme
readme: ## Regenerate README tables from results/
	$(BIN)/python scripts/render_readme.py

.PHONY: eval-semantic
eval-semantic: ## Re-run the semantic-cache threshold evaluation (needs .[eval])
	$(BIN)/python scripts/eval_semantic_threshold.py --verifier cross-encoder/quora-distilroberta-base

.PHONY: check
check: lint typecheck test ## Everything CI runs
