# Kairos — see PROJECT_BRIEF.md
.DEFAULT_GOAL := help
SHELL := /bin/bash
COMPOSE := docker compose
API_PORT ?= $(shell grep -E '^KAIROS_API_PORT=' .env 2>/dev/null | cut -d= -f2)
API_PORT := $(if $(API_PORT),$(API_PORT),8010)
WEB_PORT ?= $(shell grep -E '^KAIROS_WEB_PORT=' .env 2>/dev/null | cut -d= -f2)
WEB_PORT := $(if $(WEB_PORT),$(WEB_PORT),5173)

.PHONY: help up down restart logs ps build health test test-local lint venv seed demo nuke shell-api shell-db

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

up:  ## Build and start everything (postgres, mlflow, api, web)
	@test -f .env || cp .env.example .env
	$(COMPOSE) up -d --build
	@echo ""
	@echo "  web      http://localhost:$(WEB_PORT)"
	@echo "  api      http://localhost:$(API_PORT)/api/health"
	@echo "  docs     http://localhost:$(API_PORT)/docs"
	@echo "  mlflow   http://localhost:5001"
	@echo ""
	@$(MAKE) --no-print-directory health

down:  ## Stop everything (keeps data volumes)
	$(COMPOSE) down

restart:  ## Restart the API only
	$(COMPOSE) restart api

nuke:  ## Stop everything AND drop the volumes
	$(COMPOSE) down -v

ps:  ## Container status
	$(COMPOSE) ps

logs:  ## Tail logs (make logs S=api)
	$(COMPOSE) logs -f --tail=120 $(S)

build:  ## Rebuild images without starting
	$(COMPOSE) build

health:  ## Wait for the API to answer, then print its health report
	@echo "waiting for api…"
	@for i in $$(seq 1 60); do \
		if curl -fsS http://localhost:$(API_PORT)/api/health >/dev/null 2>&1; then \
			curl -fsS http://localhost:$(API_PORT)/api/health | python3 -m json.tool; exit 0; \
		fi; sleep 2; \
	done; \
	echo "API did not come up in 120s — try: make logs S=api"; exit 1

test:  ## Run the test suite inside the api container
	$(COMPOSE) exec -T api pytest -q

test-local: venv  ## Run the test suite on the host venv
	cd backend && ../.venv/bin/pytest -q

venv:  ## Create the local 3.11 venv for fast test iteration
	@test -d .venv || uv venv --python 3.11 .venv
	@uv pip install --python .venv/bin/python -q -r backend/pyproject.toml --extra dev

lint:  ## Ruff check + format
	cd backend && ../.venv/bin/ruff check --fix . && ../.venv/bin/ruff format .

seed:  ## Build the pre-baked demo run (Phase 10)
	$(COMPOSE) exec -T api python -m scripts.seed_demo

demo:  ## Open the demo in a browser
	@open http://localhost:$(WEB_PORT) 2>/dev/null || xdg-open http://localhost:$(WEB_PORT)

shell-api:  ## Shell into the API container
	$(COMPOSE) exec api bash

shell-db:  ## psql into Postgres
	$(COMPOSE) exec postgres psql -U kairos -d kairos
