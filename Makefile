.PHONY: up down logs ps db-init seed data verify test lint clean reset

COMPOSE = docker compose -p aegisflow

## Start the full stack
up:
	$(COMPOSE) up -d --build
	@echo "Stack started. Run 'make ps' to check status."

## Stop containers and preserve volumes
down:
	$(COMPOSE) down

## Stop containers and remove volumes
reset:
	$(COMPOSE) down -v

## Show container status
ps:
	$(COMPOSE) ps

## Follow service logs
logs:
	$(COMPOSE) logs -f --tail=100

## Initialize the database
db-init:
	$(COMPOSE) exec -T postgres sh -c 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB" -f /docker-entrypoint-initdb.d/01_init.sql'

## Load M1 demo data (direct SQL historical transactions; use 'make ingest-stream' for M2 API streaming)
seed:
	python scripts/seed_db.py

## Download datasets
data:
	python scripts/download_datasets.py

## Run milestone verification
verify:
	python scripts/verify_m1.py

## Run tests
test:
	python -m pytest tests/ -v

## Format and lint
lint:
	python -m black .
	python -m flake8 .

## Remove Python cache directories
clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +

# ---- M2 targets ----
# RECIPEPREFIX applies to rules written after this declaration.
.RECIPEPREFIX = >

PY ?= python
M2_PY := $(if $(wildcard .venv-m2/bin/python),.venv-m2/bin/python,$(PY))
M2_DVC := $(if $(wildcard .venv-m2/bin/dvc),.venv-m2/bin/dvc,dvc)
M2_PATH := PYTHONPATH=services/ingestion/src:ml/src:.
M2_COMPOSE := docker compose -f docker-compose.yml -f docker-compose.m2.yml

up-m2:
> $(M2_COMPOSE) up -d --build ingestion graph-job

migrate:
> $(M2_PATH) $(M2_PY) -m alembic upgrade head

seed-fraud:
> $(M2_PATH) $(M2_PY) db/seed/seed_known_fraud.py

## Start M2 Ingestion API on :8001 (/health, /ready, /metrics, POST /api/v1/transactions)
ingest:
> $(M2_PATH) $(M2_PY) -m uvicorn ingestion.main:app --reload --port 8001

## Stream M1 demo transactions through the M2 Ingestion API -> Postgres + Redis
ingest-stream:
> $(M2_PATH) $(M2_PY) scripts/ingest_transactions.py

## Reconcile Postgres-persisted API transactions missing Redis online velocity features
recover-features:
> $(M2_PATH) $(M2_PY) scripts/recover_pending_features.py

graph-once:
> $(M2_PATH) $(M2_PY) -m aegis_ml.features.graph_features --once

features-pit:
> $(if $(wildcard .venv-m2/bin),PATH="$(abspath .venv-m2/bin):$$PATH" )$(M2_DVC) repro features_pit

parity:
> $(M2_PATH) PARITY_REDIS_URL=$${PARITY_REDIS_URL:-redis://localhost:6380/15} $(M2_PY) -m pytest -q tests/m2/unit/test_m2_parity.py

test-m2:
> $(M2_PATH) $(M2_PY) -m pytest -q tests/m2/unit tests/m2/api

test-m2-int:
> $(M2_PATH) $(M2_PY) -m pytest -q tests/m2/integration

verify-m2:
> $(M2_PATH) $(M2_PY) scripts/verify_m2.py

m2: test-m2 test-m2-int
> $(M2_PATH) $(M2_PY) scripts/verify_m2.py --skip-tests

.PHONY: up-m2 migrate seed-fraud ingest ingest-stream recover-features graph-once features-pit parity test-m2 test-m2-int verify-m2 m2

# ---- M3 targets: append below the M2 block. .RECIPEPREFIX applies only to rules
# written after it, so M2's tab-indented rules keep working and every M3 rule
# survives being copied out of this PDF. ----
.RECIPEPREFIX = >
PY ?= python
M3_PATH := PYTHONPATH=ml/src:.
ACTOR ?=

prepare-ieee:
> dvc repro prepare_ieee

train-fraud:
> dvc repro train_fraud

compare:
> $(M3_PATH) $(PY) scripts/champion.py --compare

promote:
> $(M3_PATH) $(PY) scripts/champion.py --promote $(if $(strip $(ACTOR)),--actor $(strip $(ACTOR)),)

rollback:
> $(M3_PATH) $(PY) scripts/champion.py --rollback $(if $(strip $(ACTOR)),--actor $(strip $(ACTOR)),)

generate:
> $(M3_PATH) $(PY) -m aegis_ml.scenarios.generate_typologies --out data/processed

typologies:
> $(M3_PATH) $(PY) -m aegis_ml.evaluation.typology_eval

test-m3:
> $(M3_PATH) $(PY) -m pytest -q tests/m3/unit

test-m3-int:
> $(M3_PATH) $(PY) -m pytest -q tests/m3/integration

verify-m3:
> $(M3_PATH) $(PY) scripts/verify_m3.py

m3: test-m3 train-fraud typologies test-m3-int verify-m3

.PHONY: prepare-ieee train-fraud compare promote rollback typologies \
 test-m3 test-m3-int verify-m3 m3 generate
