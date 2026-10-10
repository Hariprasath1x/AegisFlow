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

## Load demo data
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
