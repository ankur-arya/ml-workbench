PYTHON ?= python
MLFLOW ?= mlflow
TRACKING_URI ?= sqlite:///mlflow.db
HOST ?= 127.0.0.1
PORT ?= 5000
export PYTHONPATH := $(CURDIR)

.PHONY: help install install-dev train compare info ui server test clean

help:
	@echo "MLflow + scikit-learn workbench"
	@echo ""
	@echo "  make install     Create .venv and install pinned dependencies"
	@echo "  make train       Train the default iris RandomForest config"
	@echo "  make compare     Compare several models on iris, wine, diabetes"
	@echo "  make info        List datasets and models"
	@echo "  make ui          Start the local MLflow tracking UI (port $(PORT))"
	@echo "  make server      Start mlflow server (SQLite + ./mlartifacts)"
	@echo "  make test        Run unit/smoke tests"
	@echo "  make clean       Remove local tracking files and caches"

install:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

install-dev: install
	.venv/bin/pip install -e .

train:
	$(PYTHON) -m workbench --tracking-uri $(TRACKING_URI) train --config configs/iris_random_forest.yaml

compare:
	$(PYTHON) -m workbench --tracking-uri $(TRACKING_URI) compare --dataset iris --dataset wine --dataset diabetes

info:
	$(PYTHON) -m workbench info

ui:
	$(MLFLOW) ui --backend-store-uri $(TRACKING_URI) --host $(HOST) --port $(PORT)

server:
	$(MLFLOW) server --backend-store-uri $(TRACKING_URI) --default-artifact-root ./mlartifacts --host $(HOST) --port $(PORT)

test:
	$(PYTHON) -m pytest

clean:
	rm -rf mlruns mlartifacts mlflow.db mlflow.db-journal .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
