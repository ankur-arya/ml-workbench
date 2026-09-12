PYTHON ?= python
HOST ?= 127.0.0.1
PORT ?= 8000
export PYTHONPATH := $(CURDIR)

.PHONY: help install install-dev frontend train compare demo info ui test clean

help:
	@echo "Workbench — evaluate, compare, promote (no Docker)"
	@echo ""
	@echo "  make install     Create .venv and install Python deps"
	@echo "  make frontend    npm install + build the SPA"
	@echo "  make ui          Build UI if needed and serve API+UI on :$(PORT)"
	@echo "  make demo        Run the built-in iris bakeoff"
	@echo "  make train       Train the default iris RandomForest config"
	@echo "  make compare     Compare three classifiers on iris"
	@echo "  make test        pytest"
	@echo "  make clean       Remove local store, caches, frontend dist"

install:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

install-dev: install
	.venv/bin/pip install -e .

frontend:
	cd frontend && npm install && npm run build

train:
	$(PYTHON) -m workbench train --config configs/iris_random_forest.yaml

compare:
	$(PYTHON) -m workbench compare --config configs/compare-models.yaml

demo:
	$(PYTHON) -m workbench demo

info:
	$(PYTHON) -m workbench info

ui:
	$(PYTHON) -m workbench ui --host $(HOST) --port $(PORT)

test:
	$(PYTHON) -m pytest

clean:
	rm -rf .workbench .pytest_cache frontend/dist frontend/node_modules
	find . -type d -name __pycache__ -exec rm -rf {} +
