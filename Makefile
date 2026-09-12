PYTHON ?= python
export PYTHONPATH := $(CURDIR)

.PHONY: help install install-dev train compare compare-models compare-datasets info promote clearml-up clearml-down clearml-status clearml-init test clean

help:
	@echo "ClearML + scikit-learn workbench"
	@echo ""
	@echo "  make install           Create .venv and install pinned dependencies"
	@echo "  make train             Train the default iris RandomForest config"
	@echo "  make compare           Compare default models on iris, wine, diabetes"
	@echo "  make compare-models    Several classifiers on iris"
	@echo "  make compare-datasets  Random forest across classification datasets"
	@echo "  make info              List datasets and models"
	@echo "  make clearml-up        Start local ClearML Server / WebApp (:8080)"
	@echo "  make clearml-down      Stop the local ClearML stack"
	@echo "  make clearml-init      Print local SDK env / config hints"
	@echo "  make promote           Tag+publish best test_accuracy in sklearn-workbench"
	@echo "  make test              Run unit/smoke tests (in-process tracker)"
	@echo "  make clean             Remove caches and local ClearML offline folders"

install:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

install-dev: install
	.venv/bin/pip install -e .

train:
	$(PYTHON) -m workbench train --config configs/iris_random_forest.yaml

compare:
	$(PYTHON) -m workbench compare --dataset iris --dataset wine --dataset diabetes

compare-models:
	$(PYTHON) -m workbench compare --config configs/compare-models.yaml

compare-datasets:
	$(PYTHON) -m workbench compare --config configs/compare-datasets.yaml

info:
	$(PYTHON) -m workbench info

promote:
	$(PYTHON) -m workbench promote --experiment sklearn-workbench --metric test_accuracy --tag production --tag champion

clearml-up:
	./scripts/clearml-server.sh up

clearml-down:
	./scripts/clearml-server.sh down

clearml-status:
	./scripts/clearml-server.sh status

clearml-init:
	./scripts/clearml-server.sh init-config

test:
	$(PYTHON) -m pytest

clean:
	rm -rf .pytest_cache .clearml artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
