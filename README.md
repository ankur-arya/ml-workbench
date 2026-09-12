# ClearML + scikit-learn workbench

Local **classical ML** experiment tracking. Train sklearn models on bundled
datasets, log parameters / metrics / plots / models to **ClearML**, compare
candidates in the ClearML WebApp, and promote the winner through ClearML’s
model catalog (tag + publish).

This repository no longer uses MLflow or a custom Streamlit/React UI. The
ClearML WebApp **is** the UI.

## Evaluate → compare → promote

```
train several candidates  →  compare in the WebApp  →  publish the winner
     (ClearML Task)              (project Compare)         (model catalog)
```

1. **Evaluate.** `python -m workbench train` (or `compare`) fits sklearn
   models and auto-logs a ClearML **Task**: hyperparameters, train/test
   scalars, confusion-matrix or residual plots, a `model.joblib` artifact,
   and an **OutputModel** in the project catalog.
2. **Compare.** Sibling tasks share one ClearML **project** (the YAML
   `experiment` field). In the WebApp, select two or more tasks → **Compare**.
3. **Promote.** In **Models**, open the winner → **Publish** and tag
   `production` / `champion`. Same thing from the CLI: `workbench promote`.

## What you get

- **Datasets:** `iris`, `wine`, `breast_cancer` (classification) and `diabetes` (regression)
- **Models:** logistic regression, random forest, gradient boosting, SVC, linear / ridge / lasso, SVR
- **Tracking:** ClearML `Task`, `Logger`, `OutputModel`, optional `Dataset`
- **CLI + YAML** for project/experiment name, dataset(s), model, hyperparameters
- **Multi-model** and **multi-dataset** comparison as sibling tasks in one project
- **Registry:** OutputModel on every train (default), then tag + `publish()` toward production

Linear and kernel models are wrapped in a `StandardScaler` pipeline. Tree ensembles are not.

## Requirements

- Python **3.10–3.13** (developed on 3.12)
- Docker + Compose v2 to run the **local ClearML Server / WebApp** (preferred)
- `pip` and a virtual environment (`venv`, Poetry, or `uv`)

## Setup

```bash
git clone https://github.com/ankur-arya/ml-workbench.git
cd ml-workbench

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Dependencies are pinned in `requirements.txt` (`clearml`, `scikit-learn`,
`pandas`, `numpy`, `matplotlib`, `PyYAML`, `joblib`, `pytest`). MLflow is
not a dependency.

Optional editable install (adds the `workbench` / `mlw` console scripts):

```bash
pip install -e .
```

If you skip the editable install, run commands from the repo root (or
`export PYTHONPATH=$PWD`) so `python -m workbench` can import the package.

## Start ClearML Server + WebApp (preferred)

The workbench is built around a **self-hosted** ClearML Server so compare
and promote stay on your machine.

```bash
# Linux: Elasticsearch needs a higher vm.max_map_count
sudo sysctl -w vm.max_map_count=262144

chmod +x scripts/*.sh
./scripts/clearml-server.sh up
# or: make clearml-up
```

That compose file lives at `deploy/clearml/docker-compose.yml` (official
ClearML images, volumes under `deploy/clearml/data/` instead of `/opt/clearml`).

| URL | Role |
| --- | --- |
| http://localhost:8080 | **WebApp** (compare tasks, model catalog) |
| http://localhost:8008 | API server |
| http://localhost:8081 | File / artifact server |

Wait until the WebApp loads (first boot pulls several images and can take a
few minutes). Then:

1. Open http://localhost:8080 and create a local user.
2. **Settings → Workspace → Create new credentials.**
3. Point the SDK at the server — either:

```bash
./scripts/clearml-server.sh init-config
cp deploy/clearml/clearml.conf.example ~/clearml.conf
# edit access_key / secret_key
```

or export:

```bash
export CLEARML_WEB_HOST=http://localhost:8080
export CLEARML_API_HOST=http://localhost:8008
export CLEARML_FILES_HOST=http://localhost:8081
export CLEARML_API_ACCESS_KEY=...
export CLEARML_API_SECRET_KEY=...
```

or run the wizard: `clearml-init`.

Stop the stack with `./scripts/clearml-server.sh down` (data is kept).

### Hosted quick start (no Docker)

If you cannot run the server locally, use ClearML’s free hosted workspace:

```bash
clearml-init
# Web app:  https://app.clear.ml
# paste the credentials the wizard prints
```

Train/compare/promote are the same commands; the UI is the hosted WebApp
instead of localhost:8080.

`clearml-session` is **not** a replacement for this UI. It opens a remote
Jupyter/VS Code workstation on a ClearML agent. Use it only if you want a
dev box attached to the same server.

### Offline (no server, no WebApp)

```bash
python -m workbench --offline train --config configs/iris_random_forest.yaml
# or:  CLEARML_OFFLINE_MODE=1 python -m workbench train ...
```

ClearML writes a local offline session you can later upload:

```python
from clearml import Task
Task.import_offline_session("/path/to/offline-session.zip")
```

Tests use an in-process recorder (`WORKBENCH_TRACKER=memory`) so CI never
talks to a server.

## Train

From the repo root, with the venv active and ClearML configured:

```bash
# Default: iris RandomForest from configs/iris_random_forest.yaml
python -m workbench train --config configs/iris_random_forest.yaml
make train

# Explicit dataset / model / hyperparameters
python -m workbench train \
  --dataset wine \
  --model logistic_regression \
  --experiment sklearn-workbench \
  --param C=0.5 \
  --param max_iter=800

# Regression
python -m workbench train --config configs/diabetes_ridge.yaml

# Breast cancer Gradient Boosting
python -m workbench train --config configs/breast_cancer_gb.yaml
```

`experiment` is the **ClearML project** name. Related tasks show up together.

### Defaults

| Setting | Default |
| --- | --- |
| Project / experiment | `sklearn-workbench` |
| Dataset | `iris` |
| Model | `random_forest` |
| Test split | `0.2` (stratified for classification) |
| Seed | `42` |
| OutputModel | on (`dataset-model`, e.g. `iris-random_forest`) |
| ClearML Dataset | off (`--register-dataset` to upload the table) |

## Multi-model and multi-dataset comparison

Every pair becomes its own ClearML task. A summary task logs a results table.
Incompatible pairs (e.g. `ridge` on `iris`) are skipped.

```bash
# Several models × several datasets (defaults: all valid models on iris, wine, diabetes)
python -m workbench compare --dataset iris --dataset wine --dataset diabetes
make compare

# Several classifiers on one dataset
python -m workbench compare --config configs/compare-models.yaml
make compare-models

# One model family across datasets
python -m workbench compare --config configs/compare-datasets.yaml
make compare-datasets

# Explicit lists
python -m workbench compare \
  --dataset iris --dataset wine \
  --model logistic_regression --model random_forest
```

List what the workbench knows about:

```bash
python -m workbench info
python -m workbench info --dataset iris
```

Shell wrappers (same defaults, extra args forwarded):

```bash
./scripts/train.sh --config configs/iris_random_forest.yaml
./scripts/compare.sh --config configs/compare-models.yaml
./scripts/promote.sh --experiment sklearn-workbench --metric test_accuracy
```

## Compare runs in the WebApp

1. Open http://localhost:8080 (or https://app.clear.ml).
2. Open project **sklearn-workbench**.
3. In the experiments table, tick two or more tasks (filter by tag
   `compare`, `compare-models`, or a `compare-…` group tag if you used
   `compare`).
4. Click **COMPARE** in the action bar.
5. Use the tabs:
   - **Hyperparameters** — side-by-side `model__*` and split settings
   - **Scalars** — `test_accuracy` / `test_f1_macro` / `test_r2` / `test_rmse`
   - **Plots** — confusion matrices, residuals, feature importances
   - **Artifacts / Models** — `model.joblib`, reports, OutputModel links
6. The `compare-iris-wine-…` controller task holds the summary table if you
   used the compare command.

## Promote / publish the winning model

ClearML’s registry pattern is: keep candidates as OutputModels, then
**tag** the winner and **Publish** it (immutable, catalog-ready). Published
or tagged models can also trigger ClearML Serving.

### WebApp clicks

1. In the same project, open **Models** (project Models tab or the left nav).
2. Open the winning model (name like `iris-random_forest`).
3. Confirm lineage: it should point at the training task you compared.
4. **Add tag** → `production` and `champion` (optional: `staging` first).
5. Click **Publish**. The model becomes read-only.
6. Optional: archive older production models so only the champion is active.

### CLI

```bash
# You already know the model id (Models page → ID)
python -m workbench promote --model-id <MODEL_ID> --publish --tag production --tag champion

# First output model on a training task
python -m workbench promote --task-id <TASK_ID> --publish --tag production

# Auto-select the best scalar in the project (classification default)
python -m workbench promote \
  --experiment sklearn-workbench \
  --metric test_accuracy \
  --publish --tag production --tag champion

# Regression: lower RMSE is better
python -m workbench promote \
  --experiment sklearn-workbench \
  --metric test_rmse --minimize \
  --publish --tag production
```

`make promote` runs the auto-select `test_accuracy` path.

To load a published model later:

```python
from clearml import Model
model = Model.query_models(
    project_name="sklearn-workbench",
    tags=["production", "champion"],
    only_published=True,
    max_results=1,
)[0]
path = model.get_local_copy()  # joblib file
```

## Example session

```bash
source .venv/bin/activate
pip install -r requirements.txt
./scripts/clearml-server.sh up
# create WebApp user + credentials, then clearml-init or ~/clearml.conf

python -m workbench train --dataset iris --model random_forest --param n_estimators=200
python -m workbench train --dataset wine --model gradient_boosting
python -m workbench compare --config configs/compare-models.yaml

# WebApp: http://localhost:8080 → sklearn-workbench → select tasks → Compare
python -m workbench promote --experiment sklearn-workbench --metric test_accuracy --tag production
```

## Project layout

```
workbench/                 Python package (CLI + training)
  cli.py                   argparse: train / compare / promote / info
  train.py                 fit, evaluate, log to ClearML
  promote.py               tag + publish OutputModels
  datasets.py              sklearn dataset loaders
  models.py                estimator factory + pipelines
  evaluate.py / plots.py   metrics and figure artifacts
  tracking.py              ClearML Task / Logger / OutputModel / Dataset
  config.py                YAML / CLI config
configs/                   ready-to-run experiment + compare YAML
deploy/clearml/            docker-compose + clearml.conf.example
scripts/                   train / compare / promote / clearml-server
Makefile                   install, train, compare, clearml-*, promote, test
requirements.txt           pinned dependencies (no MLflow)
tests/                     unit tests (in-process tracker + mocked ClearML)
```

## Tests

```bash
python -m pytest
make test
```

Tests set `WORKBENCH_TRACKER=memory` so they never contact a ClearML Server.
`tracking.py` / `promote.py` SDK calls are unit-tested with mocks.

## MLflow (removed)

The previous revision used a local MLflow SQLite store (`mlflow.db`,
`mlflow ui` on :5000). That path is gone:

- `mlflow` is not in `requirements.txt`
- `scripts/start_mlflow_ui.sh` and `make ui` / `make server` are removed
- `MLFLOW_TRACKING_URI` is ignored
- leftover `mlflow.db` / `mlruns/` / `mlartifacts/` are gitignored

Use ClearML Server + WebApp instead.

## Notes

- Classical sklearn only — no GenAI / LLM eval features.
- `deploy/clearml/data/` is created at runtime and is not committed.
- Python 3.10+ is required (`list[str]` typing).
