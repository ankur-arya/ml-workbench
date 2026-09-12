# MLflow + scikit-learn workbench

Local experiment tracking for classical sklearn models. Train on bundled
scikit-learn datasets, log parameters / metrics / plots / models to MLflow,
and inspect everything in a local tracking UI. Nothing leaves your machine —
no Databricks, no cloud secrets.

The previous README described Streamlit dataset/trainer apps that were never
committed. This repository is now a runnable MLflow workbench.

## What you get

- **Datasets:** `iris`, `wine`, `breast_cancer` (classification) and `diabetes` (regression)
- **Models:** logistic regression, random forest, gradient boosting, SVC, linear / ridge / lasso, SVR
- **Tracking:** params, train/test metrics, confusion-matrix or residual plots, feature importances, `mlflow.sklearn` model artifacts, optional Model Registry entries
- **CLI + YAML configs** for experiment name, dataset, model, and hyperparameters
- **Comparison runs** that nest several models under one parent MLflow run

Default local store (created in the repo root when you train):

| Path | Role |
| --- | --- |
| `mlflow.db` | SQLite backend: experiments, runs, model registry |
| `mlruns/` | Run artifacts when you train against the SQLite URI directly |
| `mlartifacts/` | Artifact root if you start `mlflow server` instead of `mlflow ui` |

Those paths are gitignored. Use the same backend URI for training and for the UI.

## Requirements

- Python **3.10–3.13** (developed on 3.12)
- `pip` and a virtual environment (`venv`, Poetry, or `uv` all work)

## Setup

```bash
git clone https://github.com/ankur-arya/ml-workbench.git
cd ml-workbench

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Dependencies are pinned in `requirements.txt` (`mlflow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `PyYAML`, `joblib`, `pytest`).

Optional editable install (adds the `workbench` console script):

```bash
pip install -e .
```

If you skip the editable install, run commands from the repo root (or `export PYTHONPATH=$PWD`) so `python -m workbench` can import the package.

**Poetry**

```bash
poetry env use python3.12
poetry run pip install -r requirements.txt
```

**uv**

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Train experiments

From the repo root, with the venv active:

```bash
# Default: iris RandomForest from configs/iris_random_forest.yaml
python -m workbench train --config configs/iris_random_forest.yaml

# Equivalent Makefile target
make train

# Explicit dataset / model / hyperparameters
python -m workbench train \
  --dataset wine \
  --model logistic_regression \
  --experiment sklearn-workbench \
  --param C=0.5 \
  --param max_iter=800

# Regression example
python -m workbench train --config configs/diabetes_ridge.yaml

# Breast cancer Gradient Boosting
python -m workbench train --config configs/breast_cancer_gb.yaml
```

Compare several models (nested MLflow runs under one parent):

```bash
python -m workbench compare --dataset iris --dataset wine --dataset diabetes
make compare
```

List what the workbench knows about:

```bash
python -m workbench info
python -m workbench info --dataset iris
```

Shell wrappers (same defaults, extra args forwarded):

```bash
chmod +x scripts/*.sh
./scripts/train.sh --config configs/iris_random_forest.yaml
./scripts/compare.sh --dataset iris --model random_forest --model logistic_regression
```

### Defaults

| Setting | Default |
| --- | --- |
| Experiment | `sklearn-workbench` |
| Dataset | `iris` |
| Model | `random_forest` |
| Test split | `0.2` (stratified for classification) |
| Seed | `42` |
| Tracking URI | `sqlite:///mlflow.db` or `$MLFLOW_TRACKING_URI` |
| Model registry | on (`dataset-model`, e.g. `iris-random_forest`) |

Linear and kernel models are wrapped in a `StandardScaler` pipeline. Tree ensembles are not.

## Start the local MLflow UI

Train first so the SQLite file exists, then:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Or:

```bash
make ui
./scripts/start_mlflow_ui.sh
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

You should see:

1. Experiment **sklearn-workbench**
2. Individual runs (and a parent `compare-…` run if you used `compare`)
3. Params, tagged metadata, and `test_*` / `train_*` metrics
4. Artifacts: `model/` (sklearn flavor), `plots/`, `reports/`
5. Registered models on the **Models** page (`iris-random_forest`, …)

### Optional: `mlflow server`

Use this when you want a tracking *server* (HTTP) instead of opening the file store directly. Start the server **before** training and point the client at it:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 127.0.0.1 \
  --port 5000

export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m workbench train --config configs/iris_random_forest.yaml
```

Do not mix the two workflows in the same checkout unless you keep the backend URI and artifact root consistent. The file-store + `mlflow ui` path is the documented default.

## Example session

```bash
source .venv/bin/activate
pip install -r requirements.txt

python -m workbench train --dataset iris --model random_forest --param n_estimators=200
python -m workbench train --dataset wine --model gradient_boosting
python -m workbench compare --dataset iris --model logistic_regression --model random_forest

mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

## Project layout

```
workbench/                 Python package (CLI + training)
  cli.py                   argparse: train / compare / info
  train.py                 fit, evaluate, log to MLflow
  datasets.py              sklearn dataset loaders
  models.py                estimator factory + pipelines
  evaluate.py / plots.py   metrics and figure artifacts
  tracking.py              tracking URI + experiment helpers
  config.py                YAML / CLI config
configs/                   ready-to-run experiment YAML
scripts/                   train / compare / start MLflow UI
Makefile                   install, train, compare, ui, server, test
requirements.txt           pinned dependencies
tests/                     smoke tests (temp tracking store)
```

## Tests

```bash
python -m pytest
make test
```

Tests write to a temporary SQLite URI so they do not touch `./mlflow.db`.

## Notes

- Fully local: the default tracking URI is a file on disk.
- `mlruns/` and `mlflow.db` are created at runtime and are not committed.
- Python 3.10+ is required (`list[str]` typing, MLflow 3.x).
