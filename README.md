# Workbench

**Evaluate. Compare. Promote.**

A local-first workbench for classical (tabular / sklearn) ML. You train a few
candidate models, read a **scorecard**, and promote a winner to production with
one intentional action.

This is not a generic experiment dump and not a GenAI studio. It is built to
beat ClearML / MLflow UX for the evaluate → compare → promote loop — without
Docker, without five services, and without leaving your machine.

## Why this exists

MLflow and ClearML are run browsers. Workbench is a **decision tool**:

1. Run an experiment with several candidates on one or more datasets
2. Land on a leaderboard with a recommended winner
3. Promote that winner to the model registry (with who / when / why)

Day-1 path: `pip install` + one command. SQLite + files on disk. No Docker.

## Quick start (no Docker)

```bash
git clone https://github.com/ankur-arya/ml-workbench.git
cd ml-workbench

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Starts API + UI. First launch builds the React SPA (needs Node 18+).
python -m workbench ui
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

Click **Run demo experiment** — three iris classifiers, a ranked scorecard, and
a one-click promote.

`uv` works the same way:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
python -m workbench ui
```

Makefile equivalents: `make install`, `make ui`, `make demo`, `make test`.

## CLI

```bash
# One model
python -m workbench train --config configs/iris_random_forest.yaml

# Multi-model scorecard (prints a ranked table)
python -m workbench compare --config configs/compare-models.yaml

# Multi-dataset
python -m workbench compare --config configs/compare-datasets.yaml

# Promote the recommended winner
python -m workbench promote --experiment exp_… --note "Best holdout accuracy on iris"

# Built-in iris bakeoff
python -m workbench demo

python -m workbench info
```

Train flags still accept `--dataset`, `--model`, `--param KEY=VALUE`.

## What you get

| Surface | Role |
| --- | --- |
| **Experiments** | Status, best metric, candidate count, last updated |
| **New experiment wizard** | Datasets × models × a primary metric, then run locally |
| **Compare (hero)** | Leaderboard, dataset facets, recommended winner, side-by-side |
| **Run detail** | Params, metrics, artifacts, dataset lineage |
| **Registry** | candidate → staging → production (champion / challenger) |
| **Datasets** | sklearn builtins + local CSV registration |

Bundled datasets: `iris`, `wine`, `breast_cancer` (classification), `diabetes`
(regression). Models: logistic regression, random forest, gradient boosting,
SVC, linear / ridge / lasso, SVR.

## Local storage

Override the data directory with `WORKBENCH_HOME` (or `--home`). Default is
`./.workbench` in the current working directory.

| Path | Role |
| --- | --- |
| `.workbench/workbench.db` | SQLite: experiments, runs, metrics, registry |
| `.workbench/artifacts/` | Plots, reports, `model.joblib` |
| `.workbench/datasets/` | Copies of registered CSV files |

Those paths are gitignored.

## Architecture

```
CLI / UI ──► FastAPI ──► Store (SQLite)
                │
                ├── sklearn runner (fit, metrics, plots)
                ├── ranking (leaderboard + winner)
                └── registry (promote + audit)
```

The React + Vite SPA is the product UI. `workbench ui` builds it if needed and
serves it from the same process as the API. ClearML and MLflow are **not** on
the default path.

Optional later: point `WORKBENCH_HOME` at a shared disk, or put the API behind
a reverse proxy. The schema does not assume localhost.

## Tests

```bash
python -m pytest
```

Coverage: catalog consistency, leaderboard ranking + constraints, exclusive
production promote with an audit trail, real sklearn train/compare, REST demo
→ promote.

## Project layout

```
workbench/           SDK, store, runner, FastAPI, CLI
frontend/            React + Vite SPA
configs/             YAML for train / compare
tests/               pytest
scripts/             train / compare / promote / ui wrappers
```

## Screenshots

See [`docs/screenshots/`](docs/screenshots/) after you run the UI. The compare
screen is the home of an experiment — not a raw run list.
