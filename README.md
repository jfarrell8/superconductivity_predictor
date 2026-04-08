# Superconductivity Critical Temperature Predictor

> End-to-end, production-grade ML system for predicting the critical temperature
> of superconducting materials — from raw data through Prefect-orchestrated
> training, MLflow experiment tracking, DVC data versioning, and a live
> inference API with drift monitoring, all provisioned with Terraform.

[![Tests](https://github.com/jfarrell8/superconductivity_predictor/actions/workflows/test.yml/badge.svg)](https://github.com/jfarrell8/superconductivity_predictor/actions/workflows/test.yml)
[![Lint](https://github.com/jfarrell8/superconductivity_predictor/actions/workflows/lint.yml/badge.svg)](https://github.com/jfarrell8/superconductivity_predictor/actions/workflows/lint.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

---

## Live Demo

| Service | URL |
|---|---|
| **Interactive dashboard** | [sc-predictor-dash.onrender.com](https://sc-predictor-dash.onrender.com) |
| **REST API + Swagger UI** | [sc-predictor-api.onrender.com/docs](https://sc-predictor-api.onrender.com/docs) |

The dashboard lets you enter elemental composition features, get a predicted
critical temperature, view the distribution of all predictions made so far,
and inspect per-feature drift status against the training data distribution.

> **Note:** Hosted on Render's free tier — the first request after a period
> of inactivity may take 30–60 seconds to warm up. Subsequent predictions
> are fast.

---

## Scientific Context

Superconductors lose all electrical resistance below a material-specific
**critical temperature (Tc)**. Discovering new high-Tc materials is one of
the central challenges in condensed matter physics and materials science.

This system predicts Tc from 81 compositionally-derived **Magpie-style
descriptors** — elemental statistics (mean, weighted mean, geometric mean,
entropy, range) over atomic mass, atomic radius, electronegativity, ionisation
energy, electron affinity, fusion heat, thermal conductivity, and valence
electron count. These descriptors are standard across materials ML benchmarks
(Matbench, AFLOW, OQMD).

**Dataset**: [UCI Superconductivity](https://archive.ics.uci.edu/dataset/464/superconductivty+data)
— 21,263 superconductors, 81 features.

---

## Full Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  LIVE (Render free tier)                                        │
│                                                                 │
│  Dash dashboard  →  FastAPI inference API  →  trained model     │
│  drift monitoring  ·  prediction logging  ·  Swagger UI        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  LOCAL DEV STACK  (docker compose --profile full up)            │
│                                                                 │
│  :8000  FastAPI inference  │  :5000  MLflow UI                  │
│  :4200  Prefect UI         │  :9001  MinIO console (local S3)   │
└─────────────────────────────────────────────────────────────────┘

Training:    Prefect flow  →  MLflow tracking  →  S3 artifacts
Serving:     FastAPI (dev / Render)  /  SageMaker endpoint (prod)
Versioning:  DVC (data + models) backed by S3
Infra:       Terraform  →  S3 + IAM + SageMaker + CloudWatch
CI/CD:       GitHub Actions  →  OIDC auth (no stored AWS keys)
```

---

## Project Structure

```
superconductivity_predictor/
├── src/
│   ├── data/               DataLoader  — schema validation, SHA-256 manifest
│   ├── features/           FeatureEngineer  — 4 typed transformers (OOP)
│   ├── models/             ModelTrainer  — Optuna HPO + MLflow tracking
│   ├── evaluation/         ModelEvaluator  — metrics, SHAP, reduction sweep
│   ├── api/
│   │   ├── app.py          FastAPI inference service
│   │   └── sagemaker/      inference.py  — SageMaker container entry point
│   ├── monitoring/         DriftMonitor (KS-test) + PredictionLogger (JSONL)
│   ├── orchestration/      Prefect flow  — 6 tasks, retries, cron schedule
│   └── storage/            StorageBackend  — local / S3 abstraction
├── dash/
│   ├── app.py              Dash demo dashboard (3 panels)
│   └── requirements.txt    dash, plotly, requests, gunicorn
├── tests/
│   ├── unit/               140+ isolated tests, moto S3 mocks
│   └── integration/        FastAPI endpoint tests with mocked model state
├── infra/terraform/
│   ├── modules/            s3 / iam / ecr / sagemaker  — reusable modules
│   └── environments/       dev / prod  — environment compositions
├── configs/
│   └── experiment_v1.yaml  Single source of truth for all hyperparameters
├── scripts/
│   ├── run_flow.py         Prefect flow CLI (local or deployed)
│   ├── run_pipeline.py     Lightweight script (no Prefect server needed)
│   └── download_data.py    UCI dataset downloader
├── dvc.yaml                DVC pipeline stages
├── prefect.yaml            Prefect deployment manifest (2 deployments)
├── Dockerfile              Multi-stage build (builder → slim runtime)
├── docker-compose.yml      Full local stack (API + MLflow + MinIO + Prefect)
└── .github/workflows/      lint / test / build / deploy  (4 workflows)
```

---

## Quickstart

### Prerequisites

- Python 3.10+
- Docker + Docker Compose (for full local stack)
- AWS CLI configured (for S3/SageMaker features — optional for local dev)

### 1. Install

```bash
git clone https://github.com/jfarrell8/superconductivity_predictor.git
cd superconductivity_predictor
pip install -e ".[dev]"
```

### 2. Download data

```bash
python scripts/download_data.py
```

### 3. Run the pipeline

**Option A — Simple script (no server needed):**
```bash
python scripts/run_pipeline.py --config configs/experiment_v1.yaml --trials 5
```

**Option B — Prefect flow with MLflow tracking (recommended):**
```bash
# Terminal 1 — MLflow tracking server
mlflow ui --port 5000

# Terminal 2 — run the flow
python scripts/run_flow.py --trials 5

# Open http://localhost:5000 to see experiment runs
```

**Option C — Full local stack with all UIs:**
```bash
docker compose --profile full up -d

# Open:
#   http://localhost:8000/docs   → FastAPI Swagger UI
#   http://localhost:5000        → MLflow experiment tracking
#   http://localhost:4200        → Prefect flow runs
#   http://localhost:9001        → MinIO (local S3)  user: minioadmin

python scripts/run_flow.py --trials 5
```

### 4. Start the inference API

```bash
uvicorn src.api.app:app --reload --port 8000

# Try it:
curl http://localhost:8000/health
curl http://localhost:8000/predict/example   # returns ready-to-use payload
curl http://localhost:8000/features          # feature descriptions + ranges
```

### 5. Run tests

```bash
make test          # pytest with coverage
make lint          # ruff + mypy
make test-fast     # skip slow integration tests
```

### 6. Run the Dash dashboard locally

```bash
# With the FastAPI server running on :8000
cd dash
pip install -r requirements.txt
API_BASE_URL=http://localhost:8000 python app.py

# Open http://localhost:8050
```

### 7. Deploy infrastructure (AWS)

```bash
cd infra/terraform/environments/dev
terraform init && terraform apply
```

See [infra/README.md](infra/README.md) for the full infrastructure guide.

---

## ML Pipeline Stages

| Stage | Class | Task (Prefect) | Description |
|---|---|---|---|
| Ingestion | `DataLoader` | `ingest_data` | Load + validate CSV, SHA-256 manifest, optional S3 pull |
| Features | `FeatureEngineer` | `engineer_features` | Binning → Box-Cox → correlation pruning → scaling |
| Split | — | `split_data` | Stratified 80/20 train/test split |
| Training | `ModelTrainer` | `train_model` | Optuna TPE HPO across 4 model families, MLflow tracking |
| Evaluation | `ModelEvaluator` | `evaluate_model` | RMSE/MAE/R², SHAP, feature reduction sweep, Prefect artifact |
| Persistence | `ModelRegistry` | `persist_artifacts` | Save to disk + S3, MLflow Staging promotion |

---

## API Reference

**Live base URL**: `https://sc-predictor-api.onrender.com`
**Local base URL**: `http://localhost:8000`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — model load status + logging enabled |
| `GET` | `/model/info` | Model type, CV RMSE, feature names, training rows |
| `GET` | `/predict/example` | Ready-to-use payload with current feature names and medians |
| `GET` | `/features` | Feature descriptions, medians, min/max/p25/p75 ranges |
| `POST` | `/predict` | Single-sample inference — missing features auto-imputed |
| `POST` | `/predict/batch` | Batch inference (up to 1,000 samples) |
| `GET` | `/monitoring/drift` | KS-test drift detection vs. training reference data |
| `GET` | `/monitoring/logs` | Prediction log summary statistics |

**Quickest way to try it:**
```bash
# 1. Get a ready-to-use payload with correct feature names
curl https://sc-predictor-api.onrender.com/predict/example

# 2. Copy the "features" object and POST it
curl -X POST https://sc-predictor-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"features": { <paste from step 1> }}'
```

**Response:**
```json
{
  "predicted_critical_temp_boxcox": 4.312,
  "request_id": "3f4a9c2e-1b7d-4e8a-9f3c-2d1e5a8b7c4f",
  "model_type": "random_forest",
  "n_features_used": 15,
  "imputed_features": []
}
```

Feature names change between training runs depending on which model wins the
Optuna search. Always call `/predict/example` or `/features` first to get
the correct names for the currently loaded model.

To convert the Box-Cox prediction back to Kelvin:
```python
from scipy.special import inv_boxcox
tc_kelvin = inv_boxcox(predicted_boxcox, fitted_lambda)
# fitted_lambda is logged in models/model_metadata.json and MLflow
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Box-Cox target transform** | Raw Tc distribution is right-skewed (skewness ≈ 1.4). Box-Cox reduces this to < 0.1, improving regression convergence. Lambda stored for inference-time inverse transform. |
| **Two-stage feature reduction** | Correlation pruning (81→28) removes redundant elemental statistics. Post-HPO top-15 sweep shows RMSE plateaus — fewer features means fewer lab measurements per sample. |
| **Joint model+HPO search** | Optuna TPE explores model type and architecture jointly, avoiding selection bias from pre-choosing a model family. |
| **Median imputation at inference** | Training set medians saved to `feature_medians.json`. Missing features are filled automatically at inference time — partial inputs are accepted gracefully. |
| **Fit/transform protocol** | Every transformer follows sklearn's fit/transform interface — individually testable, composable, and serialisable. No leakage between train and test. |
| **StorageBackend abstraction** | One config line switches local↔S3. Unit tests use `LocalBackend`; CI uses `moto`; production uses `S3Backend`. |
| **OIDC for CI/CD** | No `AWS_ACCESS_KEY_ID` stored in GitHub Secrets. GitHub's OIDC JWT is exchanged for short-lived STS credentials via an assumed IAM role. |
| **Manual production gate** | Models auto-promote to MLflow `Staging` if RMSE improves. `Staging→Production` requires explicit human approval — intentional safety gate. |

---

## Monitoring

Every prediction through the API is logged to `logs/predictions.jsonl` with
a UUID `request_id`, input features, and timestamp:

```python
from src.monitoring.logger import PredictionLogger
from src.monitoring.drift import DriftMonitor

# Load all logged predictions as a DataFrame
df = PredictionLogger.load_as_dataframe("logs/predictions.jsonl")

# Run KS-test drift detection against training reference data
import pandas as pd
ref = pd.read_csv("data/processed/train_reduced.csv")
monitor = DriftMonitor(reference_data=ref, significance_level=0.05)
report = monitor.detect(df)
print(report.summary())
```

The `/monitoring/drift` endpoint runs this automatically against the current
prediction log and returns per-feature KS statistics and drift flags. The
Dash dashboard polls this endpoint every 30 seconds and displays a live
green/red status per feature.

---

## CI/CD Workflows

| Workflow | Trigger | Steps |
|---|---|---|
| `lint.yml` | Every PR + push | Ruff lint/format check + mypy type checking |
| `test.yml` | Every PR + push | pytest + ≥70% coverage enforcement |
| `build.yml` | Push to main | OIDC auth → Docker build → ECR push |
| `deploy.yml` | Push to main (after tests) | Package `model.tar.gz` → S3 → SageMaker endpoint update → smoke test |

---

## Data Versioning (DVC)

```bash
# One-time setup
dvc init
dvc remote add -d s3remote s3://sc-predictor-artifacts/dvc-cache
dvc remote modify s3remote region us-east-1

# Track a new data version
dvc add data/raw/train.csv
git add data/raw/train.csv.dvc && git commit -m "chore: update training data"
dvc push

# Reproduce only changed pipeline stages
dvc repro

# Roll back to a previous data version
git checkout v0.1.0 && dvc checkout
```

---

## Prefect Deployments

```bash
# Deploy both flows to a work pool
prefect deploy --all

# Trigger weekly retraining manually
prefect deployment run 'superconductivity-training-pipeline/weekly-retraining'

# Quick smoke test (5 trials)
prefect deployment run 'superconductivity-training-pipeline/smoke-test'
```

---

## Makefile Targets

```
make install        pip install -e ".[dev]"
make test           pytest with coverage
make test-fast      pytest -m "not slow"
make lint           ruff check + mypy
make format         ruff format + ruff check --fix
make api            uvicorn src.api.app:app --reload
make flow           python scripts/run_flow.py --trials 5
make pipeline       python scripts/run_pipeline.py --trials 5
make stack-up       docker compose --profile full up -d
make stack-down     docker compose --profile full down
make api-logs       tail and pretty-print prediction log
make package-sm     bundle model artifacts for SageMaker
make tf-plan        terraform plan (dev environment)
make tf-apply       terraform apply (dev environment)
make clean          remove build artifacts and caches
```

---

## License

MIT — see [LICENSE](LICENSE).