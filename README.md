# Superconductivity Critical Temperature Predictor

> End-to-end, production-grade ML system for predicting the critical temperature
> of superconducting materials — from raw data through Prefect-orchestrated
> training, MLflow experiment tracking, DVC data versioning, and SageMaker
> deployment, all provisioned with Terraform.

[![Tests](https://github.com/jfarrell8/superconductivity-predictor/.github/workflows/test.yml/badge.svg)](https://github.com/jfarrell8/superconductivity-predictor/.github/workflows/test.yml)
[![Lint](https://github.com/jfarrell8/superconductivity-predictor/.github/workflows/lint.yml/badge.svg)](https://github.com/jfarrell8/superconductivity-predictor/.github/workflows/lint.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

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
│  LOCAL DEV STACK  (docker compose --profile full up)            │
│                                                                 │
│  :8000  FastAPI inference  │  :5000  MLflow UI                  │
│  :4200  Prefect UI         │  :9001  MinIO console (local S3)   │
└─────────────────────────────────────────────────────────────────┘

Training:    Prefect flow  →  MLflow tracking  →  S3 artifacts
Serving:     FastAPI (dev)  /  SageMaker endpoint (prod)
Versioning:  DVC (data + models)  backed by S3
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
│   ├── monitoring/         DriftMonitor  — KS-test per feature
│   ├── orchestration/      Prefect flow  — 6 tasks, retries, cron schedule
│   └── storage/            StorageBackend  — local / S3 abstraction
├── tests/
│   ├── unit/               40+ isolated tests (features, loader, eval, drift,
│   │                         storage, flows), moto S3 mocks
│   └── integration/        FastAPI endpoint tests with mocked model state
├── infra/terraform/
│   ├── modules/            s3 / iam / sagemaker  — reusable modules
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
- Docker + Docker Compose (for local stack)
- AWS CLI configured (for S3/SageMaker features — optional for local dev)

### 1. Install

```bash
git clone https://github.com/jfarrell8/superconductivity-predictor.git
cd superconductivity-predictor
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

**Option B — Prefect flow (recommended):**
```bash
python scripts/run_flow.py --trials 5
```

**Option C — Full local stack with UI:**
```bash
docker compose --profile full up -d

# Open:
#   http://localhost:8000/docs   → FastAPI Swagger UI
#   http://localhost:5000        → MLflow experiment tracking
#   http://localhost:4200        → Prefect flow runs
#   http://localhost:9001        → MinIO (local S3)  user: minioadmin

python scripts/run_flow.py --trials 5
```

### 4. Run tests

```bash
make test          # pytest with coverage
make lint          # ruff + mypy
make test-fast     # skip slow integration tests
```

### 5. Deploy infrastructure (AWS)

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

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Box-Cox target transform** | Raw Tc distribution is right-skewed (skewness ≈ 1.4). Box-Cox reduces this to < 0.1, improving regression convergence. Lambda stored for inference-time inverse transform. |
| **Two-stage feature reduction** | Correlation pruning (81→28) removes redundant elemental statistics. Post-HPO top-15 sweep shows RMSE plateaus — fewer features means fewer lab measurements per sample. |
| **Joint model+HPO search** | Optuna TPE explores model type and architecture jointly, avoiding selection bias from pre-choosing a model family. |
| **Fit/transform protocol** | Every transformer is individually testable, composable, and serialisable. Inference reuses fitted state with no leakage. |
| **StorageBackend abstraction** | One config line switches local↔S3. Unit tests always use `LocalBackend`; CI uses `moto`; production uses `S3Backend`. |
| **OIDC for CI/CD** | No `AWS_ACCESS_KEY_ID` stored in GitHub Secrets. GitHub's OIDC JWT is exchanged for short-lived STS credentials via an assumed IAM role. |
| **Manual production gate** | Models auto-promote to MLflow `Staging` if RMSE improves. `Staging→Production` requires explicit human approval in the MLflow UI — intentional safety gate. |

---

## CI/CD Workflows

| Workflow | Trigger | Steps |
|---|---|---|
| `lint.yml` | Every PR + push | Ruff lint/format check + mypy strict |
| `test.yml` | Every PR + push | pytest + ≥80% coverage enforcement |
| `build.yml` | Push to main | OIDC auth → ECR build + push |
| `deploy.yml` | Push to main (after tests) | Package `model.tar.gz` → S3 → SageMaker endpoint update → smoke test |

---

## API Reference

**Base URL**: `http://localhost:8000`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — model load status + logging enabled flag |
| `GET` | `/model/info` | Metadata: model type, CV RMSE, feature names |
| `POST` | `/predict` | Single-sample inference (returns `request_id` for tracing) |
| `POST` | `/predict/batch` | Batch inference (up to 1,000 samples) |
| `GET` | `/monitoring/drift` | Run KS-test drift detection against reference data |
| `GET` | `/monitoring/logs` | Summary statistics from the prediction log |

**Example:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "mean_atomic_mass": 88.5,
      "wtd_mean_atomic_mass": 91.2,
      "gmean_atomic_mass": 85.3,
      "mean_fie": 6.2,
      "mean_atomic_radius": 1.55,
      "wtd_mean_atomic_radius": 1.48,
      "mean_Density": 5400.0,
      "wtd_mean_Density": 5100.0,
      "mean_ElectronAffinity": 1.2,
      "wtd_mean_ElectronAffinity": 1.1,
      "mean_FusionHeat": 14.5,
      "wtd_mean_FusionHeat": 13.8,
      "mean_ThermalConductivity": 12.0,
      "wtd_mean_ThermalConductivity": 11.5,
      "num_elements_simplified": 3
    }
  }'
```

**Response:**
```json
{
  "predicted_critical_temp_boxcox": 4.312,
  "request_id": "3f4a9c2e-1b7d-4e8a-9f3c-2d1e5a8b7c4f",
  "model_type": "lightgbm",
  "n_features_used": 15
}
```

Every prediction is logged to `logs/predictions.jsonl` with the `request_id`,
input features, and timestamp — enabling downstream drift analysis:

```python
from src.monitoring.logger import PredictionLogger
from src.monitoring.drift import DriftMonitor

# Load logged predictions
df = PredictionLogger.load_as_dataframe("logs/predictions.jsonl")

# Run drift detection against reference training data
import pandas as pd
ref = pd.read_csv("data/processed/train_reduced.csv")[feature_cols]
monitor = DriftMonitor(reference_data=ref, significance_level=0.05)
report = monitor.detect(df)
print(report.summary())
```

To convert the Box-Cox prediction back to Kelvin:
```python
from scipy.special import inv_boxcox
tc_kelvin = inv_boxcox(predicted_boxcox, fitted_lambda)
# fitted_lambda is in models/model_metadata.json
```

---

## Data Versioning (DVC)

```bash
# One-time setup
dvc init
dvc remote add -d s3remote s3://sc-predictor-artifacts/dvc-cache
dvc remote modify s3remote region us-east-1

# Track new data version
dvc add data/raw/train.csv
git add data/raw/train.csv.dvc && git commit -m "chore: update training data"
dvc push

# Reproduce only changed stages
dvc repro

# Compare experiments in a table
dvc exp run --name exp-more-trees --set-param modeling.n_trials=100
dvc exp show
```

---

## Prefect Deployments

```bash
# Deploy both flows to a work pool
prefect deploy --all

# Trigger the weekly retraining manually
prefect deployment run 'superconductivity-training-pipeline/weekly-retraining'

# Quick smoke test (5 trials)
prefect deployment run 'superconductivity-training-pipeline/smoke-test'

# View run history
prefect deployment ls
```

---

## Makefile Targets

```
make install      pip install -e ".[dev]"
make test         pytest with coverage
make test-fast    pytest -m "not slow"
make lint         ruff check + mypy
make format       ruff format
make api          uvicorn src.api.app:app --reload
make flow         python scripts/run_flow.py --trials 5
make stack-up     docker compose --profile full up -d
make stack-down   docker compose --profile full down
make tf-plan      terraform plan (dev environment)
make tf-apply     terraform apply (dev environment)
make clean        remove build artifacts and caches
```

---

## License

MIT — see [LICENSE](LICENSE).
