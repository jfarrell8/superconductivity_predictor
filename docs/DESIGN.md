# Technical Design Document
## Superconductivity Critical Temperature Predictor — v0.2.0

**Author**: [Your Name]
**Domain**: Materials Informatics / Physical Sciences ML

---

## 1. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        TRAINING PATH                                 │
│                                                                      │
│  Raw CSV (UCI / S3)                                                  │
│       │                                                              │
│       ▼                                                              │
│  ┌────────────┐   SHA-256 + schema   ┌──────────────────┐           │
│  │ DataLoader │─────── manifest ────▶│  DataManifest    │           │
│  └─────┬──────┘                      │  (provenance)    │           │
│        │                             └──────────────────┘           │
│        ▼                                                             │
│  ┌──────────────────────┐                                            │
│  │   FeatureEngineer    │  BinningTransformer                        │
│  │                      │  TargetTransformer  (Box-Cox, stores λ)    │
│  │                      │  FeaturePruner      (corr + MI)            │
│  │                      │  FeatureScaler      (StandardScaler)       │
│  └──────────┬───────────┘                                            │
│             │                                                        │
│             ▼                                                        │
│  ┌──────────────────┐     ┌─────────────────────────────────┐       │
│  │  ModelTrainer    │────▶│  MLflow Tracking Server         │       │
│  │  (Optuna TPE)    │     │  • Every trial as nested run    │       │
│  │  4 model families│     │  • Best params + metrics        │       │
│  └──────────┬───────┘     │  • Model artifact (xgb/lgbm)   │       │
│             │             │  • Model Registry (Staging)     │       │
│             ▼             └─────────────────────────────────┘       │
│  ┌──────────────────┐                                                │
│  │  ModelEvaluator  │  RMSE / MAE / R² / SHAP / reduction sweep     │
│  └──────────┬───────┘                                                │
│             │                                                        │
│             ▼                                                        │
│  ┌──────────────────┐     ┌──────────────────────────────────┐      │
│  │  ModelRegistry   │────▶│  S3 / Local  (StorageBackend)    │      │
│  │  (save artifacts)│     │  models/best_model_top15.pkl     │      │
│  └──────────────────┘     │  models/top_features.json        │      │
│                           │  models/model_metadata.json      │      │
│                           └──────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION LAYER                            │
│                                                                     │
│  Prefect 3 Flow: training_pipeline                                  │
│  ┌──────────┐  ┌─────────────┐  ┌────────┐  ┌──────┐  ┌────────┐  │
│  │ingest    │─▶│engineer     │─▶│ split  │─▶│train │─▶│persist │  │
│  │data      │  │features     │  │ data   │  │model │  │artifacts│  │
│  │(retries=3│  │(retries=2)  │  │        │  │      │  │(retries=│  │
│  │cache=✓)  │  │             │  │        │  │      │  │3)       │  │
│  └──────────┘  └─────────────┘  └────────┘  └──┬───┘  └────────┘  │
│                                                 │                   │
│                                          evaluate_model             │
│                                          (metrics + SHAP +          │
│                                           Prefect artifact)         │
│                                                                     │
│  Deployed: prefect.yaml  │  Schedule: cron "0 2 * * 0" (weekly)    │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                       SERVING PATH                                   │
│                                                                      │
│  FastAPI  (local dev / ECS)     SageMaker Endpoint  (production)     │
│  POST /predict                  aws sagemaker-runtime invoke-endpoint│
│  POST /predict/batch            inference.py: model_fn / predict_fn  │
│  GET  /health                                                        │
│  GET  /model/info               DataCapture → S3 → DriftMonitor     │
│  GET  /monitoring/drift                                              │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                    DATA VERSIONING (DVC)                             │
│                                                                      │
│  git tracks:  dvc.yaml, configs/, src/, models/top_features.json    │
│  dvc tracks:  data/raw/*.csv, data/processed/*.csv, models/*.pkl     │
│  S3 stores:   DVC cache + artifacts (sc-predictor-artifacts bucket)  │
│                                                                      │
│  dvc repro   → re-runs only changed pipeline stages                  │
│  dvc push    → syncs data + artifacts to S3                          │
│  dvc exp run → tracks experiment as a DVC experiment (alternative    │
│                to MLflow for lightweight comparisons)                │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                  CLOUD INFRASTRUCTURE (Terraform)                    │
│                                                                      │
│  modules/s3          — data + artifact buckets, versioning, KMS     │
│  modules/iam         — least-privilege roles (SageMaker, CI/CD OIDC,│
│                        Prefect worker)                               │
│  modules/sagemaker   — model package group, endpoint, data capture,  │
│                        auto-scaling, CloudWatch alarms               │
│                                                                      │
│  environments/dev    — ml.m5.large, single AZ, 90-day retention     │
│  environments/prod   — ml.m5.4xlarge, multi-AZ, 365-day, cross-     │
│                        region replication, auto-scaling 1-4 instances│
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Inventory

| Layer | Component | File | Responsibility |
|---|---|---|---|
| Data | `DataLoader` | `src/data/loader.py` | Load, validate, SHA-256 manifest |
| Features | `FeatureEngineer` | `src/features/engineer.py` | Box-Cox, binning, correlation pruning, scaling |
| Storage | `StorageBackend` | `src/storage/store.py` | Unified local/S3 abstraction |
| Training | `ModelTrainer` | `src/models/trainer.py` | Optuna HPO + MLflow tracking |
| Evaluation | `ModelEvaluator` | `src/evaluation/evaluator.py` | Metrics, SHAP, feature reduction sweep |
| Serving | FastAPI app | `src/api/app.py` | REST inference + drift endpoint |
| Serving | SageMaker script | `src/api/sagemaker/inference.py` | Production endpoint entry point |
| Monitoring | `DriftMonitor` | `src/monitoring/drift.py` | KS-test per feature |
| Orchestration | Prefect flow | `src/orchestration/flows.py` | DAG, retries, schedules, artifacts |
| Infra | Terraform | `infra/terraform/` | S3, IAM, SageMaker, CloudWatch |
| Versioning | DVC | `dvc.yaml`, `.dvc/config` | Data + model artifact lineage |

---

## 3. Full JD Mapping

| Lila JD Requirement | Implementation | Files |
|---|---|---|
| **End-to-end ML pipelines** | Prefect flow with 6 tasks, retry policies, cron schedule | `src/orchestration/flows.py`, `prefect.yaml` |
| **Data ingestion** | `DataLoader` + SHA-256 manifest + S3 pull | `src/data/loader.py` |
| **Feature engineering** | 4 typed transformers (OOP, fit/transform API) | `src/features/engineer.py` |
| **Training + HPO** | Optuna TPE, 4 model families, MLflow nested runs | `src/models/trainer.py` |
| **Evaluation** | RMSE/MAE/R², SHAP, feature-reduction sweep | `src/evaluation/evaluator.py` |
| **Deployment (FastAPI)** | `/predict`, `/batch`, `/health`, `/drift` | `src/api/app.py` |
| **Deployment (SageMaker)** | Built-in sklearn container, data capture, auto-scaling | `src/api/sagemaker/inference.py`, `infra/terraform/modules/sagemaker/` |
| **Monitoring** | KS-test drift per feature + SageMaker DataCapture | `src/monitoring/drift.py` |
| **Observability** | MLflow run tracking, Prefect UI artifacts, CloudWatch alarms | Throughout |
| **Robust testing** | 50+ unit + integration tests, moto S3 mocks, ≥80% coverage | `tests/` |
| **CI/CD** | lint → test → build → deploy, OIDC (no stored keys) | `.github/workflows/` |
| **Containers** | Multi-stage Dockerfile, full docker-compose stack | `Dockerfile`, `docker-compose.yml` |
| **Cloud infra** | Terraform modules for S3, IAM, SageMaker; dev + prod envs | `infra/terraform/` |
| **Data provenance** | DVC pipeline + S3 remote, SHA-256 manifests | `dvc.yaml`, `src/data/loader.py` |
| **Workflow orchestration** | Prefect 3 flow with parameterised deployments | `src/orchestration/flows.py`, `prefect.yaml` |
| **Model registry** | MLflow Model Registry + SageMaker Model Package Group | `src/models/trainer.py`, `infra/terraform/modules/sagemaker/` |
| **Strong typing** | Full annotations, mypy strict | All `src/` modules, `pyproject.toml` |
| **Scientific domain** | Superconductivity (materials science), Magpie descriptors | `README.md`, `docs/` |

---

## 4. Local Development Quickstart

```bash
# 1. Clone and install
git clone https://github.com/yourhandle/superconductivity-predictor
cd superconductivity-predictor
pip install -e ".[dev]"

# 2. Download data
python scripts/download_data.py

# 3a. Run pipeline (simple, no orchestration)
python scripts/run_pipeline.py --config configs/experiment_v1.yaml --trials 5

# 3b. Run via Prefect flow (recommended)
python scripts/run_flow.py --trials 5

# 4. Start full local stack (API + MLflow + MinIO + Prefect)
docker compose --profile full up -d

# 5. Open UIs
#   API docs:    http://localhost:8000/docs
#   MLflow:      http://localhost:5000
#   Prefect:     http://localhost:4200
#   MinIO:       http://localhost:9001  (minioadmin / minioadmin)

# 6. Run tests
pytest tests/ -v
```

---

## 5. Data Versioning Workflow

```bash
# First-time setup
dvc init
dvc remote add -d s3remote s3://sc-predictor-artifacts/dvc-cache
dvc remote modify s3remote region us-east-1

# After downloading new data
dvc add data/raw/train.csv
git add data/raw/train.csv.dvc .gitignore
git commit -m "chore: update training data"
dvc push

# Reproduce pipeline (only re-runs changed stages)
dvc repro

# Compare experiments
dvc exp run --name experiment-v2 --set-param modeling.n_trials=100
dvc exp show
```

---

## 6. Infrastructure Workflow

```bash
# One-time bootstrap (creates S3 state bucket + DynamoDB lock table first)
cd infra/terraform/environments/dev
terraform init
terraform plan
terraform apply

# Update infrastructure
terraform plan -out=tfplan
terraform apply tfplan

# Destroy dev environment
terraform destroy
```

---

## 7. Production Deployment Flow

```
Developer opens PR
       │
       ▼
GitHub Actions: lint.yml + test.yml (must pass)
       │
       ▼ (merge to main)
GitHub Actions: build.yml
  → OIDC auth to AWS (no stored keys)
  → Build + push Docker image to ECR
       │
       ▼
GitHub Actions: deploy.yml
  → Package SageMaker inference bundle (model.tar.gz → S3)
  → Create new SageMaker endpoint config
  → Update endpoint (zero-downtime blue/green)
  → Smoke test live endpoint
       │
       ▼ (weekly, Sunday 02:00 UTC)
Prefect: training_pipeline flow
  → Pull latest data from S3
  → Retrain with full HPO
  → Evaluate + log to MLflow
  → Promote to Staging if RMSE < threshold
  → Production promotion requires manual approval in MLflow UI
```
