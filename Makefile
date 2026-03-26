# Makefile — Developer shortcuts. Run `make help` to list targets.

.PHONY: help install test test-fast test-unit test-integration \
        lint format type-check \
        pipeline flow flow-deploy \
        api api-check \
        stack-up stack-down stack-logs stack-api \
        dvc-setup dvc-repro dvc-push dvc-pull \
        prefect-server prefect-deploy prefect-run \
        tf-init tf-plan tf-apply tf-destroy tf-fmt tf-validate \
        clean clean-all

CYAN  := \033[0;36m
RESET := \033[0m

CONFIG ?= configs/experiment_v1.yaml
TRIALS ?= 5
TF_ENV ?= dev
TF_DIR  = infra/terraform/environments/$(TF_ENV)
PYTHON  = python

# ─────────────────────────────────────────────────────────────────────────────

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | sort \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-22s$(RESET) %s\n", $$1, $$2}'

install: ## Install package and all dev dependencies
	pip install -e ".[dev]"

test: ## Run full test suite with coverage (>=80% enforced)
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html:htmlcov --cov-fail-under=80

test-fast: ## Run tests excluding slow tests
	pytest tests/ -v -m "not slow" --cov=src --cov-report=term-missing

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

lint: ## Run ruff linter + mypy type checker
	ruff check src/ tests/
	mypy src/

format: ## Auto-format with ruff
	ruff format src/ tests/
	ruff check --fix src/ tests/

type-check: ## Run mypy strict type checking only
	mypy src/

pipeline: ## Run pipeline directly (no Prefect server required)
	$(PYTHON) scripts/run_pipeline.py --config $(CONFIG) --trials $(TRIALS)

flow: ## Run the Prefect flow locally
	$(PYTHON) scripts/run_flow.py --config $(CONFIG) --trials $(TRIALS)

flow-deploy: ## Submit a run to a deployed Prefect work pool
	$(PYTHON) scripts/run_flow.py --deploy --name weekly-retraining

api: ## Start the FastAPI inference server with hot-reload
	uvicorn src.api.app:app --reload --port 8000 --host 0.0.0.0

api-check: ## Hit the local health endpoint
	curl -s http://localhost:8000/health | python -m json.tool

api-logs: ## Tail the prediction log (last 20 entries)
	tail -n 20 logs/predictions.jsonl 2>/dev/null | python -c "import sys,json; [print(json.dumps(json.loads(l), indent=2)) for l in sys.stdin]" || echo "No prediction log found."

stack-up: ## Start the full local dev stack (API + MLflow + MinIO + Prefect)
	docker compose --profile full up -d
	@printf "\n  $(CYAN)API docs:$(RESET)  http://localhost:8000/docs\n"
	@printf "  $(CYAN)MLflow:$(RESET)    http://localhost:5000\n"
	@printf "  $(CYAN)Prefect:$(RESET)   http://localhost:4200\n"
	@printf "  $(CYAN)MinIO:$(RESET)     http://localhost:9001  (minioadmin / minioadmin)\n\n"

stack-down: ## Stop and remove the local dev stack
	docker compose --profile full down

stack-logs: ## Tail logs from all stack services
	docker compose --profile full logs -f

stack-api: ## Start API service only
	docker compose up -d api

dvc-setup: ## Initialise DVC and configure S3 remote
	dvc init
	dvc remote add -d s3remote s3://sc-predictor-artifacts/dvc-cache
	dvc remote modify s3remote region us-east-1

dvc-repro: ## Re-run only changed DVC pipeline stages
	dvc repro

dvc-push: ## Push tracked data and artifacts to S3
	dvc push

dvc-pull: ## Pull tracked data and artifacts from S3
	dvc pull

package-sm: ## Package model artifacts into SageMaker-ready tar.gz
	$(PYTHON) scripts/package_sagemaker.py --model-dir models --output models/best_model_top15.tar.gz

package-sm-upload: ## Package and upload model bundle to S3
	$(PYTHON) scripts/package_sagemaker.py --model-dir models --output models/best_model_top15.tar.gz --upload

prefect-server: ## Start a local Prefect server
	prefect server start

prefect-deploy: ## Deploy both flows to the configured work pool
	prefect deploy --all

prefect-run: ## Trigger the smoke-test deployment (5 trials)
	prefect deployment run 'superconductivity-training-pipeline/smoke-test'

tf-init: ## Initialise Terraform (TF_ENV=dev|prod)
	cd $(TF_DIR) && terraform init

tf-plan: ## Plan Terraform changes
	cd $(TF_DIR) && terraform plan -out=tfplan

tf-apply: ## Apply planned Terraform changes
	cd $(TF_DIR) && terraform apply tfplan

tf-destroy: ## Destroy Terraform-managed resources (CAUTION)
	@echo "WARNING: Destroying environment: $(TF_ENV)"
	cd $(TF_DIR) && terraform destroy

tf-fmt: ## Format all Terraform files
	terraform fmt -recursive infra/

tf-validate: ## Validate Terraform configurations
	cd $(TF_DIR) && terraform validate

clean: ## Remove Python caches and test artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage coverage.xml

clean-all: clean ## Remove everything including built packages
	rm -rf dist/ build/ *.egg-info/
