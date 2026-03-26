# Infrastructure Guide

This directory contains Terraform code that provisions all cloud resources
for the Superconductivity Predictor on AWS.

---

## Architecture

```
infra/terraform/
├── modules/
│   ├── s3/             Versioned buckets (data + artifacts), lifecycle rules,
│   │                   optional cross-region replication
│   ├── iam/            Least-privilege roles: SageMaker execution,
│   │                   GitHub Actions OIDC (no stored keys), Prefect worker
│   └── sagemaker/      Model Package Group, inference endpoint,
│                       data capture, auto-scaling, CloudWatch alarms
└── environments/
    ├── dev/            ml.m5.large, single-AZ, 90-day retention
    └── prod/           ml.m5.4xlarge, multi-AZ, 365-day, cross-region replication,
                        auto-scaling 1–4 instances, data capture enabled
```

---

## Prerequisites

- [Terraform](https://developer.hashicorp.com/terraform/downloads) >= 1.7
- AWS CLI configured with credentials that have `AdministratorAccess`
  (for initial bootstrap only — narrow permissions thereafter)
- An existing S3 bucket for Terraform remote state (created manually once)
- A DynamoDB table for state locking (created manually once)

---

## One-Time Bootstrap

Before using the remote backend, you need to create the state bucket and
lock table. Run this once per AWS account:

```bash
# State bucket (versioning + encryption enabled)
aws s3api create-bucket \
  --bucket sc-predictor-tfstate-dev \
  --region us-east-1

aws s3api put-bucket-versioning \
  --bucket sc-predictor-tfstate-dev \
  --versioning-configuration Status=Enabled

aws s3api put-bucket-encryption \
  --bucket sc-predictor-tfstate-dev \
  --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'

# DynamoDB lock table
aws dynamodb create-table \
  --table-name sc-predictor-tfstate-lock \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1
```

Then update the `backend "s3"` block in `environments/dev/main.tf` with
your actual bucket name.

---

## Deploy Dev Environment

```bash
cd infra/terraform/environments/dev

# Initialise — downloads providers and configures remote backend
terraform init

# Preview changes — always review before applying
terraform plan -out=tfplan

# Apply
terraform apply tfplan
```

Or use the Makefile shortcuts:

```bash
make tf-init TF_ENV=dev
make tf-plan TF_ENV=dev
make tf-apply TF_ENV=dev
```

---

## Deploy Prod Environment

```bash
make tf-init TF_ENV=prod
make tf-plan TF_ENV=prod
make tf-apply TF_ENV=prod
```

The prod environment additionally provisions:
- Cross-region S3 replication (us-east-1 → us-west-2)
- SageMaker endpoint data capture (every prediction logged to S3)
- Auto-scaling policy (1–4 instances, 1,000 invocations/instance/minute)
- CloudWatch alarms (p99 latency > 500ms, 5xx errors > 5/minute)

---

## Resource Inventory

After `terraform apply`, the following resources are created in dev:

| Resource | Name pattern |
|---|---|
| S3 bucket (artifacts) | `sc-predictor-artifacts-dev` |
| S3 bucket (data) | `sc-predictor-data-dev` |
| IAM role (SageMaker) | `sc-predictor-dev-sagemaker-execution` |
| IAM role (CI/CD OIDC) | `sc-predictor-dev-cicd-deploy` |
| IAM role (Prefect) | `sc-predictor-dev-prefect-worker` |
| SageMaker Model Package Group | `sc-predictor-dev-models` |
| SageMaker endpoint | `sc-predictor-dev-endpoint` |
| CloudWatch log group | `/aws/sagemaker/Endpoints/sc-predictor-dev-endpoint` |
| CloudWatch alarm | `sc-predictor-dev-endpoint-high-latency` |
| CloudWatch alarm | `sc-predictor-dev-endpoint-errors` |

---

## CI/CD OIDC Setup

The `iam` module creates an OIDC trust relationship between GitHub Actions
and AWS. No `AWS_ACCESS_KEY_ID` is ever stored in GitHub Secrets.

After `terraform apply`, add these secrets to your GitHub repository:

| Secret | Value |
|---|---|
| `AWS_CICD_ROLE_ARN` | Output of `terraform output cicd_deploy_role_arn` |
| `ARTIFACTS_BUCKET` | Output of `terraform output artifacts_bucket_name` |

GitHub Actions workflows then use:
```yaml
- uses: aws-actions/configure-aws-credentials@v4
  with:
    role-to-assume: ${{ secrets.AWS_CICD_ROLE_ARN }}
    aws-region: us-east-1
```

---

## SageMaker Model Packaging

Before the endpoint can serve predictions, package the trained model:

```bash
# After running the training pipeline
mkdir -p /tmp/sm_model
cp models/best_model_top15.pkl /tmp/sm_model/
cp models/top_features.json /tmp/sm_model/
cp src/api/sagemaker/inference.py /tmp/sm_model/
tar -czf models/best_model_top15.tar.gz -C /tmp/sm_model .

# Upload to S3
aws s3 cp models/best_model_top15.tar.gz \
  s3://sc-predictor-artifacts-dev/models/best_model_top15.tar.gz
```

The `deploy.yml` CI/CD workflow automates this on every merge to `main`.

---

## Updating the Endpoint

The SageMaker endpoint is updated via GitHub Actions `deploy.yml` on every
merge to `main`. The update uses a blue/green strategy:

1. Creates a new `EndpointConfig` pointing at the latest model artifact
2. Calls `update-endpoint` (zero-downtime — SageMaker keeps the old variant
   alive until the new one is healthy)
3. Waits for `endpoint-in-service`
4. Runs a smoke test against the live endpoint

To update manually:
```bash
aws sagemaker update-endpoint \
  --endpoint-name sc-predictor-dev-endpoint \
  --endpoint-config-name <new-config-name>
```

---

## Cost Estimates (Dev Environment)

| Resource | Approximate monthly cost |
|---|---|
| `ml.t2.medium` endpoint (1 instance, ~720 hrs) | ~$40 |
| S3 storage (< 10 GB) | < $1 |
| CloudWatch logs (< 1 GB) | < $1 |
| **Total (dev)** | **~$42/month** |

Stop the endpoint when not in use:
```bash
aws sagemaker delete-endpoint --endpoint-name sc-predictor-dev-endpoint
```

Prod with `ml.m5.xlarge` × 2 instances ≈ $280/month.

---

## Teardown

```bash
make tf-destroy TF_ENV=dev
```

> **Note**: S3 buckets are set to `force_destroy = true` in dev, so they
> will be emptied and deleted. In prod, `force_destroy = false` — you must
> empty the buckets manually first or Terraform will error.
