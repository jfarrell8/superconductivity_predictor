# infra/terraform/environments/dev/main.tf
# ─────────────────────────────────────────────────────────────────────────────
# Dev environment: single-AZ, smaller instance types, shorter retention.
# Apply: cd infra/terraform/environments/dev && terraform init && terraform apply
# ─────────────────────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.40"
    }
  }

  # Remote state — swap bucket/key for your account
  backend "s3" {
    bucket         = "sc-predictor-tfstate-dev"
    key            = "superconductivity/dev/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "sc-predictor-tfstate-lock"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "superconductivity-predictor"
      Environment = "dev"
      ManagedBy   = "terraform"
    }
  }
}

# ── Variables ─────────────────────────────────────────────────────────────────

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "project_name" {
  type    = string
  default = "sc-predictor"
}

# ── ECR module ────────────────────────────────────────────────────────────────

module "ecr" {
  source       = "../../modules/ecr"
  project_name = var.project_name
  environment  = "dev"
  max_images   = 10   # dev: keep fewer images to save storage costs
}

# ── S3 module ─────────────────────────────────────────────────────────────────

module "s3" {
  source       = "../../modules/s3"
  project_name = var.project_name
  environment  = "dev"
  # Dev: shorter retention, no replication
  log_retention_days      = 30
  artifact_retention_days = 90
  enable_replication      = false
}

# ── IAM module ────────────────────────────────────────────────────────────────

module "iam" {
  source          = "../../modules/iam"
  project_name    = var.project_name
  environment     = "dev"
  artifacts_bucket_arn = module.s3.artifacts_bucket_arn
  data_bucket_arn      = module.s3.data_bucket_arn
}

# ── SageMaker module ──────────────────────────────────────────────────────────

module "sagemaker" {
  source               = "../../modules/sagemaker"
  project_name         = var.project_name
  environment          = "dev"
  execution_role_arn   = module.iam.sagemaker_execution_role_arn
  artifacts_bucket_id  = module.s3.artifacts_bucket_id
  aws_region           = var.aws_region
  # Dev: smaller instance, single AZ
  training_instance_type  = "ml.m5.large"
  endpoint_instance_type  = "ml.t2.medium"
  endpoint_instance_count = 1
  enable_data_capture     = false
  # Uncomment to use the custom ECR image instead of the built-in sklearn container:
  # container_image = "${module.ecr.repository_url}:latest"
}

# ── Outputs ───────────────────────────────────────────────────────────────────

output "ecr_repository_url" {
  value       = module.ecr.repository_url
  description = "ECR repository URL for Docker image pushes"
}

output "artifacts_bucket_name" {
  value       = module.s3.artifacts_bucket_id
  description = "S3 bucket for ML artifacts"
}

output "sagemaker_endpoint_name" {
  value       = module.sagemaker.endpoint_name
  description = "SageMaker inference endpoint name"
}

output "training_role_arn" {
  value       = module.iam.sagemaker_execution_role_arn
  description = "IAM role ARN for SageMaker training jobs"
}
