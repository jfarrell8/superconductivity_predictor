# infra/terraform/environments/prod/main.tf
# ─────────────────────────────────────────────────────────────────────────────
# Production environment: multi-AZ, larger instances, longer retention,
# S3 cross-region replication, data capture enabled on the endpoint.
# ─────────────────────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.40"
    }
  }

  backend "s3" {
    bucket         = "sc-predictor-tfstate-prod"
    key            = "superconductivity/prod/terraform.tfstate"
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
      Environment = "prod"
      ManagedBy   = "terraform"
    }
  }
}

# Secondary provider for replication destination
provider "aws" {
  alias  = "replica"
  region = var.replica_region
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "replica_region" {
  type    = string
  default = "us-west-2"
}

variable "project_name" {
  type    = string
  default = "sc-predictor"
}

module "ecr" {
  source       = "../../modules/ecr"
  project_name = var.project_name
  environment  = "prod"
  max_images   = 30
}

module "s3" {
  source       = "../../modules/s3"
  project_name = var.project_name
  environment  = "prod"
  # Prod: longer retention, cross-region replication enabled
  log_retention_days      = 90
  artifact_retention_days = 365
  enable_replication      = true
  replica_region          = var.replica_region
}

module "iam" {
  source               = "../../modules/iam"
  project_name         = var.project_name
  environment          = "prod"
  artifacts_bucket_arn = module.s3.artifacts_bucket_arn
  data_bucket_arn      = module.s3.data_bucket_arn
}

module "sagemaker" {
  source               = "../../modules/sagemaker"
  project_name         = var.project_name
  environment          = "prod"
  execution_role_arn   = module.iam.sagemaker_execution_role_arn
  artifacts_bucket_id  = module.s3.artifacts_bucket_id
  aws_region           = var.aws_region
  # Prod: GPU-capable training, multi-instance endpoint, data capture on
  training_instance_type  = "ml.m5.4xlarge"
  endpoint_instance_type  = "ml.m5.xlarge"
  endpoint_instance_count = 2
  enable_data_capture     = true
  data_capture_bucket     = module.s3.artifacts_bucket_id
  data_capture_prefix     = "endpoint-captures"
  # Use the custom ECR image in prod for full dependency control
  container_image         = "${module.ecr.repository_url}:latest"
}

output "ecr_repository_url" {
  value = module.ecr.repository_url
}

output "artifacts_bucket_name" {
  value = module.s3.artifacts_bucket_id
}

output "sagemaker_endpoint_name" {
  value = module.sagemaker.endpoint_name
}

output "training_role_arn" {
  value = module.iam.sagemaker_execution_role_arn
}
