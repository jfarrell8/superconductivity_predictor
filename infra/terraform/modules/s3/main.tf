# infra/terraform/modules/s3/main.tf
# ─────────────────────────────────────────────────────────────────────────────
# S3 module: two buckets (data + artifacts) with versioning, encryption,
# lifecycle rules, and optional cross-region replication.
#
# Security defaults:
#   • Block all public access
#   • SSE-S3 encryption at rest
#   • Versioning enabled (required for DVC and model rollback)
#   • Lifecycle rule: transition artifacts to Intelligent-Tiering after 30 days
# ─────────────────────────────────────────────────────────────────────────────

variable "project_name"            { type = string }
variable "environment"             { type = string }
variable "log_retention_days"      { type = number; default = 30 }
variable "artifact_retention_days" { type = number; default = 90 }
variable "enable_replication"      { type = bool;   default = false }
variable "replica_region"          { type = string; default = "us-west-2" }

locals {
  artifacts_bucket = "${var.project_name}-artifacts-${var.environment}"
  data_bucket      = "${var.project_name}-data-${var.environment}"
}

# ── Artifacts bucket ──────────────────────────────────────────────────────────
# Stores: trained models (.pkl), Optuna studies, feature lists, manifests

resource "aws_s3_bucket" "artifacts" {
  bucket        = local.artifacts_bucket
  force_destroy = var.environment == "dev"   # safe to destroy in dev only
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id     = "transition-to-intelligent-tiering"
    status = "Enabled"
    filter { prefix = "models/" }

    transition {
      days          = 30
      storage_class = "INTELLIGENT_TIERING"
    }
  }

  rule {
    id     = "expire-old-experiment-artifacts"
    status = "Enabled"
    filter { prefix = "experiments/" }

    expiration {
      days = var.artifact_retention_days
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

# ── Data bucket ───────────────────────────────────────────────────────────────
# Stores: raw CSVs, processed feature CSVs, DVC cache

resource "aws_s3_bucket" "data" {
  bucket        = local.data_bucket
  force_destroy = var.environment == "dev"
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ── Optional: cross-region replication (prod only) ───────────────────────────

resource "aws_s3_bucket_replication_configuration" "artifacts" {
  count  = var.enable_replication ? 1 : 0
  bucket = aws_s3_bucket.artifacts.id
  role   = aws_iam_role.replication[0].arn

  rule {
    id     = "replicate-models"
    status = "Enabled"
    filter { prefix = "models/" }

    destination {
      bucket        = "arn:aws:s3:::${local.artifacts_bucket}-replica"
      storage_class = "STANDARD_IA"
    }
  }
}

resource "aws_iam_role" "replication" {
  count = var.enable_replication ? 1 : 0
  name  = "${var.project_name}-s3-replication-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "s3.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "replication" {
  count = var.enable_replication ? 1 : 0
  role  = aws_iam_role.replication[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetReplicationConfiguration", "s3:ListBucket"]
        Resource = aws_s3_bucket.artifacts.arn
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObjectVersionForReplication", "s3:GetObjectVersionAcl"]
        Resource = "${aws_s3_bucket.artifacts.arn}/*"
      },
      {
        Effect   = "Allow"
        Action   = ["s3:ReplicateObject", "s3:ReplicateDelete"]
        Resource = "arn:aws:s3:::${local.artifacts_bucket}-replica/*"
      }
    ]
  })
}

# ── Outputs ───────────────────────────────────────────────────────────────────

output "artifacts_bucket_id"  { value = aws_s3_bucket.artifacts.id }
output "artifacts_bucket_arn" { value = aws_s3_bucket.artifacts.arn }
output "data_bucket_id"       { value = aws_s3_bucket.data.id }
output "data_bucket_arn"      { value = aws_s3_bucket.data.arn }
