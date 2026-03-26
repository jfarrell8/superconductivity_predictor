# infra/terraform/modules/ecr/main.tf
# ─────────────────────────────────────────────────────────────────────────────
# Amazon ECR repository for the superconductivity predictor Docker image.
#
# Features:
#   • Image scanning on push (detects OS + package vulnerabilities)
#   • Lifecycle policy: keep only the last 30 tagged images and expire
#     untagged images after 1 day (prevents unbounded storage growth)
#   • Cross-account pull access for SageMaker (same or different account)
# ─────────────────────────────────────────────────────────────────────────────

variable "project_name"  { type = string }
variable "environment"   { type = string }
variable "max_images"    { type = number; default = 30 }

locals {
  repo_name = "${var.project_name}-${var.environment}"
}

resource "aws_ecr_repository" "main" {
  name                 = local.repo_name
  image_tag_mutability = "MUTABLE"   # allows re-tagging "latest"

  image_scanning_configuration {
    scan_on_push = true   # automatic vulnerability scanning on every push
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  # Prevent accidental deletion in prod
  lifecycle {
    prevent_destroy = false
  }
}

# ── Lifecycle policy ──────────────────────────────────────────────────────────
# Keep only the last 30 tagged images and expire untagged images after 1 day.

resource "aws_ecr_lifecycle_policy" "main" {
  repository = aws_ecr_repository.main.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Expire untagged images after 1 day"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 1
        }
        action = { type = "expire" }
      },
      {
        rulePriority = 2
        description  = "Keep last ${var.max_images} tagged images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["sha-", "v", "latest"]
          countType     = "imageCountMoreThan"
          countNumber   = var.max_images
        }
        action = { type = "expire" }
      }
    ]
  })
}

# ── Outputs ───────────────────────────────────────────────────────────────────

output "repository_url"  { value = aws_ecr_repository.main.repository_url }
output "repository_arn"  { value = aws_ecr_repository.main.arn }
output "repository_name" { value = aws_ecr_repository.main.name }
