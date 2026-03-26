# infra/terraform/modules/iam/main.tf
# ─────────────────────────────────────────────────────────────────────────────
# IAM module: least-privilege roles for every principal in the system.
#
# Roles created
# ─────────────
# sagemaker_execution_role  — assumed by SageMaker training jobs and endpoints.
#                             Grants: S3 read/write on specific prefixes,
#                             CloudWatch Logs write, ECR pull.
# cicd_deploy_role          — assumed by GitHub Actions OIDC to deploy
#                             (no long-lived AWS keys in CI secrets).
# prefect_task_role         — assumed by ECS Fargate tasks running Prefect workers.
# ─────────────────────────────────────────────────────────────────────────────

variable "project_name"         { type = string }
variable "environment"          { type = string }
variable "artifacts_bucket_arn" { type = string }
variable "data_bucket_arn"      { type = string }
variable "github_org"           { type = string; default = "your-github-org" }
variable "github_repo"          { type = string; default = "superconductivity-predictor" }

locals {
  name_prefix = "${var.project_name}-${var.environment}"
}

# ── SageMaker execution role ──────────────────────────────────────────────────

resource "aws_iam_role" "sagemaker_execution" {
  name = "${local.name_prefix}-sagemaker-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "sagemaker_s3" {
  name = "s3-access"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadRawData"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          var.data_bucket_arn,
          "${var.data_bucket_arn}/data/*"
        ]
      },
      {
        Sid    = "WriteArtifacts"
        Effect = "Allow"
        Action = ["s3:PutObject", "s3:GetObject", "s3:DeleteObject", "s3:ListBucket"]
        Resource = [
          var.artifacts_bucket_arn,
          "${var.artifacts_bucket_arn}/models/*",
          "${var.artifacts_bucket_arn}/experiments/*",
          "${var.artifacts_bucket_arn}/endpoint-captures/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_cloudwatch" {
  role       = aws_iam_role.sagemaker_execution.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
}

resource "aws_iam_role_policy_attachment" "sagemaker_ecr" {
  role       = aws_iam_role.sagemaker_execution.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

# ── GitHub Actions OIDC deploy role ──────────────────────────────────────────
# No long-lived secrets: GitHub's OIDC provider issues a short-lived JWT
# that STS exchanges for temporary AWS credentials.

data "aws_caller_identity" "current" {}

resource "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"

  client_id_list = ["sts.amazonaws.com"]

  # GitHub's OIDC thumbprint (stable)
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

resource "aws_iam_role" "cicd_deploy" {
  name = "${local.name_prefix}-cicd-deploy"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Federated = aws_iam_openid_connect_provider.github.arn
      }
      Action = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringLike = {
          "token.actions.githubusercontent.com:sub" = [
            "repo:${var.github_org}/${var.github_repo}:ref:refs/heads/main",
            "repo:${var.github_org}/${var.github_repo}:pull_request"
          ]
        }
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "cicd_deploy_s3" {
  name = "cicd-s3"
  role = aws_iam_role.cicd_deploy.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "PushArtifacts"
        Effect = "Allow"
        Action = ["s3:PutObject", "s3:GetObject", "s3:ListBucket"]
        Resource = [
          var.artifacts_bucket_arn,
          "${var.artifacts_bucket_arn}/*"
        ]
      },
      {
        Sid      = "UpdateEndpoint"
        Effect   = "Allow"
        Action   = [
          "sagemaker:UpdateEndpoint",
          "sagemaker:DescribeEndpoint",
          "sagemaker:CreateEndpointConfig",
          "sagemaker:DescribeEndpointConfig"
        ]
        Resource = "*"
      }
    ]
  })
}

# ── Prefect worker role (ECS Fargate) ─────────────────────────────────────────

resource "aws_iam_role" "prefect_worker" {
  name = "${local.name_prefix}-prefect-worker"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "prefect_worker_s3" {
  name = "prefect-s3"
  role = aws_iam_role.prefect_worker.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
      Resource = [
        var.data_bucket_arn, "${var.data_bucket_arn}/*",
        var.artifacts_bucket_arn, "${var.artifacts_bucket_arn}/*"
      ]
    }]
  })
}

resource "aws_iam_role_policy_attachment" "prefect_ecs" {
  role       = aws_iam_role.prefect_worker.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ── Outputs ───────────────────────────────────────────────────────────────────

output "sagemaker_execution_role_arn" { value = aws_iam_role.sagemaker_execution.arn }
output "cicd_deploy_role_arn"         { value = aws_iam_role.cicd_deploy.arn }
output "prefect_worker_role_arn"      { value = aws_iam_role.prefect_worker.arn }
