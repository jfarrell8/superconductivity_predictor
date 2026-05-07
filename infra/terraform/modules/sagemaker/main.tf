# infra/terraform/modules/sagemaker/main.tf

variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "execution_role_arn" {
  type = string
}

variable "artifacts_bucket_id" {
  type = string
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "training_instance_type" {
  type    = string
  default = "ml.m5.large"
}

variable "endpoint_instance_type" {
  type    = string
  default = "ml.t2.medium"
}

variable "endpoint_instance_count" {
  type    = number
  default = 1
}

variable "enable_data_capture" {
  type    = bool
  default = false
}

variable "data_capture_bucket" {
  type    = string
  default = ""
}

variable "data_capture_prefix" {
  type    = string
  default = "endpoint-captures"
}

variable "container_image" {
  type        = string
  default     = ""
  description = "Container image URI. Leave empty to use the AWS-managed sklearn image."
}

locals {
  name_prefix         = "${var.project_name}-${var.environment}"
  model_package_group = "${local.name_prefix}-models"

  sklearn_account_map = {
    "us-east-1"      = "683313688378"
    "us-east-2"      = "257758044811"
    "us-west-1"      = "746614075791"
    "us-west-2"      = "246618743249"
    "eu-west-1"      = "141502667606"
    "eu-central-1"   = "492215442770"
    "ap-southeast-1" = "627065512975"
    "ap-northeast-1" = "354813040037"
  }

  sklearn_account = lookup(local.sklearn_account_map, var.aws_region, "683313688378")

  resolved_container_image = var.container_image != "" ? var.container_image : "${local.sklearn_account}.dkr.ecr.${var.aws_region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
}

# ── CloudWatch log groups ─────────────────────────────────────────────────────

resource "aws_cloudwatch_log_group" "sagemaker_training" {
  name              = "/aws/sagemaker/TrainingJobs/${local.name_prefix}"
  retention_in_days = var.environment == "prod" ? 90 : 14
}

resource "aws_cloudwatch_log_group" "sagemaker_endpoint" {
  name              = "/aws/sagemaker/Endpoints/${local.name_prefix}-endpoint"
  retention_in_days = var.environment == "prod" ? 90 : 14
}

# ── Model Package Group ───────────────────────────────────────────────────────

resource "aws_sagemaker_model_package_group" "main" {
  model_package_group_name        = local.model_package_group
  model_package_group_description = "Superconductivity TC predictor models — ${var.environment}"
}

# ── SageMaker Model ───────────────────────────────────────────────────────────

resource "aws_sagemaker_model" "predictor" {
  name               = "${local.name_prefix}-predictor"
  execution_role_arn = var.execution_role_arn

  primary_container {
    image          = local.resolved_container_image
    model_data_url = "s3://${var.artifacts_bucket_id}/models/best_model_top15.tar.gz"

    environment = {
      SAGEMAKER_PROGRAM          = "inference.py"
      SAGEMAKER_SUBMIT_DIRECTORY = "s3://${var.artifacts_bucket_id}/models/code.tar.gz"
    }
  }

  lifecycle {
    create_before_destroy = true
  }
}

# ── Endpoint Configuration ────────────────────────────────────────────────────

resource "aws_sagemaker_endpoint_configuration" "main" {
  name = "${local.name_prefix}-endpoint-config"

  production_variants {
    variant_name           = "primary"
    model_name             = aws_sagemaker_model.predictor.name
    initial_instance_count = var.endpoint_instance_count
    instance_type          = var.endpoint_instance_type
    initial_variant_weight = 1.0
  }

  dynamic "data_capture_config" {
    for_each = var.enable_data_capture ? [1] : []
    content {
      enable_capture              = true
      initial_sampling_percentage = 100
      destination_s3_uri          = "s3://${var.data_capture_bucket}/${var.data_capture_prefix}"

      capture_options {
        capture_mode = "Input"
      }

      capture_options {
        capture_mode = "Output"
      }

      capture_content_type_header {
        json_content_types = ["application/json"]
      }
    }
  }

  lifecycle {
    create_before_destroy = true
  }
}

# ── Endpoint ──────────────────────────────────────────────────────────────────

resource "aws_sagemaker_endpoint" "main" {
  name                 = "${local.name_prefix}-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.main.name

  lifecycle {
    create_before_destroy = true
    ignore_changes        = [endpoint_config_name]
  }
}

# ── Auto Scaling (prod only) ──────────────────────────────────────────────────

resource "aws_appautoscaling_target" "endpoint" {
  count              = var.environment == "prod" ? 1 : 0
  max_capacity       = 4
  min_capacity       = 1
  resource_id        = "endpoint/${aws_sagemaker_endpoint.main.name}/variant/primary"
  scalable_dimension = "sagemaker:variant:DesiredInstanceCount"
  service_namespace  = "sagemaker"
}

resource "aws_appautoscaling_policy" "endpoint_scaling" {
  count              = var.environment == "prod" ? 1 : 0
  name               = "${local.name_prefix}-invocations-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.endpoint[0].resource_id
  scalable_dimension = aws_appautoscaling_target.endpoint[0].scalable_dimension
  service_namespace  = aws_appautoscaling_target.endpoint[0].service_namespace

  target_tracking_scaling_policy_configuration {
    target_value       = 1000
    scale_in_cooldown  = 300
    scale_out_cooldown = 60

    predefined_metric_specification {
      predefined_metric_type = "SageMakerVariantInvocationsPerInstance"
    }
  }
}

# ── CloudWatch Alarms ─────────────────────────────────────────────────────────

resource "aws_cloudwatch_metric_alarm" "endpoint_latency" {
  alarm_name          = "${local.name_prefix}-endpoint-high-latency"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "ModelLatency"
  namespace           = "AWS/SageMaker"
  period              = 60
  extended_statistic  = "p99"
  threshold           = 500
  alarm_description   = "Endpoint p99 latency exceeded 500ms"
  treat_missing_data  = "notBreaching"

  dimensions = {
    EndpointName = aws_sagemaker_endpoint.main.name
    VariantName  = "primary"
  }
}

resource "aws_cloudwatch_metric_alarm" "endpoint_errors" {
  alarm_name          = "${local.name_prefix}-endpoint-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "Invocation5XXErrors"
  namespace           = "AWS/SageMaker"
  period              = 60
  statistic           = "Sum"
  threshold           = 5
  alarm_description   = "More than 5 5xx errors in 1 minute on the inference endpoint"
  treat_missing_data  = "notBreaching"

  dimensions = {
    EndpointName = aws_sagemaker_endpoint.main.name
  }
}

# ── Outputs ───────────────────────────────────────────────────────────────────

output "endpoint_name" {
  value = aws_sagemaker_endpoint.main.name
}

output "model_package_group" {
  value = aws_sagemaker_model_package_group.main.model_package_group_name
}

output "endpoint_config_name" {
  value = aws_sagemaker_endpoint_configuration.main.name
}
