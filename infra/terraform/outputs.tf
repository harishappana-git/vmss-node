output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "oidc_provider_arn" {
  description = "OIDC provider ARN for IRSA"
  value       = module.eks.oidc_provider_arn
}

output "todo_table_name" {
  description = "DynamoDB table for todos"
  value       = aws_dynamodb_table.todos.name
}

output "todo_service_account_role_arn" {
  description = "IAM role ARN assumed by the Kubernetes service account"
  value       = module.todo_irsa.iam_role_arn
}
