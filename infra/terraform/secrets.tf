resource "aws_kms_key" "secrets" {
  description             = "KMS key for ${local.name} application secrets"
  deletion_window_in_days = 10
  enable_key_rotation     = true

  tags = local.tags
}

resource "aws_secretsmanager_secret" "todo_app" {
  name        = "${local.name}/application"
  kms_key_id  = aws_kms_key.secrets.arn
  description = "Application runtime configuration for the todo API"

  tags = local.tags
}

resource "aws_secretsmanager_secret_version" "todo_app" {
  secret_id     = aws_secretsmanager_secret.todo_app.id
  secret_string = jsonencode({
    DATABASE_TABLE = aws_dynamodb_table.todos.name
  })
}
