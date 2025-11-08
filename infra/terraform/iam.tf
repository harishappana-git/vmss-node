data "aws_iam_policy_document" "todo_table" {
  statement {
    sid    = "TodoTableCrud"
    effect = "Allow"

    actions = [
      "dynamodb:PutItem",
      "dynamodb:GetItem",
      "dynamodb:UpdateItem",
      "dynamodb:DeleteItem",
      "dynamodb:Query",
      "dynamodb:Scan"
    ]

    resources = [aws_dynamodb_table.todos.arn]
  }
}

module "todo_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "5.34.0"

  name = "${local.name}-todo-irsa"

  attach_policy_json = true
  policy_json        = data.aws_iam_policy_document.todo_table.json

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["app/todo-api"]
    }
  }

  tags = local.tags
}
