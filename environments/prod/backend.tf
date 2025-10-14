terraform {
  backend "azurerm" {
    resource_group_name  = "rg-tfstate-shared"
    storage_account_name = "sttfstategitops"
    container_name       = "tfstate-prod"
    key                  = "prod/terraform.tfstate"
  }
}
