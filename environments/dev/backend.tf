terraform {
  backend "azurerm" {
    resource_group_name  = "rg-tfstate-shared"
    storage_account_name = "dhaappsstategitops"
    container_name       = "tfstate-dev"
    key                  = "dev/terraform.tfstate"
  }
}
