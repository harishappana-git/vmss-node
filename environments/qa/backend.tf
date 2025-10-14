terraform {
  backend "azurerm" {
    resource_group_name  = "rg-tfstate-shared"
    storage_account_name = "dhaappsstategitops"
    container_name       = "tfstate-qa"
    key                  = "qa/terraform.tfstate"
  }
}
