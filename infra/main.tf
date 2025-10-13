locals {
  common_tags = merge(
    {
      environment = var.environment
      "managed-by" = "terraform"
    },
    var.tags,
  )
}

data "azurerm_client_config" "current" {}

resource "azurerm_resource_group" "main" {
  name     = var.resource_group_name
  location = var.location
  tags     = local.common_tags
}

module "networking" {
  source              = "./modules/networking"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  address_space       = var.address_space
  subnets             = var.subnets
  tags                = local.common_tags
}

resource "random_string" "sa" {
  length  = 6
  upper   = false
  special = false
}

resource "azurerm_storage_account" "diag" {
  name                     = "diag${random_string.sa.result}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  allow_nested_items_to_be_public = false
  min_tls_version                  = "TLS1_2"
  tags                              = local.common_tags
}

resource "azurerm_key_vault" "main" {
  name                          = "kv-${random_string.sa.result}"
  resource_group_name           = azurerm_resource_group.main.name
  location                      = azurerm_resource_group.main.location
  tenant_id                     = data.azurerm_client_config.current.tenant_id
  sku_name                      = "standard"
  purge_protection_enabled      = true
  soft_delete_retention_days    = 90
  enable_rbac_authorization     = true
  public_network_access_enabled = true
  tags                          = local.common_tags
}

module "compute" {
  source                  = "./modules/compute"
  resource_group_name     = azurerm_resource_group.main.name
  location                = azurerm_resource_group.main.location
  subnet_id               = module.networking.subnet_ids["vmss"]
  tags                    = local.common_tags
  vm_admin_username       = var.vm_admin_username
  vm_admin_password       = var.vm_admin_password
  vmss_instance_count     = var.vmss_instance_count
  sku_name                = var.sku_name
  diagnostics_storage_uri = azurerm_storage_account.diag.primary_blob_endpoint
}

resource "azurerm_key_vault_secret" "vm_admin_password" {
  name         = "vmAdminPassword"
  value        = var.vm_admin_password
  key_vault_id = azurerm_key_vault.main.id
  depends_on   = [azurerm_key_vault.main]
}
