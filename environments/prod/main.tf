locals {
  common_tags = merge({
    environment = var.environment
    managed_by  = "terraform"
  }, var.tags)
}

resource "azurerm_resource_group" "this" {
  name     = var.resource_group_name
  location = var.location
  tags     = local.common_tags
}

module "network" {
  source              = "../../modules/network"
  resource_group_name = azurerm_resource_group.this.name
  location            = azurerm_resource_group.this.location
  name                = var.virtual_network_name
  address_space       = var.virtual_network_address_space
  subnet_name         = var.subnet_name
  subnet_prefix       = var.subnet_prefix
  tags                = local.common_tags
}

module "compute" {
  source              = "../../modules/compute"
  resource_group_name = azurerm_resource_group.this.name
  location            = azurerm_resource_group.this.location
  subnet_id           = module.network.subnet_id
  vm_name             = var.vm_name
  vm_size             = var.vm_size
  admin_username      = var.admin_username
  ssh_public_key      = var.ssh_public_key
  allowed_ssh_cidrs   = var.allowed_ssh_cidrs
  enable_public_ip    = var.enable_public_ip
  tags                = local.common_tags
}
