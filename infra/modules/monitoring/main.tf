resource "azurerm_log_analytics_workspace" "main" {
  name                = "law-${var.resource_group_name}"
  location            = var.location
  resource_group_name = var.resource_group_name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = var.tags
}

resource "azurerm_monitor_action_group" "alerts" {
  name                = "ag-${var.resource_group_name}"
  resource_group_name = var.resource_group_name
  short_name          = "ag${substr(var.resource_group_name, 0, 6)}"

  tags = var.tags
}
