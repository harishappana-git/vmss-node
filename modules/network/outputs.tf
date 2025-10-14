output "virtual_network_id" {
  description = "ID of the created virtual network."
  value       = azurerm_virtual_network.this.id
}

output "subnet_id" {
  description = "ID of the created subnet."
  value       = azurerm_subnet.this.id
}
