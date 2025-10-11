output "virtual_network_id" {
  value = azurerm_virtual_network.main.id
}

output "subnet_ids" {
  value = { for k, subnet in azurerm_subnet.this : k => subnet.id }
}
