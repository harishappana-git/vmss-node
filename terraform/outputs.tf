output "resource_group_name" {
  description = "Resource group where the VM was provisioned."
  value       = azurerm_resource_group.vm.name
}

output "virtual_network_id" {
  description = "ID of the created virtual network."
  value       = azurerm_virtual_network.vm.id
}

output "vm_public_ip_address" {
  description = "Public IP address assigned to the Linux VM."
  value       = azurerm_public_ip.vm.ip_address
}
