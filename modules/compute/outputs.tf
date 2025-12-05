output "vm_id" {
  description = "ID of the virtual machine."
  value       = azurerm_linux_virtual_machine.this.id
}

output "private_ip_address" {
  description = "Private IP address of the VM."
  value       = azurerm_network_interface.this.private_ip_address
}

output "public_ip_address" {
  description = "Public IP address of the VM, if created."
  value       = length(azurerm_public_ip.this) > 0 ? azurerm_public_ip.this[0].ip_address : null
}
