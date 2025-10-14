output "resource_group_name" {
  description = "Resource group deployed for the environment."
  value       = azurerm_resource_group.this.name
}

output "virtual_network_id" {
  description = "ID of the virtual network."
  value       = module.network.virtual_network_id
}

output "subnet_id" {
  description = "ID of the subnet."
  value       = module.network.subnet_id
}

output "vm_id" {
  description = "ID of the virtual machine."
  value       = module.compute.vm_id
}

output "vm_private_ip" {
  description = "Private IP address of the virtual machine."
  value       = module.compute.private_ip_address
}

output "vm_public_ip" {
  description = "Public IP address of the virtual machine if created."
  value       = module.compute.public_ip_address
}
