output "resource_group_name" {
  value       = azurerm_resource_group.main.name
  description = "Resource group that contains the deployment."
}

output "vmss_public_ip" {
  value       = azurerm_public_ip.vmss_pip.ip_address
  description = "Public IP address for the VM scale set load balancer."
}

output "key_vault_uri" {
  value       = azurerm_key_vault.main.vault_uri
  description = "URI of the Key Vault created for secrets management."
}
