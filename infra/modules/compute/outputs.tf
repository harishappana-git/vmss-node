output "public_ip_address" {
  value = azurerm_public_ip.vmss.ip_address
}

output "scale_set_id" {
  value = azurerm_linux_virtual_machine_scale_set.vmss.id
}
