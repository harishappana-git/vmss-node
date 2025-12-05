environment          = "dev"
location             = "eastus"
resource_group_name  = "rg-enterprise-dev"
vm_admin_username    = "devopsadmin"
vm_admin_password    = "ChangeMe123!"
vmss_instance_count  = 2
sku_name             = "Standard_B2s"
address_space        = ["10.20.0.0/16"]
subnets = {
  vmss = {
    address_prefixes = ["10.20.1.0/24"]
    service_endpoints = ["Microsoft.Storage"]
  }
  data = {
    address_prefixes = ["10.20.2.0/24"]
    service_endpoints = ["Microsoft.KeyVault"]
  }
}
tags = {
  project = "vmss-app"
  owner   = "iac-team"
}
