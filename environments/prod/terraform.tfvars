environment          = "prod"
location             = "eastus2"
resource_group_name  = "rg-enterprise-prod"
vm_admin_username    = "devopsadmin"
vm_admin_password    = "ChangeMe123!"
vmss_instance_count  = 3
sku_name             = "Standard_B2ms"
address_space        = ["10.30.0.0/16"]
subnets = {
  vmss = {
    address_prefixes = ["10.30.1.0/24"]
    service_endpoints = ["Microsoft.Storage"]
  }
  data = {
    address_prefixes = ["10.30.2.0/24"]
    service_endpoints = ["Microsoft.KeyVault"]
  }
}
tags = {
  project = "vmss-app"
  owner   = "iac-team"
  cost-center = "prod-shared"
}
