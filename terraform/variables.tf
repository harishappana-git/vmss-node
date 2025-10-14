variable "prefix" {
  description = "Prefix used for all resource names to keep them unique."
  type        = string
}

variable "resource_group_name" {
  description = "Name of the resource group that will contain the infrastructure."
  type        = string
}

variable "location" {
  description = "Azure region where the resources will be created."
  type        = string
  default     = "eastus"
}

variable "vnet_name" {
  description = "Name of the virtual network."
  type        = string
}

variable "vnet_address_space" {
  description = "Address space for the virtual network."
  type        = list(string)
  default     = ["10.10.0.0/16"]
}

variable "subnet_name" {
  description = "Name of the subnet to create inside the virtual network."
  type        = string
}

variable "subnet_address_prefix" {
  description = "Address prefix for the subnet."
  type        = string
}

variable "ssh_allowed_cidr" {
  description = "CIDR block allowed to access the VM via SSH."
  type        = string
  default     = "0.0.0.0/0"
}

variable "vm_size" {
  description = "Size (SKU) of the Linux VM."
  type        = string
  default     = "Standard_B2s"
}

variable "vm_admin_username" {
  description = "Admin username for the Linux VM."
  type        = string
  default     = "azureuser"
}

variable "vm_admin_password" {
  description = "Admin password for the Linux VM (must meet Azure complexity requirements)."
  type        = string
  sensitive   = true
}

variable "common_tags" {
  description = "Common tags to apply to all resources."
  type        = map(string)
  default = {
    environment = "production"
    managed-by  = "terraform"
  }
}
