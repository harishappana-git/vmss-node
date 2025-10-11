variable "environment" {
  type        = string
  description = "Deployment environment name (e.g. dev, staging, prod)."
}

variable "location" {
  type        = string
  description = "Azure region for all resources."
  default     = "eastus"
}

variable "resource_group_name" {
  type        = string
  description = "Name of the resource group to create."
}

variable "tags" {
  description = "Common resource tags."
  type        = map(string)
  default     = {}
}

variable "address_space" {
  description = "Address space for the virtual network."
  type        = list(string)
  default     = ["10.10.0.0/16"]
}

variable "subnets" {
  description = "Definition of subnets to create."
  type = map(object({
    address_prefixes = list(string)
    service_endpoints = optional(list(string), [])
  }))
}

variable "vm_admin_username" {
  description = "Admin username for virtual machines."
  type        = string
}

variable "vm_admin_password" {
  description = "Admin password for virtual machines (used for demos; prefer SSH keys in production)."
  type        = string
  sensitive   = true
}

variable "vmss_instance_count" {
  description = "Number of VM instances in the scale set."
  type        = number
  default     = 2
}

variable "sku_name" {
  description = "Azure VM SKU for the scale set instances."
  type        = string
  default     = "Standard_B2ms"
}
