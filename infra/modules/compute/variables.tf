variable "resource_group_name" {
  type = string
}

variable "location" {
  type = string
}

variable "subnet_id" {
  type = string
}

variable "vm_admin_username" {
  type = string
}

variable "vm_admin_password" {
  type      = string
  sensitive = true
}

variable "vmss_instance_count" {
  type = number
}

variable "sku_name" {
  type = string
}

variable "tags" {
  type = map(string)
}

variable "diagnostics_storage_uri" {
  type = string
}

