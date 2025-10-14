variable "resource_group_name" {
  description = "Name of the resource group where the virtual network will be created."
  type        = string
}

variable "location" {
  description = "Azure region for the virtual network."
  type        = string
}

variable "name" {
  description = "Name of the virtual network."
  type        = string
}

variable "address_space" {
  description = "List of address spaces for the virtual network."
  type        = list(string)
}

variable "subnet_name" {
  description = "Name of the primary subnet."
  type        = string
}

variable "subnet_prefix" {
  description = "Address prefix for the primary subnet."
  type        = string
}

variable "tags" {
  description = "Tags to apply to the virtual network resources."
  type        = map(string)
  default     = {}
}
