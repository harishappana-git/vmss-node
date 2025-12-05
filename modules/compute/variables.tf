variable "resource_group_name" {
  description = "Name of the resource group for the virtual machine."
  type        = string
}

variable "location" {
  description = "Azure region for the virtual machine."
  type        = string
}

variable "subnet_id" {
  description = "ID of the subnet where the NIC will be placed."
  type        = string
}

variable "vm_name" {
  description = "Name of the virtual machine."
  type        = string
}

variable "vm_size" {
  description = "Size of the virtual machine."
  type        = string
  default     = "Standard_B2s"
}

variable "admin_username" {
  description = "Admin username for the VM."
  type        = string
}

variable "ssh_public_key" {
  description = "SSH public key for the VM admin user."
  type        = string
}

variable "os_disk_size_gb" {
  description = "Size of the OS disk in GB."
  type        = number
  default     = 64
}

variable "source_image_reference" {
  description = "Image definition for the VM."
  type = object({
    publisher = string
    offer     = string
    sku       = string
    version   = string
  })
  default = {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts-gen2"
    version   = "latest"
  }
}

variable "tags" {
  description = "Tags to apply to compute resources."
  type        = map(string)
  default     = {}
}

variable "allowed_ssh_cidrs" {
  description = "List of CIDR blocks allowed to SSH into the VM. Leave empty to disallow public SSH."
  type        = list(string)
  default     = []
}

variable "enable_public_ip" {
  description = "Whether to create and associate a public IP address with the VM."
  type        = bool
  default     = false
}
