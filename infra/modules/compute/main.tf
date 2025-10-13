resource "azurerm_public_ip" "vmss" {
  name                = "pip-vmss"
  resource_group_name = var.resource_group_name
  location            = var.location
  allocation_method   = "Static"
  sku                 = "Standard"
  tags                = var.tags
}

resource "azurerm_lb" "vmss" {
  name                = "lb-vmss"
  location            = var.location
  resource_group_name = var.resource_group_name
  sku                 = "Standard"
  frontend_ip_configuration {
    name                 = "public"
    public_ip_address_id = azurerm_public_ip.vmss.id
  }
  tags = var.tags
}

resource "azurerm_lb_backend_address_pool" "vmss" {
  name            = "bepool"
  loadbalancer_id = azurerm_lb.vmss.id
}

resource "azurerm_lb_probe" "http" {
  name                = "http-health"
  loadbalancer_id     = azurerm_lb.vmss.id
  protocol            = "Tcp"
  port                = 80
  interval_in_seconds = 15
  number_of_probes    = 2
}

resource "azurerm_lb_rule" "http" {
  name                           = "http"
  loadbalancer_id                = azurerm_lb.vmss.id
  protocol                       = "Tcp"
  frontend_port                  = 80
  backend_port                   = 80
  frontend_ip_configuration_name = "public"
  backend_address_pool_id        = azurerm_lb_backend_address_pool.vmss.id
  probe_id                       = azurerm_lb_probe.http.id
}

resource "azurerm_linux_virtual_machine_scale_set" "vmss" {
  name                = "vmss-app"
  location            = var.location
  resource_group_name = var.resource_group_name
  sku                 = var.sku_name
  instances           = var.vmss_instance_count
  admin_username      = var.vm_admin_username
  admin_password      = var.vm_admin_password
  upgrade_mode        = "Rolling"
  overprovision       = false

  network_interface {
    name    = "vmss-ni"
    primary = true

    ip_configuration {
      name                                   = "internal"
      primary                                = true
      subnet_id                              = var.subnet_id
      load_balancer_backend_address_pool_ids = [azurerm_lb_backend_address_pool.vmss.id]
    }
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-focal"
    sku       = "20_04-lts"
    version   = "latest"
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
  }

  boot_diagnostics {
    storage_account_uri = var.diagnostics_storage_uri
  }

  extension {
    name                 = "install-nginx"
    publisher            = "Microsoft.Azure.Extensions"
    type                 = "CustomScript"
    type_handler_version = "2.1"

    settings = jsonencode({
      commandToExecute = "#!/bin/bash\napt-get update -y\napt-get install -y nginx\nsystemctl enable nginx\n(cat <<'HTML' > /var/www/html/index.html\n<html><body><h1>Welcome from Terraform VMSS</h1></body></html>\nHTML\n)"
    })
  }

  identity {
    type = "SystemAssigned"
  }

  tags = var.tags
}

