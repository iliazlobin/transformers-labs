# https://learn.microsoft.com/en-us/azure/virtual-machines/windows/quick-create-terraform

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~>3.99"
    }
  }
}

provider "azurerm" {
  # https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/features-block
  features {
    virtual_machine {
      delete_os_disk_on_deletion = false
    }
  }
  skip_provider_registration = true
  subscription_id            = "7d3d96a4-fb3d-436b-a57a-b9df03298658" # azure-iz
}

# resource "azurerm_resource_group" "this" {
#   name     = "machine-learning"
#   location = "East US"
# }

data "azurerm_resource_group" "this" {
  name = "machine-learning"
}

resource "azurerm_virtual_network" "this" {
  name                = "workspace-vnet"
  resource_group_name = data.azurerm_resource_group.this.name
  location            = data.azurerm_resource_group.this.location
  address_space       = ["10.0.0.0/24"]
}

resource "azurerm_subnet" "this" {
  name                 = "external"
  resource_group_name  = data.azurerm_resource_group.this.name
  virtual_network_name = azurerm_virtual_network.this.name
  address_prefixes     = ["10.0.0.0/28"]
}

resource "azurerm_public_ip" "this" {
  name                = "workspace-public-ip"
  sku                 = "Standard"
  location            = data.azurerm_resource_group.this.location
  resource_group_name = data.azurerm_resource_group.this.name
  allocation_method   = "Static"
}

resource "azurerm_network_interface" "this" {
  name                = "workspace-nic"
  location            = data.azurerm_resource_group.this.location
  resource_group_name = data.azurerm_resource_group.this.name

  ip_configuration {
    name                          = "public"
    subnet_id                     = azurerm_subnet.this.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.this.id
  }
}

resource "azurerm_network_security_group" "this" {
  name                = "workstation-nsg"
  location            = data.azurerm_resource_group.this.location
  resource_group_name = data.azurerm_resource_group.this.name

  security_rule {
    name                       = "SSH"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
}

resource "azurerm_network_interface_security_group_association" "example" {
  network_interface_id      = azurerm_network_interface.this.id
  network_security_group_id = azurerm_network_security_group.this.id
}

# resource "azurerm_virtual_machine" "test" {
#   name                  = "workstation-test"
#   location              = data.azurerm_resource_group.this.location
#   resource_group_name   = data.azurerm_resource_group.this.name
#   network_interface_ids = [azurerm_network_interface.this.id]
#   vm_size               = "Standard_B1s"

#   delete_os_disk_on_termination    = true
#   delete_data_disks_on_termination = true

#   storage_image_reference {
#     publisher = "Canonical"
#     offer     = "0001-com-ubuntu-confidential-vm-jammy"
#     sku       = "22_04-lts-cvm"
#     version   = "latest"
#   }

#   storage_os_disk {
#     name              = "workstation-standard-disk"
#     caching           = "ReadWrite"
#     create_option     = "FromImage"
#     managed_disk_type = "Standard_LRS"
#   }

#   os_profile {
#     computer_name  = "workstation"
#     admin_username = "izlobin"
#   }

#   os_profile_linux_config {
#     disable_password_authentication = true
#     ssh_keys {
#       path     = "/home/izlobin/.ssh/authorized_keys"
#       key_data = file("~/.ssh/id_rsa.pub")
#     }
#   }
# }

# resource "azurerm_virtual_machine" "d2s" {
#   name                  = "workstation-d2s"
#   location              = data.azurerm_resource_group.this.location
#   resource_group_name   = data.azurerm_resource_group.this.name
#   network_interface_ids = [azurerm_network_interface.this.id]

#   vm_size = "Standard_D2s_v3"
#   # vm_size = "Standard_NC24ads_A100_v4" # A100

#   delete_os_disk_on_termination    = true
#   delete_data_disks_on_termination = true

#   storage_image_reference {
#     publisher = "Canonical"
#     offer     = "0001-com-ubuntu-confidential-vm-jammy"
#     sku       = "22_04-lts-cvm"
#     version   = "latest"
#   }

#   storage_os_disk {
#     name              = "workstation-standard-disk"
#     caching           = "ReadWrite"
#     create_option     = "FromImage"
#     managed_disk_type = "Standard_LRS"
#   }

#   os_profile {
#     computer_name  = "workstation"
#     admin_username = "izlobin"
#   }

#   os_profile_linux_config {
#     disable_password_authentication = true
#     ssh_keys {
#       path     = "~/.ssh/authorized_keys"
#       key_data = file("~/.ssh/id_rsa.pub")
#     }
#   }

#   connection {
#     type        = "ssh"
#     user        = "izlobin"
#     private_key = file("~/.ssh/id_rsa")
#     host        = azurerm_public_ip.this.ip_address
#   }

#   provisioner "file" {
#     source      = "~/back.tar.gz"
#     destination = "~/back.tar.gz"
#   }

#   provisioner "file" {
#     source      = "~/.ssh/id_rsa"
#     destination = "~/.ssh/id_rsa"
#   }

#   provisioner "remote-exec" {
#     inline = [
#       "tar xvzf ~/back.tar.gz",
#       "cat ~/.bashrc.iz >> ~/.bashrc",
#       "ls -la",
#     ]
#   }
# }

# https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/resources/linux_virtual_machine#diff_disk_settings
# https://azure.microsoft.com/en-us/pricing/calculator/
resource "azurerm_linux_virtual_machine" "workstation-nc24" {
  name                  = "workstation-nc24"
  location              = data.azurerm_resource_group.this.location
  resource_group_name   = data.azurerm_resource_group.this.name
  network_interface_ids = [azurerm_network_interface.this.id]

  size = "Standard_NC24ads_A100_v4" # A100

  computer_name  = "workstation"
  admin_username = "izlobin"

  admin_ssh_key {
    username   = "izlobin"
    public_key = file("~/.ssh/id_rsa.pub")
  }

  # https://learn.microsoft.com/en-us/azure/virtual-machines/disks-types#premium-ssd-v2
  os_disk {
    name                 = "workstation-nc24-os-disk"
    caching              = "ReadWrite"
    storage_account_type = "StandardSSD_LRS"
    # storage_account_type = "Premium_LRS"
    disk_size_gb = 30
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-confidential-vm-jammy"
    sku       = "22_04-lts-cvm"
    version   = "latest"
  }

  priority        = "Spot"
  max_bid_price   = "1.0"
  eviction_policy = "Deallocate"

  # patch_mode = "AutomaticByPlatform"
  # reboot_setting = "Never"

  # connection {
  #   type        = "ssh"
  #   user        = "izlobin"
  #   private_key = file("~/.ssh/id_rsa")
  #   host        = azurerm_public_ip.this.ip_address
  # }

  # provisioner "file" {
  #   source      = "~/back.tar.gz"
  #   destination = "~/back.tar.gz"
  # }

  # provisioner "file" {
  #   source      = "calcula"
  #   destination = "~/.ssh/id_rsa"
  # }

  # provisioner "remote-exec" {
  #   inline = [
  #     "tar xvzf ~/back.tar.gz",
  #     "cat ~/.bashrc.iz >> ~/.bashrc",
  #     "ls -la",
  #   ]
  # }
}

resource "azurerm_subnet_network_security_group_association" "example" {
  subnet_id                 = azurerm_subnet.this.id
  network_security_group_id = azurerm_network_security_group.this.id
}

resource "azurerm_managed_disk" "izlobin-home" {
  name                 = "izlobin-home"
  location             = data.azurerm_resource_group.this.location
  resource_group_name  = data.azurerm_resource_group.this.name
  storage_account_type = "StandardSSD_LRS"
  # storage_account_type = "Premium_LRS"
  create_option = "Empty"
  disk_size_gb  = 200
}

# resource "azurerm_snapshot" "example" {
#   name                = "izlobin-home-240418"
#   location            = data.azurerm_resource_group.example.location
#   resource_group_name = data.azurerm_resource_group.example.name
#   create_option       = "Copy"
#   source_uri          = azurerm_managed_disk.izlobin-home.id
# }

resource "azurerm_virtual_machine_data_disk_attachment" "izlobin-home" {
  managed_disk_id    = azurerm_managed_disk.izlobin-home.id
  virtual_machine_id = azurerm_linux_virtual_machine.workstation-nc24.id
  lun                = "10"
  caching            = "ReadWrite"
}

output "public_ip_address" {
  value = azurerm_public_ip.this.ip_address
}
