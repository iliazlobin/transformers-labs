provider "azurerm" {
  features {}
  skip_provider_registration = true
  subscription_id            = "ab751f4a-e2ff-438f-ad7d-f936ae89369c" # iz24a-subscription
}

resource "azurerm_resource_group" "this" {
  name     = "machine-learning"
  location = "East US 2"
}

resource "azurerm_virtual_network" "this" {
  name                = "workspace-vnet"
  resource_group_name = azurerm_resource_group.this.name
  location            = azurerm_resource_group.this.location
  address_space       = ["10.0.0.0/16"]
}

resource "azurerm_network_security_group" "this" {
  name                = "workspace-nsg"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
}

resource "azurerm_subnet" "this" {
  name                 = "external"
  resource_group_name  = azurerm_resource_group.this.name
  virtual_network_name = azurerm_virtual_network.this.name
  address_prefixes     = ["10.0.0.0/24"]
}

resource "azurerm_public_ip" "this" {
  name                = "workspace-public-ip"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  allocation_method   = "Dynamic"
}

resource "azurerm_network_interface" "this" {
  name                = "workspace-nic"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name

  ip_configuration {
    name                          = "internal"
    subnet_id                     = azurerm_subnet.this.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.this.id
  }
}

# resource "azurerm_virtual_machine" "test" {
#   name                  = "workstation-test"
#   location              = azurerm_resource_group.this.location
#   resource_group_name   = azurerm_resource_group.this.name
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

resource "azurerm_virtual_machine" "d2s" {
  name                  = "workstation-d2s"
  location              = azurerm_resource_group.this.location
  resource_group_name   = azurerm_resource_group.this.name
  network_interface_ids = [azurerm_network_interface.this.id]
  vm_size               = "Standard_D2s_v3"

  delete_os_disk_on_termination    = true
  delete_data_disks_on_termination = true

  storage_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-confidential-vm-jammy"
    sku       = "22_04-lts-cvm"
    version   = "latest"
  }

  storage_os_disk {
    name              = "workstation-standard-disk"
    caching           = "ReadWrite"
    create_option     = "FromImage"
    managed_disk_type = "Standard_LRS"
  }

  os_profile {
    computer_name  = "workstation"
    admin_username = "izlobin"
  }

  os_profile_linux_config {
    disable_password_authentication = true
    ssh_keys {
      path     = "/home/izlobin/.ssh/authorized_keys"
      key_data = file("~/.ssh/id_rsa.pub")
    }
  }

  connection {
    type        = "ssh"
    user        = "izlobin"
    private_key = file("~/.ssh/id_rsa")
    host        = azurerm_public_ip.this.ip_address
  }

  provisioner "file" {
    source      = "~/back.tar.gz"
    destination = "~/back.tar.gz"
  }

  provisioner "file" {
    source      = "~/.ssh/id_rsa"
    destination = "~/.ssh/id_rsa"
  }

  provisioner "remote-exec" {
    inline = [
      "tar xvzf ~/back.tar.gz",
      "cat ~/.bashrc.iz >> ~/.bashrc",
      "ls -la",
    ]
  }
}

output "public_ip_address" {
  value = azurerm_public_ip.this.ip_address
}
