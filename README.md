# VMSS Node Infrastructure

This repository now includes a Terraform configuration and GitHub Actions workflow to provision a production Azure virtual machine and virtual network.

- All infrastructure-as-code lives under [`terraform/`](terraform/README.md).
- CI/CD automation is defined in [`.github/workflows/terraform-production.yml`](.github/workflows/terraform-production.yml).

Before merging changes to `main`, review the [Terraform README](terraform/README.md) for prerequisites such as remote state configuration, required GitHub secrets, and deployment steps.
