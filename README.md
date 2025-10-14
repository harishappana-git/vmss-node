# VMSS Node Infrastructure

This repository now includes a Terraform configuration and GitHub Actions workflow to provision a production Azure virtual machine and virtual network using username/password authentication. The project is designed to run safely from the `main` branch with a remotely stored Terraform state file.

## Repository Map

- Infrastructure as code: [`terraform/`](terraform/README.md)
- CI/CD workflow: [`.github/workflows/terraform-production.yml`](.github/workflows/terraform-production.yml)
- Remote state bootstrap script: [`scripts/create-remote-state.sh`](scripts/create-remote-state.sh)

## Before You Merge

1. Complete the prerequisites and remote state configuration described in the [Terraform README](terraform/README.md).
2. Populate the required GitHub secrets so the pipeline can authenticate to Azure and configure the backend.
3. Test locally (if desired) with `terraform plan` to confirm your variable values and backend settings.

Once these steps are complete, open a pull request. The GitHub Actions workflow validates the configuration on every PR and automatically applies it after changes are merged to `main`.
