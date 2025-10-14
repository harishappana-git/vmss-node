# Production Terraform Deployment

This folder contains a minimal Terraform configuration that provisions a production-ready Azure virtual machine and its supporting virtual network. The configuration is designed to run from the `main` branch and stores its state remotely in an Azure Storage Account to avoid local state drift.

## Architecture Overview

The configuration creates the following Azure resources:

- A resource group to contain all production resources.
- A virtual network and subnet.
- A network security group that only allows SSH from a configurable CIDR.
- A public IP address and network interface.
- An Ubuntu Linux virtual machine that uses SSH key authentication.

Common production tags are applied to all resources by default.

## Prerequisites

Before running Terraform locally **or** merging to `main`, make sure the following prerequisites are complete:

1. **Install tooling**
   - [Terraform CLI](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) `>= 1.5.0`
   - [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli)
2. **Azure subscription access** with permission to create resource groups, networks, network security groups, public IPs and virtual machines.
3. **Service principal** with `Contributor` role on the target subscription. Capture the following values for GitHub secrets:
   - `ARM_CLIENT_ID`
   - `ARM_CLIENT_SECRET`
   - `ARM_TENANT_ID`
   - `ARM_SUBSCRIPTION_ID`
4. **Remote state storage account** dedicated to Terraform state (recommended production practice).
   - Create a resource group (e.g. `rg-tfstate`).
   - Create a storage account with Standard LRS replication (e.g. `sttfstateprod`).
   - Create a blob container (e.g. `tfstate`).
   - Decide on a unique key name for the state file (e.g. `production.terraform.tfstate`).
5. **GitHub Secrets** for CI/CD:
   - `ARM_CLIENT_ID`, `ARM_CLIENT_SECRET`, `ARM_TENANT_ID`, `ARM_SUBSCRIPTION_ID` (from the service principal).
   - `TF_BACKEND_RESOURCE_GROUP`
   - `TF_BACKEND_STORAGE_ACCOUNT`
   - `TF_BACKEND_CONTAINER`
   - `TF_BACKEND_STATE_KEY`
   - `TF_VAR_vm_admin_ssh_public_key` containing the SSH public key you want Terraform to use.

## Repository Files

| File | Purpose |
| --- | --- |
| `providers.tf` | Defines the AzureRM provider, version constraints, and the remote backend block (values supplied at runtime). |
| `main.tf` | Declares the resource group, networking, NSG rules, and Linux VM resources. |
| `variables.tf` | Lists configurable inputs. Sensitive values (such as the SSH key) should be passed through variables or secrets. |
| `outputs.tf` | Provides key outputs after `apply`, such as the VM public IP address. |
| `terraform.tfvars.example` | Example variable file to copy to `terraform.tfvars` for local usage. |
| `backend.hcl.example` | Example backend configuration for remote state. Provide the real values via a `backend.hcl` file that is **not** checked into git. |

## Initial Setup Steps

1. Copy the backend example and fill in the remote state details:
   ```bash
   cd terraform
   cp backend.hcl.example backend.hcl
   # edit backend.hcl with your storage account details
   ```
2. Copy the variable example and update values for production:
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   # edit terraform.tfvars
   ```
3. Authenticate with Azure locally:
   ```bash
   az login
   az account set --subscription "<subscription-id>"
   ```
4. Initialize Terraform using the backend configuration:
   ```bash
   terraform init -backend-config=backend.hcl
   ```
5. Review the execution plan and apply:
   ```bash
   terraform plan
   terraform apply
   ```

> **Important:** Keep the SSH public key out of source control. When running in GitHub Actions, the secret `TF_VAR_vm_admin_ssh_public_key` automatically feeds the sensitive variable.

## CI/CD with GitHub Actions

The workflow in `.github/workflows/terraform-production.yml` performs the following:

- Triggers on pull requests for validation (`terraform fmt -check`, `terraform validate`, and `terraform plan`).
- Triggers on pushes to `main` to run `terraform apply` for production.
- Logs into Azure using the service principal secrets.
- Generates a temporary `backend.hcl` file from the backend secrets.
- Uses the same Terraform configuration stored in this folder to ensure consistent infrastructure.

### Required GitHub Repository Secrets

| Secret | Description |
| --- | --- |
| `ARM_CLIENT_ID` | Service principal application (client) ID. |
| `ARM_CLIENT_SECRET` | Service principal client secret. |
| `ARM_TENANT_ID` | Azure tenant ID for the service principal. |
| `ARM_SUBSCRIPTION_ID` | Target subscription ID. |
| `TF_BACKEND_RESOURCE_GROUP` | Resource group that hosts the storage account for the remote state. |
| `TF_BACKEND_STORAGE_ACCOUNT` | Storage account name that stores the state file. |
| `TF_BACKEND_CONTAINER` | Blob container name. |
| `TF_BACKEND_STATE_KEY` | Name of the state file (key). |
| `TF_VAR_vm_admin_ssh_public_key` | SSH public key for the VM admin user. |

### Promotion Workflow

1. Open a pull request targeting `main`. The workflow will run Terraform validation and a `plan`, commenting on the PR with the results.
2. After approval, merge to `main`. The push trigger runs `terraform apply` automatically, using the remote state.
3. Monitor the workflow run in GitHub Actions to ensure the deployment succeeded.

## Updating the Infrastructure

1. Modify the Terraform files (`*.tf`) to reflect the desired infrastructure changes.
2. Commit the changes and open a pull request.
3. Review the plan output from the PR checks to ensure changes are expected.
4. Merge once satisfied—`apply` will run automatically on `main`.

## Destroying (If Ever Needed)

Production environments typically retain resources, but if you need to tear down the infrastructure:

```bash
terraform destroy -var-file=terraform.tfvars
```

> Always double-check you are targeting the correct state file and subscription before destroying resources.
