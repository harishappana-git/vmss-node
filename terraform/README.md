# Production Terraform Deployment

This folder contains a minimal Terraform configuration that provisions a production-ready Azure virtual machine and its supporting virtual network. The configuration is designed to run from the `main` branch and stores its state remotely in an Azure Storage Account to avoid local state drift.

## Architecture Overview

The configuration creates the following Azure resources:

- A resource group to contain all production resources.
- A virtual network and subnet.
- A network security group that only allows SSH from a configurable CIDR.
- A public IP address and network interface.
- An Ubuntu Linux virtual machine that uses local username/password authentication over SSH.

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
   ```bash
   az ad sp create-for-rbac \
     --name "terraform-production-ci" \
     --role Contributor \
     --scopes "/subscriptions/<subscription-id>"
   ```
   The command returns `appId`, `password`, and `tenant`. Map them directly to the `ARM_CLIENT_ID`, `ARM_CLIENT_SECRET`, and `ARM_TENANT_ID` secrets. The GitHub Actions workflow signs in with this client secret, so no federated identity configuration is required.
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
   - `TF_VAR_vm_admin_password` storing a strong password that satisfies [Azure's complexity requirements](https://learn.microsoft.com/azure/virtual-machines/linux/faq#are-there-any-requirements-for-user-names-or-passwords-when-creating-a-vm-).

## Bootstrap the Remote State Backend

To streamline creation of the storage account that hosts Terraform state, run the helper script from the repository root **after** signing in with the Azure CLI:

```bash
az login
./scripts/create-remote-state.sh \
  <resource-group-name> \
  <azure-region> \
  <storage-account-name> \
  <container-name>
```

The script is idempotent and will create or update:

1. The resource group.
2. A Standard LRS storage account.
3. A blob container for the Terraform state file.

After the script runs, record the values and add them to the GitHub secrets listed above. Use the same values inside `backend.hcl` for local runs.

## Repository Files

| File | Purpose |
| --- | --- |
| `providers.tf` | Defines the AzureRM provider, version constraints, and the remote backend block (values supplied at runtime). |
| `main.tf` | Declares the resource group, networking, NSG rules, and Linux VM resources. |
| `variables.tf` | Lists configurable inputs. Sensitive values (such as the admin password) should be passed through variables or secrets. |
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
2. Copy the variable example and update values for production (do **not** commit the resulting file):
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   # edit terraform.tfvars
   ```
   Ensure `vm_admin_password` is a strong, unique password.
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

> **Important:** Never commit real passwords. When running in GitHub Actions, the secret `TF_VAR_vm_admin_password` automatically feeds the sensitive variable at runtime.

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
| `TF_VAR_vm_admin_password` | Strong password for the VM admin user. |

### End-to-End Pipeline Flow

1. **Feature branch work** – Update Terraform files and push to a feature branch.
2. **Open a pull request** – Target `main`. The workflow performs:
   - `terraform fmt -check`
   - `terraform validate`
   - `terraform plan`
   The plan output appears in the workflow logs for reviewer approval.
3. **Merge to `main`** – On merge, the workflow reuses the generated backend file and runs `terraform apply -auto-approve` with the production variables supplied from GitHub secrets.
4. **Monitor the run** – Confirm the apply succeeded in the Actions tab and optionally verify the new/updated resources in the Azure Portal.

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
