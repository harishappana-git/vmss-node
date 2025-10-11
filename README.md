# Azure VM Scale Set Terraform Project

This repository contains an opinionated Terraform implementation for deploying an Azure Virtual Machine Scale Set (VMSS) landing zone that mirrors how many enterprise teams structure infrastructure-as-code projects. It demonstrates:

- A modular Terraform layout with reusable components (networking, compute, monitoring).
- Remote state stored in Azure Storage with state locking.
- Secure secret storage in Azure Key Vault.
- Azure Monitor integration via Log Analytics.
- CI/CD automation with GitHub Actions for plan/apply workflows.

The solution provisions a resource group, a virtual network, network security controls, a VMSS behind a load balancer, diagnostic storage, and monitoring resources. The VMSS uses a custom script extension to install NGINX and publish a simple landing page for verification.

## Repository structure

```
├── infra/
│   ├── main.tf                # Root module orchestrating the deployment
│   ├── outputs.tf
│   ├── providers.tf
│   ├── variables.tf
│   ├── versions.tf            # Terraform and provider constraints
│   └── modules/
│       ├── compute/           # VMSS, load balancer, diagnostics
│       ├── monitoring/        # Log Analytics and action group
│       └── networking/        # VNet, subnets, NSGs
├── environments/
│   └── dev/
│       ├── backend.hcl        # Example backend configuration for remote state
│       └── terraform.tfvars   # Example variable file for dev environment
└── .github/workflows/
    └── terraform.yml          # CI/CD pipeline definition
```

## Prerequisites

- [Terraform](https://developer.hashicorp.com/terraform/downloads) v1.5+ locally.
- An Azure subscription with permissions to create resource groups, networking, compute, monitoring, and Key Vault resources.
- An Azure Storage account to host the Terraform remote state (can be bootstrapped via CLI or portal).
- Azure AD application/service principal for GitHub Actions with at least `Contributor` on the target subscription.
- GitHub repository secrets populated with the Azure service principal credentials (`ARM_CLIENT_ID`, `ARM_CLIENT_SECRET`, `ARM_SUBSCRIPTION_ID`, `ARM_TENANT_ID`).

## Remote state bootstrap

1. Create an Azure resource group, storage account, and blob container for the Terraform state. Example using Azure CLI:

   ```bash
   az group create --name rg-terraform-state --location eastus
   az storage account create \
     --name tfstate${RANDOM} \
     --resource-group rg-terraform-state \
     --location eastus \
     --sku Standard_LRS \
     --encryption-services blob
   az storage container create \
     --name tfstate \
     --account-name <storage-account-name> \
     --auth-mode login
   ```

2. Update `environments/dev/backend.hcl` with the actual storage account name and confirm the blob container exists:

   ```hcl
   resource_group_name  = "rg-terraform-state"
   storage_account_name = "tfstate123456"
   container_name       = "tfstate"
   key                  = "dev.terraform.tfstate"
   ```

   > :warning: Backend files should **never** be committed with live storage account keys. Using `backend.hcl` keeps sensitive data out of the configuration and enables environment-specific settings.

## Local development workflow

1. Authenticate with Azure using the CLI or environment variables for the same identity that owns the storage account and desired subscription.
2. Navigate to the `infra/` directory and initialize Terraform using the backend config:

   ```bash
   cd infra
   terraform init -backend-config=../environments/dev/backend.hcl
   ```

3. Format, validate, and generate a plan with the provided dev variables:

   ```bash
   terraform fmt -recursive
   terraform validate
   terraform plan -var-file=../environments/dev/terraform.tfvars
   ```

4. Apply when ready:

   ```bash
   terraform apply -var-file=../environments/dev/terraform.tfvars
   ```

5. After provisioning completes, Terraform outputs the resource group name, the VMSS public IP address, and the Key Vault URI. You can verify the deployment by browsing to `http://<public-ip>` to see the NGINX welcome page.

6. To destroy the environment:

   ```bash
   terraform destroy -var-file=../environments/dev/terraform.tfvars
   ```

## GitHub Actions CI/CD pipeline

The workflow in `.github/workflows/terraform.yml` implements a two-stage pipeline:

- **validate** (PRs & pushes): checks formatting, initializes the backend, and runs `terraform validate` and `terraform plan`. Plans run using federated identity with the Azure service principal configured via GitHub secrets.
- **apply** (main branch only): runs after `validate` passes and requires the `production` environment approval if enabled in the repository. It applies the configuration automatically using the same backend and variable file.

Because the workflow references `environments/dev/backend.hcl` and `environments/dev/terraform.tfvars`, each environment can have its own directory with tailored `backend.hcl` and `terraform.tfvars` files. This pattern mirrors many enterprise setups where each workspace or stage has isolated state and configuration.

### Required GitHub configuration

1. Configure an environment named `production` (or change the workflow) with required approvers if you want manual gates before apply.
2. Add the following secrets to the repository or organization:

   - `ARM_CLIENT_ID`
   - `ARM_CLIENT_SECRET`
   - `ARM_SUBSCRIPTION_ID`
   - `ARM_TENANT_ID`

3. If using OIDC-based federation instead of a client secret, replace the login mechanism with [`azure/login`](https://github.com/Azure/login) and update the workflow accordingly.

## Security & operational considerations

- **Key Vault access**: The Key Vault uses RBAC authorization. Grant the VMSS managed identity access to the vault if workloads need to retrieve secrets.
- **Secrets hygiene**: The sample `terraform.tfvars` includes a placeholder password. Replace it with a strong secret or prefer SSH keys. Store real secrets in a secure vault and load them via CI variables.
- **State protection**: Enable soft delete and immutable blob policies on the state storage account for recovery.
- **Tagging**: Extend the `tags` map to enforce organizational tagging standards.
- **Policy compliance**: Integrate with Azure Policy by assigning initiatives to the resource group or subscription and ensuring Terraform deployments pass policy checks.
- **Monitoring**: The monitoring module creates an action group shell. Configure email/webhook receivers and alert rules to meet SRE requirements.
- **Scaling**: Adjust `vmss_instance_count`, autoscale rules, and VM SKU to meet workload requirements. Add Autoscale settings via `azurerm_monitor_autoscale_setting` as needed.

## Testing

- `terraform fmt` ensures standard formatting.
- `terraform validate` confirms configuration syntax and provider compatibility.

Both commands are encapsulated in the GitHub Actions workflow and should be executed locally using the steps above before opening a pull request.

## Extending to additional environments

To spin up staging or production environments, duplicate the `environments/dev` folder (e.g., `environments/prod`) with its own `backend.hcl` and `terraform.tfvars`. Update the workflow to trigger on those files or use input parameters to select the environment dynamically.

## Troubleshooting

- **Authentication errors** during `terraform init` or `plan`: verify Azure CLI login or environment variables. For CI, ensure the service principal has `Storage Blob Data Contributor` on the state storage account and `Contributor` on the deployment subscription.
- **Plan/apply drift**: Use `terraform state pull` for inspection and consider enabling Terraform Cloud or Azure DevOps for enhanced collaboration features if desired.
- **Locked state**: If a prior run crashed, release the lock using `az storage blob lease break` with the state blob path.

## Cleanup

Destroying the Terraform-managed resources does not remove the remote state storage account. To avoid accidental loss, delete the state resources manually only after confirming no environments rely on them.
