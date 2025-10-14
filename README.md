# VMSS Node Infrastructure

This repository delivers an end-to-end Terraform project and GitHub Actions pipeline that deploys a production Azure virtual machine and virtual network using username/password authentication. Terraform state is stored remotely in Azure Storage and all changes are applied from the `main` branch via CI/CD.

Use this README as the quick-start checklist. The [Terraform documentation](terraform/README.md) contains deeper architectural details, variable references, and troubleshooting guidance.

## Repository Map

- Infrastructure as code: [`terraform/`](terraform/README.md)
- CI/CD workflow: [`.github/workflows/terraform-production.yml`](.github/workflows/terraform-production.yml)
- Remote state bootstrap script: [`scripts/create-remote-state.sh`](scripts/create-remote-state.sh)

## Prerequisites

Before changing infrastructure or merging to `main`, confirm the following are complete:

1. **Tooling installed**
   - Terraform CLI `>= 1.5.0`
   - Azure CLI
2. **Azure access**
   - Subscription with permissions to create resource groups, networks, network security groups, public IPs, and VMs.
   - Service principal with at least `Contributor` rights on the subscription. Generate an Azure credential JSON (using the `--sdk-auth` flag) that will be stored as a GitHub secret:
     ```bash
     az ad sp create-for-rbac \
       --name "terraform-production-ci" \
       --role Contributor \
       --scopes "/subscriptions/<subscription-id>" \
       --sdk-auth
     ```
     Copy the command output and save it as the GitHub secret `AZURE_CREDENTIALS`. The JSON includes the client ID, client secret, tenant ID, and subscription ID. The workflow uses that secret directly with `azure/login`, so no federated credential configuration is required.
3. **Remote Terraform state storage**
   - Either provision manually or run the helper script:
     ```bash
     az login
     ./scripts/create-remote-state.sh \
       <resource-group-name> \
       <azure-region> \
       <storage-account-name> \
       <container-name>
     ```
   - Capture the resulting resource group, storage account, container, and desired state key.
4. **GitHub repository secrets** (required before the pipeline will succeed)
   - `AZURE_CREDENTIALS` – The full JSON returned by `az ad sp create-for-rbac ... --sdk-auth`.
   - `TF_BACKEND_RESOURCE_GROUP`
   - `TF_BACKEND_STORAGE_ACCOUNT`
   - `TF_BACKEND_CONTAINER`
   - `TF_BACKEND_STATE_KEY`
   - `TF_VAR_vm_admin_password` (strong password that satisfies Azure requirements)

   To create a secret in GitHub, navigate to **Repository → Settings → Secrets and variables → Actions → New repository secret**. Paste the entire JSON document into the value field when creating `AZURE_CREDENTIALS`.

## Configure the Terraform Project

Complete these steps **in order** from the repository root to prepare for a plan/apply—locally or in CI:

1. Copy the backend configuration template:
   ```bash
   cd terraform
   cp backend.hcl.example backend.hcl
   # edit backend.hcl with the resource group, storage account, container, and key recorded above
   ```
2. Copy and customize the variable file (do not commit the resulting file):
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   # edit terraform.tfvars with your production values (including vm_admin_password if testing locally)
   ```
3. Authenticate with Azure when running locally:
   ```bash
   az login
   az account set --subscription "<subscription-id>"
   ```
4. Initialize Terraform to download providers and connect to the backend:
   ```bash
   terraform init -backend-config=backend.hcl
   ```
5. (Optional) Validate and preview changes before opening a PR:
   ```bash
   terraform fmt
   terraform validate
   terraform plan
   ```

Return to the repository root (`cd ..`) after configuring files if you plan to work with git commands.

## Triggering the GitHub Actions Pipeline

The workflow automatically validates and applies Terraform. Use this sequence for end-to-end deployments:

1. **Create or update a feature branch**
   ```bash
   git checkout -b feature/my-change
   # edit Terraform files
   git add .
   git commit -m "Describe your change"
   git push origin feature/my-change
   ```
2. **Open a pull request targeting `main`**
   - The workflow runs `terraform fmt -check`, `terraform validate`, and `terraform plan`.
   - Review the plan output in the Actions logs to confirm expected changes.
3. **Merge the pull request once approved**
   - A push to `main` triggers `terraform apply -auto-approve` using the secrets-configured backend and variables.
   - Monitor the workflow run in GitHub Actions for success and verify resources in Azure as needed.

If you ever need to rerun the pipeline (for example after updating secrets), use the **Re-run jobs** button in the Actions tab on the relevant workflow run.

## Before You Merge

Run through this final checklist:

1. Prerequisites are satisfied and repository secrets are in place.
2. Local configuration files (`backend.hcl`, `terraform.tfvars`) are updated and excluded from git.
3. Optional local `terraform plan` outputs match expectations.

After the checklist, proceed with the PR workflow above. Every merge to `main` results in a fresh, automated Terraform apply.
