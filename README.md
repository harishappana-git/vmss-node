# Terraform GitOps Blueprint for Azure

This repository implements a GitOps-friendly Terraform workflow for deploying a minimal Azure estate (resource group, virtual network, subnet, and Linux virtual machine) across **development**, **QA**, and **production** environments. The design emphasises repeatability, auditability, security, and collaboration by making Git the source of truth and automating validation, planning, and application through GitHub Actions.

## Repository structure

```
├── modules
│   ├── compute            # Provision the VM, NIC, NSG, and optional public IP
│   └── network            # Provision the VNet and primary subnet
├── environments
│   ├── dev                # Development environment definitions
│   ├── qa                 # QA / staging environment definitions
│   └── prod               # Production environment definitions
├── scripts
│   └── create-service-principal.sh
└── .github
    └── workflows
        └── terraform-prod.yml
```

Each environment folder is a standalone Terraform root module with its own backend configuration, variables, and outputs. Shared logic lives in reusable modules.

## Azure authentication

Authenticate Terraform via an Azure AD service principal stored as GitHub secrets. Run the following command (replace the placeholders) to create a principal with subscription-scoped **Contributor** rights:

```bash
az ad sp create-for-rbac \
  --name "gitops-terraform" \
  --role "Contributor" \
  --scopes "/subscriptions/<SUBSCRIPTION_ID>" \
  --sdk-auth
```

The command prints a JSON payload containing `clientId`, `clientSecret`, `subscriptionId`, and `tenantId`. Store this JSON in the repository secret `AZURE_CREDENTIALS` and share the principal with collaborators as needed. To also retrieve the individual values (for local CLI use), you can output them separately:

```bash
az ad sp create-for-rbac \
  --name "gitops-terraform" \
  --role "Contributor" \
  --scopes "/subscriptions/<SUBSCRIPTION_ID>" \
  --query "{clientId:appId, clientSecret:password, tenantId:tenant, subscriptionId:id}" \
  --output tsv
```

> **Tip:** Limit access by assigning the principal to dedicated resource groups once they exist. Use Azure role assignments to grant collaborators the least privilege required.

## Remote state storage

All environments share a single Azure Storage account that stores the Terraform state files and provides state locking via blob leases. Create the storage account and per-environment containers once:

```bash
RESOURCE_GROUP="rg-tfstate-shared"
LOCATION="eastus"
STORAGE_ACCOUNT="sttfstategitops"  # must be globally unique

az group create --name "$RESOURCE_GROUP" --location "$LOCATION"
az storage account create \
  --name "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --sku Standard_LRS \
  --kind StorageV2 \
  --min-tls-version TLS1_2

for ENV in dev qa prod; do
  az storage container create \
    --name "tfstate-${ENV}" \
    --account-name "$STORAGE_ACCOUNT" \
    --auth-mode login
done
```

The backend for each environment is configured in `backend.tf` and points to the corresponding container and key.

## Working with environments

1. Copy the sample variables file and adjust per environment:
   ```bash
   cp environments/prod/terraform.tfvars.example environments/prod/terraform.tfvars
   ```
   Populate `ssh_public_key`, CIDR allow lists, and naming conventions as required. **Never** commit `.tfvars` files (they are ignored by `.gitignore`).
2. Initialise Terraform for the target environment:
   ```bash
   cd environments/prod
   terraform init
   terraform plan
   ```
3. Apply changes locally only when necessary; prefer going through pull requests.

Each environment uses a unique resource group (`rg-gitops-dev`, `rg-gitops-qa`, `rg-gitops-prod` in the examples) but can share the same Azure subscription.

## GitOps workflow

1. Create a feature branch and implement infrastructure changes (modules or environment files).
2. Open a pull request into `main`. The GitHub Actions workflow automatically runs `terraform fmt`, `terraform init`, `terraform validate`, and `terraform plan` for the production environment.
3. Reviewers inspect the plan output and approve the PR.
4. Merging the PR into `main` triggers the workflow again and runs `terraform apply -auto-approve`, promoting the desired state into production.
5. Repeat the pattern for dedicated non-production branches (e.g., `develop`) once validated.

### GitHub Actions secrets

Add these repository secrets for the workflow to authenticate and configure Terraform:

- `AZURE_CREDENTIALS` – JSON output from `az ad sp create-for-rbac --sdk-auth`.
- `TFSTATE_RESOURCE_GROUP` – remote state resource group (`rg-tfstate-shared`).
- `TFSTATE_STORAGE_ACCOUNT` – storage account name (`sttfstategitops`).
- `TFSTATE_CONTAINER` – container name for production (`tfstate-prod`).

Update the workflow environment variable `TF_BACKEND_KEY` if you change the backend key from the default `prod/terraform.tfstate`.

Optionally set environment-specific secrets for Dev/QA workflows when added later.

### Local development

For local plans, export the Azure credentials as environment variables:

```bash
export ARM_CLIENT_ID="<clientId>"
export ARM_CLIENT_SECRET="<clientSecret>"
export ARM_SUBSCRIPTION_ID="<subscriptionId>"
export ARM_TENANT_ID="<tenantId>"
```

## Extending the workflow

- Duplicate `.github/workflows/terraform-prod.yml` for QA/Dev pipelines once the branching strategy is established.
- Introduce `tfvars` per environment for different VM sizes, allowed CIDRs, etc.
- Add policy checks (e.g., Terraform Cloud policy sets, Azure Policy) for stronger governance.

## Scripts

`scripts/create-service-principal.sh` automates service principal creation and prints the credentials in JSON and tab-separated formats.

## Security considerations

- SSH access is controlled with explicit CIDR allow lists; leave `allowed_ssh_cidrs` empty to block ingress entirely.
- Disable public IPs by default in non-production environments unless remote access is required.
- Leverage Azure Key Vault or GitHub OIDC for secretless deployments if desired.

Happy automating!
