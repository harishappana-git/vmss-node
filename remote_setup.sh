RESOURCE_GROUP="rg-tfstate-shared"
LOCATION="eastus"
STORAGE_ACCOUNT="dhaappsstategitops" 



for ENV in dev qa prod; do
  az storage container create \
    --name "tfstate-${ENV}" \
    --account-name "$STORAGE_ACCOUNT" \
    --auth-mode login
done