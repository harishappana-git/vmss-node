#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <resource-group-name> <azure-region> <storage-account-name> <container-name>" >&2
  exit 1
fi

RESOURCE_GROUP="$1"
LOCATION="$2"
STORAGE_ACCOUNT="$3"
CONTAINER="$4"

if ! command -v az >/dev/null 2>&1; then
  echo "Azure CLI (az) is required but not installed." >&2
  exit 1
fi

echo "Ensuring resource group '${RESOURCE_GROUP}' exists in '${LOCATION}'..."
az group create --name "${RESOURCE_GROUP}" --location "${LOCATION}" >/dev/null

echo "Ensuring storage account '${STORAGE_ACCOUNT}' exists..."
az storage account create \
  --name "${STORAGE_ACCOUNT}" \
  --resource-group "${RESOURCE_GROUP}" \
  --location "${LOCATION}" \
  --sku Standard_LRS \
  --kind StorageV2 \
  --allow-blob-public-access false \
  --min-tls-version TLS1_2 \
  >/dev/null

ACCOUNT_KEY=$(az storage account keys list \
  --resource-group "${RESOURCE_GROUP}" \
  --account-name "${STORAGE_ACCOUNT}" \
  --query '[0].value' \
  --output tsv)

echo "Ensuring blob container '${CONTAINER}' exists..."
az storage container create \
  --name "${CONTAINER}" \
  --account-name "${STORAGE_ACCOUNT}" \
  --account-key "${ACCOUNT_KEY}" \
  >/dev/null

echo "Remote state backend ready:"
echo "  Resource Group: ${RESOURCE_GROUP}"
echo "  Storage Account: ${STORAGE_ACCOUNT}"
echo "  Container: ${CONTAINER}"
