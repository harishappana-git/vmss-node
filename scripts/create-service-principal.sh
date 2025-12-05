#!/usr/bin/env bash
set -euo pipefail

if ! command -v az >/dev/null 2>&1; then
  echo "Azure CLI (az) is required" >&2
  exit 1
fi

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <service-principal-name> <subscription-id>" >&2
  exit 1
fi

NAME="$1"
SUBSCRIPTION_ID="$2"

JSON_OUTPUT=$(az ad sp create-for-rbac \
  --name "$NAME" \
  --role "Contributor" \
  --scopes "/subscriptions/${SUBSCRIPTION_ID}" \
  --sdk-auth)

echo "Service principal created. Store the following JSON in the AZURE_CREDENTIALS secret:" >&2
echo "${JSON_OUTPUT}"

if command -v python3 >/dev/null 2>&1; then
  echo >&2
  echo "clientId clientSecret tenantId subscriptionId" >&2
  python3 - <<'PY'
import json
data = json.loads('''${JSON_OUTPUT}''')
fields = [data.get('clientId', ''), data.get('clientSecret', ''), data.get('tenantId', ''), data.get('subscriptionId', '')]
print('\t'.join(fields))
PY
else
  echo >&2
  echo "Install python3 to print individual credential values." >&2
fi
