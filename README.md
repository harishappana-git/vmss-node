# Online Shopping App with Blue-Green Deployment

This project contains a simple Node.js online shopping API and infrastructure automation for deploying it to Azure App Service using a blue-green strategy. The container image is pushed to Azure Container Registry (ACR) and rolled out to App Service slots via GitHub Actions.

## Features

- Node.js + Express API exposing sample product data
- Docker container with production ready configuration
- GitHub Actions workflow that builds, tests, pushes to ACR and swaps App Service slots
- Health check endpoint used during deployment
- Runbooks for rollout and rollback of blue-green deployments
- Traffic Manager can be configured for DNS‑based failover between regions

## Prerequisites

- Azure subscription
- Azure CLI (`az`) logged in
- Azure Container Registry
- Azure App Service for Linux with two slots (`production` and `green`)
- GitHub repository secrets configured:
  - `ACR_LOGIN_SERVER` – e.g. `myregistry.azurecr.io`
  - `AZURE_CREDENTIALS` – output of `az ad sp create-for-rbac --sdk-auth`
  - `AZURE_RESOURCE_GROUP`
  - `AZURE_WEBAPP_NAME`

## Local Development

```bash
npm ci
npm test
npm start
```

Access the API at `http://localhost:3000/api/products`.

## Deployment Overview

1. GitHub Actions builds the Docker image and pushes it to ACR.
2. The `green` slot of the Web App is updated to the new image and restarted.
3. The workflow waits for the `/health` endpoint to report healthy.
4. A slot swap promotes `green` to production. Previous production becomes idle for rollback.

Traffic Manager can be pointed at the `production` hostname for primary traffic and a secondary region for failover.

## Runbooks

- [Rollout](docs/runbooks/rollout.md)
- [Rollback](docs/runbooks/rollback.md)

## License

MIT
