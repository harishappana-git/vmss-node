# Rollout Runbook

1. Ensure the `green` slot is idle and `production` is serving traffic.
2. Trigger the GitHub Actions workflow or push to `main`.
3. Monitor the workflow logs:
   - Image builds and pushes to ACR.
   - Container settings are applied to the `green` slot.
   - Health check passes.
   - Slot swap occurs.
4. Verify the application in production.
