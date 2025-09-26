# Rollback Runbook

If an issue is detected after a deployment:

1. Swap slots to promote the previous production slot back:
   ```bash
   az webapp deployment slot swap \
     --resource-group <RESOURCE_GROUP> \
     --name <WEBAPP_NAME> \
     --slot production \
     --target-slot green
   ```
2. Disable the failed slot or redeploy a fixed build.
3. Create an incident report and follow up with root‑cause analysis.
