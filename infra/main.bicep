param location string = resourceGroup().location
param acrName string
param webAppName string
param planName string = '${webAppName}-plan'
param trafficMgrName string

resource acr 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' {
  name: acrName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

resource plan 'Microsoft.Web/serverfarms@2022-03-01' {
  name: planName
  location: location
  sku: {
    name: 'B1'
    tier: 'Basic'
    size: 'B1'
    capacity: 1
  }
  properties: {
    reserved: true
  }
}

resource webApp 'Microsoft.Web/sites@2022-03-01' {
  name: webAppName
  location: location
  kind: 'app,linux,container'
  properties: {
    serverFarmId: plan.id
    siteConfig: {
      linuxFxVersion: 'DOCKER|mcr.microsoft.com/azure-app-service/samples/aspnethelloworld:latest'
    }
  }
  resource slot 'slots' 'Microsoft.Web/sites/slots@2022-03-01' = {
    name: 'green'
    properties: {
      siteConfig: {
        linuxFxVersion: webApp.properties.siteConfig.linuxFxVersion
      }
    }
  }
}

resource traffic 'Microsoft.Network/trafficManagerProfiles@2022-04-01' {
  name: trafficMgrName
  location: 'global'
  properties: {
    trafficRoutingMethod: 'Priority'
    dnsConfig: {
      relativeName: trafficMgrName
      ttl: 60
    }
    monitorConfig: {
      protocol: 'HTTP'
      port: 80
      path: '/health'
    }
    endpoints: [
      {
        name: 'primary'
        type: 'Microsoft.Network/trafficManagerProfiles/azureEndpoints'
        properties: {
          targetResourceId: webApp.id
          priority: 1
        }
      }
    ]
  }
}
