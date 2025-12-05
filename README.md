# AKS Hazard & Spots Demo

This repository provides an end-to-end Azure Kubernetes Service (AKS) sample that demonstrates the Kubernetes building blocks you listed: Deployments, StatefulSets, DaemonSets, meaningful Services (ClusterIP and LoadBalancer), StorageClasses with PVCs, ConfigMaps, Secrets, readiness/liveness probes, resource requests & limits, Horizontal Pod Autoscalers (HPA), labels/selectors and more. The manifests are intentionally small so you can apply them to a dev AKS cluster and observe how every component works together without extra bloat.

## Architecture overview

```
+---------------------------+        +--------------------------+
| ripley-hazard Service     | -----> | ripley-hazard Pods       |
| (LoadBalancer)            |        | (Deployment + HPA)       |
+---------------------------+        +--------------------------+
                                               |
                                               v
                                     +---------------------+
                                     | spots-api Service   |
                                     | (ClusterIP)         |
                                     +---------------------+
                                               |
              +----------------------+          v
              | node-observer        |   +----------------------+
              | DaemonSet            |   | spots-api Deployment |
              +----------------------+   +----------------------+
                                               |
                                               v
                                      +---------------------+
                                      | spots-db StatefulSet|
                                      | + PVCs on Azure Disk|
                                      +---------------------+
```

* **Namespace** – everything runs in the `hazard-spots` namespace defined in `k8s/namespaces/namespace.yaml`.
* **Storage** – `k8s/storage/storageclass.yaml` models a realistic Azure Disk CSI storage class and the PostgreSQL `StatefulSet` dynamically provisions PVCs from it.
* **Config & Secrets** – `ripley-hazard` and `spots-api` read runtime settings from ConfigMaps and Secrets.
* **Workloads** –
  * `ripley-hazard` is a frontend `Deployment` (uses `nicholasjackson/fake-service`) exposed externally through a `LoadBalancer` service. It has readiness & liveness probes, resource requests/limits, and an HPA.
  * `spots-api` is an API `Deployment` (also `fake-service`) that fronts the database and consumes the secret.
  * `spots-db` is a PostgreSQL `StatefulSet` with a `headless` service, PVC templates, and retention policy.
  * `node-observer` is a `DaemonSet` (Fluent Bit) that runs once per node, demonstrating daemon workloads.
* **Networking** – A dedicated `Service` of type `LoadBalancer` publishes the frontend through an external Azure Load Balancer while the internal API and database are exposed via `ClusterIP` services.

## Repository layout

```
.
├── README.md
└── k8s
    ├── apps
    │   ├── daemonset-node-observer.yaml
    │   ├── deployment-ripley-hazard.yaml
    │   ├── deployment-spots-api.yaml
    │   ├── hpa-ripley-hazard.yaml
    │   ├── service-ripley-hazard.yaml
    │   ├── service-spots-api.yaml
    │   ├── service-spots-db-headless.yaml
    │   └── statefulset-spots-db.yaml
    ├── config
    │   ├── configmap-ripley-hazard.yaml
    │   ├── configmap-spots-api.yaml
    │   └── secret-spots.yaml
    ├── namespaces
    │   └── namespace.yaml
    └── storage
        └── storageclass.yaml
```

## Prerequisites

1. **AKS cluster** with the [Azure CNI](https://learn.microsoft.com/azure/aks/configure-azure-cni) plugin enabled and an outbound connection to Azure Load Balancer.
2. `kubectl` >= 1.27 and the `kubelogin` Azure CLI plugin if you log in with AAD.

## Deployment steps

> **Tip:** Apply the manifests in order so that dependent resources exist when workloads start.

```bash
# 1. Namespace and storage primitives
kubectl apply -f k8s/namespaces/namespace.yaml
kubectl apply -f k8s/storage/storageclass.yaml

# 2. ConfigMaps and Secrets
kubectl apply -f k8s/config/configmap-ripley-hazard.yaml
kubectl apply -f k8s/config/configmap-spots-api.yaml
kubectl apply -f k8s/config/secret-spots.yaml

# 3. Data layer (StatefulSet + headless service)
kubectl apply -f k8s/apps/service-spots-db-headless.yaml
kubectl apply -f k8s/apps/statefulset-spots-db.yaml

# 4. API and frontend Deployments + services
kubectl apply -f k8s/apps/deployment-spots-api.yaml
kubectl apply -f k8s/apps/service-spots-api.yaml
kubectl apply -f k8s/apps/deployment-ripley-hazard.yaml
kubectl apply -f k8s/apps/service-ripley-hazard.yaml

# 5. Autoscaling and daemon workloads
kubectl apply -f k8s/apps/hpa-ripley-hazard.yaml
kubectl apply -f k8s/apps/daemonset-node-observer.yaml
```

Once the frontend service finishes provisioning the Azure Load Balancer, note its external IP with:

```bash
kubectl get service -n hazard-spots ripley-hazard
```

Browse to `http://<EXTERNAL-IP>/` to reach the frontend. The frontend proxies calls to `spots-api`, which in turn stores metadata in PostgreSQL. Pods advertise readiness via `/health` endpoints provided by the container image. The HPA scales the frontend between 2 and 5 replicas based on CPU usage.

## Cleaning up

```bash
kubectl delete -f k8s --recursive
```

This deletes workloads but retains the Azure Disks because the storage class uses a `Retain` reclaim policy—allowing you to snapshot or manually delete the disks later.

## Next steps

* Replace the sample container images with your own frontend/backend builds.
* Swap PostgreSQL for Cosmos DB using the Azure service operator.
* Parameterize the frontend service type with Helm or Kustomize overlays to support both LoadBalancer and internal-only deployments.

