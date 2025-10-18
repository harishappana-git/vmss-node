# 3D GPU Cluster & CUDA Hierarchy Explorer

This repository implements the 3D GPU explorer described in the design brief. It contains a Go backend that can emit synthetic
topology/metrics and a React + react-three-fiber frontend that renders physical drill-down scenes. The frontend now boots with
Blackwell-era defaults (DGX B200 class nodes) so the physical experience works instantly, even without the backend running.

## Getting Started

### Prerequisites

* Go 1.21+
* Node.js 18+

### Backend (optional for the physical view)

```
cd backend
go run ./...
```

The backend exposes REST endpoints under `http://localhost:8080/v1/*` and a WebSocket stream at `ws://localhost:8080/stream`. You
only need it when you want to replace the seeded demo data or stream live metrics.

### Frontend

```
cd frontend
npm install
npm run dev
```

By default the SPA loads the DGX B200 demo dataset in the browser and skips network calls, preventing the previous proxy
connection errors. To wire it up to the Go backend, provide the URLs at launch:

```
VITE_BACKEND_HTTP=http://localhost:8080 VITE_BACKEND_WS=ws://localhost:8080/stream npm run dev
```

## Features

* **Blackwell physical explorer** – cluster → rack → node → GPU drill-down with camera presets and breadcrumbs.
* **DGX B200 defaults** – node cards show Xeon 8570 CPUs, 1.44 TB HBM3e, NVLink 5 (~1.8 TB/s), and 400 Gb/s ConnectX-7 fabric.
* **GPU internals cutaway** – HBM stacks, NVSwitch hub references, and MIG guidance rendered in a schematic 3D view.
* **Responsive info panel** – tabs for overview, link capacities, and raw JSON payloads for nodes/GPUs.
* **Zoom toolbar** – home/fit/± controls wired to a camera rig with smooth easing.

## Project Structure

```
backend/   Go aggregator, topology seed, and HTTP/WebSocket server
frontend/  React + react-three-fiber SPA with seeded DGX B200 dataset
```

## Next Steps

* Re-introduce the CUDA execution view with the new navigation + breadcrumb system.
* Stream live metrics into the physical view (requires the Go backend and real adapters).
* Persist historical metrics for time travel queries in ClickHouse.
* Harden WebSocket framing and implement topic-based subscriptions.
* Add authentication and RBAC guardrails.
