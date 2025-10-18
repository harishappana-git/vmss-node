# 3D GPU Cluster & CUDA Hierarchy Explorer

This repository provides an MVP-grade implementation of the design brief for a 3D web experience that visualises GPU datacenter topology together with CUDA execution state. The project is split into a Go backend that synthesises topology and metrics streams and a React + three.js frontend that renders interactive physical and CUDA views.

## Getting Started

### Prerequisites

* Go 1.21+
* Node.js 18+

### Backend

```
cd backend
go run ./...
```

The backend exposes REST endpoints under `http://localhost:8080/v1/*` and a WebSocket-compatible stream at `ws://localhost:8080/stream` that emits synthetic metrics frames.

### Frontend

```
cd frontend
npm install
npm run dev
```

Vite proxies API calls to the Go backend (running on port 8080). Open the printed local URL to access the explorer.

## Features

* Dual mode view (physical & CUDA) with a header toggle
* GPU/node/link selection in the physical view with live metrics overlay
* CUDA kernel grid abstraction rendered as instanced blocks
* Synthetic topology seeded with realistic datacenter defaults
* Minimal websocket implementation in Go to avoid external dependencies

## Project Structure

```
backend/   Go aggregator, topology seed, and HTTP/WebSocket server
frontend/  React + react-three-fiber SPA
```

## Next Steps

* Expand CUDA scene to visualise warps/threads and SM assignment
* Persist historical metrics for time travel queries
* Harden WebSocket framing and implement topic-based subscriptions
* Add authentication and RBAC guardrails
