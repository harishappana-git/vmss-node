# AI Distributed Training Stack Explorer

An interactive 3D visualisation that helps you reason about the full distributed training stack used by GPU-accelerated AI workloads. Adjust dataset size, batch size, GPU count, and server nodes to see how CUDA components (threads, warps, blocks, grids) and the broader distributed architecture change. Click any element to zoom in and inspect finer execution levels.


## Tech stack

- [React](https://react.dev/) + [Vite](https://vitejs.dev/) for fast interactive UI development
- [react-three-fiber](https://docs.pmnd.rs/react-three-fiber/getting-started/introduction) and [Three.js](https://threejs.org/) for 3D scene management
- [@react-three/drei](https://github.com/pmndrs/drei) utilities for camera control, labels, and lighting helpers

## Getting started

1. **Install dependencies**

   ```bash
   npm install
   ```

2. **Start the development server**

   ```bash
   npm run dev
   ```

   The server listens on `http://localhost:5173` by default. Use the on-screen controls to tweak dataset and hardware parameters, then click on any rendered component to dive deeper into the hierarchy. Breadcrumbs let you jump back up the stack instantly.

3. **Build for production**

   ```bash
   npm run build
   ```

   To inspect the static build locally run `npm run preview` and open the provided URL.

## Exploring the visualisation

- **Distributed system level** – understand how many servers participate and the chosen interconnect.
- **Server/GPU level** – click a server to reveal its GPUs, SM estimates, and device memory footprint.
- **CUDA hierarchy** – keep drilling down through grids, blocks, warps, and threads. Each level updates the descriptive panel so you can map terminology to execution responsibilities.
- **Breadcrumbs & camera** – use the breadcrumb trail to navigate back up. Orbit controls let you pan, rotate, and zoom the 3D scene for better spatial awareness.

## Notes

- Thread, warp, block, and grid counts are derived from your dataset and batch size inputs using heuristics to keep the scene performant while conveying execution scale.
- The model favours architectural clarity over visual extravagance—colours are muted so you can focus on structure and relationships.
- Extend the data model in `src/topology.js` if you want to encode device-specific capabilities such as tensor core counts or NVLink bandwidth.
