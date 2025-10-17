# AI Distributed Training Stack Explorer

An interactive explorer for understanding how distributed AI workloads map from dataset scale down to CUDA execution units. Adjust dataset size, batch size, GPU count, and server nodes to watch the hierarchy respond. Click any element to zoom in and keep the surrounding context visible, or switch to a sunburst treemap for an information-dense overview. Export the generated topology for offline study.

## Tech stack

- [React](https://react.dev/) + [Vite](https://vitejs.dev/) for the reactive UI shell
- [react-three-fiber](https://docs.pmnd.rs/react-three-fiber/getting-started/introduction) / [Three.js](https://threejs.org/) for layered 3D scene composition
- [D3 hierarchy utilities](https://github.com/d3/d3-hierarchy) for the sunburst treemap layout

## Getting started

1. **Install dependencies**

   ```bash
   npm install
   ```

2. **Start the development server**

   ```bash
   npm run dev
   ```

   The server listens on `http://localhost:5173` by default. Use the controls to configure dataset size, batch size, GPUs, and nodes. Click inside the spatial scene to drill into servers, GPUs, grids, blocks, warps, and threads while keeping the full hierarchy visible in faded context. Hover cards reveal metadata for each component.

3. **Build for production**

   ```bash
   npm run build
   ```

   To inspect the static build locally run `npm run preview` and open the provided URL.

## Exploring the visualisation

- **Spatial matrices** – stacked tables render servers, GPUs, grids, blocks, warps, and threads with colour-coded cards, keeping outer layers present even when you zoom inside.
- **Breadcrumb navigation** – always-visible breadcrumbs and a dedicated *Show full system* button make it easy to reset or traverse back up.
- **Hover insights** – hover on any card (or sunburst segment) to surface latency, occupancy, memory, and scheduling metadata. Click to focus a layer.
- **Performance overlay** – throughput, memory footprint, interconnect bandwidth, and epoch duration update live from your inputs.
- **Sunburst treemap** – switch to the sunburst view for a compressed representation of the full hierarchy alongside the 3D scene.
- **Legend & glossary** – colour legend and terminology cheat sheet stay pinned on the right for quick reference.
- **CSV export** – capture the generated topology (with metadata) as a CSV for notebooks, reports, or sharing.

## Notes

- Thread, warp, block, and grid counts are derived from dataset and batch inputs using heuristics to stay performant while conveying scale.
- Colours prioritise clarity over decoration: servers (blue), GPUs (green), grids (orange), blocks (yellow), warps (purple), threads (gray).
- Extend the data model in `src/topology.js` to incorporate device-specific attributes like tensor cores or network topology if required.
