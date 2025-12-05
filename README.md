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

- **Spatial matrices with inline metrics** – each level renders as a matrix of colour-coded cards that surface utilisation, memory, and scheduling statistics directly inside the cell. A dynamic data-flow pulse highlights the path of the active batch through the hierarchy.
- **Context breadcrumbs & mini-map** – breadcrumbs remain clickable for instant jumps, while a collapsible mini-map keeps the full tree visible so you always know where you are and can hop to any ancestor or sibling layer.
- **Side-by-side table view** – toggle the tabular companion panel to inspect the focused layer as a sortable-style matrix complete with inline metrics, focus shortcuts, and compare toggles.
- **Layer comparison panel** – pin up to four components (↕) to review their properties side-by-side in the dedicated comparison drawer.
- **Performance & data exports** – view derived throughput, memory, and bandwidth summaries, and export either the full topology, the current focus subtree, or any specific hierarchy level as CSV.
- **Sunburst treemap** – switch to the sunburst mode for an information-dense overview; segments that sit on the active path glow to mirror the 3D view.
- **Legend with tooltips** – hover any legend or glossary entry for inline definitions so domain concepts stay at your fingertips.

## Notes

- Thread, warp, block, and grid counts are derived from dataset and batch inputs using heuristics to stay performant while conveying scale.
- Colours prioritise clarity over decoration: servers (blue), GPUs (green), grids (orange), blocks (yellow), warps (purple), threads (gray).
- Extend the data model in `src/topology.js` to incorporate device-specific attributes like tensor cores or network topology if required.
