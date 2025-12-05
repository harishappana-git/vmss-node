# CUDA Program Visualizer (Mode A)

A simulation-only Next.js web application that lets you explore how canonical CUDA kernel presets use GPU resources. Choose a kernel, tweak launch parameters, and visualize memory transfers and timing without executing any user code.

## Getting started

```bash
npm install
npm run dev
```

The app runs entirely in the browser using TypeScript math models, so no GPU is required.

## Features

- Preset CUDA kernels: vector add, SAXPY, reduce sum, transpose (naive vs coalesced), and tiled matrix multiplication.
- Adjustable launch parameters (problem size, block size, tile size, bandwidth assumptions).
- Real-time simulation of device/host memory traffic, PCIe transfers, and kernel roofline estimates.
- Canvas-based visualization of memory hierarchy and a Gantt-style timeline.
- Read-only kernel snippets rendered with Monaco editor for learning context.
