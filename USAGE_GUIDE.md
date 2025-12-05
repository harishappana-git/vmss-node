# CUDA Visualizer Usage Guide

This guide walks through the core workflows in the CUDA Program Visualizer MVP. Each scenario
shows which kernel preset to choose, which parameters to tweak, and what to look for in the
simulation output.

> **Prerequisite:** Install dependencies and start the dev server.
>
> ```bash
> npm install
> npm run dev
> ```
>
> Open the app at http://localhost:3000.

## 1. Exploring Vector Addition Throughput

**Goal:** Understand how PCIe transfers dominate simple bandwidth-bound kernels.

1. Set **Kernel** to `Vector Add`.
2. Use the default `N = 1,000,000` and `blockDim.x = 256`.
3. Leave the bandwidth sliders at `HBM 900 GB/s` and `PCIe 32 GB/s`.

**What to observe**

- Timeline shows ~3.9 ms end-to-end, with PCIe transfers taking the majority of time.
- Memory arrows display 80 MB host→device and 40 MB device→host traffic.
- Occupancy indicator highlights high utilization due to 256-thread blocks.

**Try this tweak**

- Increase `N` to `5,000,000`. Watch PCIe durations scale linearly while kernel time stays small, reinforcing the bandwidth-bound nature.

## 2. Tuning Reduce Sum Block Sizes

**Goal:** See how block sizing impacts grid size and device traffic for reductions.

1. Switch **Kernel** to `Reduce Sum`.
2. Keep `N = 8,388,608` (a power of two) and try `blockDim.x = 128` vs `blockDim.x = 512`.

**What to observe**

- The **Grid blocks** counter adjusts inversely with block size (more blocks for smaller blocks).
- The notes panel explains the additional device traffic from writing per-block partial sums.
- Kernel duration changes slightly because the simulator accounts for extra global writes.

**Try this tweak**

- Lower PCIe bandwidth preset to `16 GB/s` to simulate PCIe Gen3. Observe longer transfer bars on the timeline.

## 3. Comparing Naive vs. Coalesced Transpose

**Goal:** Visualize the penalty of non-coalesced global memory access.

1. Choose **Kernel** `Transpose`.
2. Toggle the **Mode** control between `Coalesced` and `Naive`.
3. Keep `N = 1,048,576` (1024² matrix) and `blockDim.x = 256`.

**What to observe**

- Kernel duration grows ~4× in `Naive` mode because the simulator multiplies device traffic by a penalty factor.
- Notes update to explain the reasoning behind each mode.
- Kernel snippet panel switches between the naive and tiled implementations.

**Try this tweak**

- Raise `HBM` bandwidth to `1555 GB/s` (H100). Kernel time drops proportionally; compare both modes again.

## 4. Estimating Tiled Matrix Multiplication Cost

**Goal:** Inspect how matrix dimensions and tile size affect workload estimates.

1. Set **Kernel** to `MatMul Tiled`.
2. Enter `M = 1024`, `N = 1024`, `K = 1024`, `tile = 32`.
3. Keep `HBM` and `PCIe` at defaults.

**What to observe**

- The kernel event displays aggregate FLOPs and grid block count based on tiling.
- Timeline shows a longer kernel section relative to transfers, reflecting high device traffic.
- Occupancy indicator can dip if tile size implies larger per-block resource usage.

**Try this tweak**

- Reduce `tile` to `16`. Grid block count increases, but each block becomes lighter; compare kernel duration.

## 5. Saving Scenarios Manually

While the MVP runs entirely in-browser, you can manually capture interesting configurations:

1. Adjust parameters until you reach a scenario you want to share.
2. Copy the values from the controls into a markdown snippet or screenshot.
3. Paste the corresponding kernel snippet from the panel for quick reference.

This approach makes it easy to build a mini library of teaching examples while full share links are out of scope for Mode A.

---

Use these scenarios as starting points and branch out—tweak block sizes, bandwidth presets, and problem dimensions to see how the simulator responds. The visualizer updates live, so you can iterate rapidly without compiling or running CUDA code.
