export type Kernel =
  | "vector_add"
  | "saxpy"
  | "reduce_sum"
  | "transpose"
  | "matmul_tiled";

export type TransposeMode = "coalesced" | "naive";

export interface SimInput {
  kernel: Kernel;
  N: number; // for 1D kernels; for matmul use N = P
  M?: number; // matmul MxK · KxN
  K?: number;
  blockDimX: number; // threads per block (x)
  tile?: number; // e.g., 16 or 32 for matmul
  hbmGBps: number; // e.g., 900 for A100
  pcieGBps: number; // e.g., 32 for PCIe Gen4
  transposeMode?: TransposeMode;
  maxThreadsPerSM?: number;
  maxBlocksPerSM?: number;
  warpSize?: number;
  smCount?: number;
}

export interface SimEvent {
  name: "H2D" | "kernel" | "D2H";
  ts_ms: number;
  dur_ms: number;
  meta: Record<string, unknown>;
}

export interface SimResult {
  timeline: SimEvent[];
  bytes: {
    h2d: number;
    d2h: number;
    device: number;
  };
  roofline_ms: number;
  gridX: number;
  blockDimX: number;
  occupancy: number;
  notes: string[];
  warpSize: number;
  smCount: number;
  maxThreadsPerSM: number;
  totalThreads: number;
  totalBlocks: number;
  totalDurationMs: number;
}

const DEFAULTS = {
  blockDimX: 256,
  hbmGBps: 900,
  pcieGBps: 32,
  tile: 16,
  warpSize: 32,
  maxThreadsPerSM: 2048,
  maxBlocksPerSM: 32,
  smCount: 108,
};

function clampNumber(value: number, min: number, max: number) {
  if (Number.isNaN(value) || !Number.isFinite(value)) {
    return min;
  }
  return Math.min(Math.max(value, min), max);
}

function msFor(bytes: number, GBps: number) {
  return ((bytes / 1e9) / Math.max(GBps, 1e-6)) * 1000;
}

function estimateOccupancy(
  blockDim: number,
  {
    warpSize,
    maxThreadsPerSM,
    maxBlocksPerSM,
  }: {
    warpSize: number;
    maxThreadsPerSM: number;
    maxBlocksPerSM: number;
  }
) {
  const warpsPerBlock = Math.ceil(blockDim / warpSize);
  const maxWarpsPerSM = maxThreadsPerSM / warpSize;
  const blocksPerSM = Math.min(Math.floor(maxThreadsPerSM / blockDim), maxBlocksPerSM);
  const occupancy = blocksPerSM * warpsPerBlock;
  return clampNumber(occupancy / maxWarpsPerSM, 0, 1);
}

export function simulate(input: SimInput): SimResult {
  const blockDimX = clampNumber(
    input.blockDimX || DEFAULTS.blockDimX,
    32,
    1024
  );
  const N = clampNumber(input.N, 1, 50_000_000) | 0;
  const gridX = Math.ceil(N / blockDimX);
  const hbm = clampNumber(input.hbmGBps || DEFAULTS.hbmGBps, 1, 1_000);
  const pcie = clampNumber(input.pcieGBps || DEFAULTS.pcieGBps, 1, 128);
  const warpSize = clampNumber(input.warpSize || DEFAULTS.warpSize, 16, 64);
  const maxThreadsPerSM = clampNumber(
    input.maxThreadsPerSM || DEFAULTS.maxThreadsPerSM,
    blockDimX,
    4_096
  );
  const maxBlocksPerSM = clampNumber(
    input.maxBlocksPerSM || DEFAULTS.maxBlocksPerSM,
    1,
    64
  );
  const smCount = clampNumber(input.smCount || DEFAULTS.smCount, 1, 512);
  const occupancy = estimateOccupancy(blockDimX, {
    warpSize,
    maxThreadsPerSM,
    maxBlocksPerSM,
  });
  const notes: string[] = [];

  if (blockDimX % 32 !== 0) {
    notes.push("Block dimension rounded to warp multiple for occupancy estimate.");
  }

  if (input.kernel === "vector_add" || input.kernel === "saxpy") {
    const device = 12 * N;
    const h2d = 8 * N;
    const d2h = 4 * N;
    const tH2D = msFor(h2d, pcie);
    const tKernel = msFor(device, hbm);
    const tD2H = msFor(d2h, pcie);
    const totalDurationMs = tH2D + tKernel + tD2H;
    const totalThreads = gridX * blockDimX;
    const timeline: SimEvent[] = [
      { name: "H2D", ts_ms: 0, dur_ms: tH2D, meta: { bytes: h2d } },
      {
        name: "kernel",
        ts_ms: tH2D,
        dur_ms: tKernel,
        meta: { bytes: device, gridX, blockDimX },
      },
      { name: "D2H", ts_ms: tH2D + tKernel, dur_ms: tD2H, meta: { bytes: d2h } },
    ];

    return {
      timeline,
      bytes: { h2d, d2h, device },
      roofline_ms: tKernel,
      gridX,
      blockDimX,
      occupancy,
      notes,
      warpSize,
      smCount,
      maxThreadsPerSM,
      totalThreads,
      totalBlocks: gridX,
      totalDurationMs,
    };
  }

  if (input.kernel === "reduce_sum") {
    const device = 4 * N + 4 * gridX;
    const h2d = 4 * N;
    const d2h = 4;
    const tH2D = msFor(h2d, pcie);
    const tKernel = msFor(device, hbm);
    const tD2H = msFor(d2h, pcie);
    const totalDurationMs = tH2D + tKernel + tD2H;
    const totalThreads = gridX * blockDimX;
    const timeline: SimEvent[] = [
      { name: "H2D", ts_ms: 0, dur_ms: tH2D, meta: { bytes: h2d } },
      {
        name: "kernel",
        ts_ms: tH2D,
        dur_ms: tKernel,
        meta: { deviceBytes: device, gridX, blockDimX },
      },
      { name: "D2H", ts_ms: tH2D + tKernel, dur_ms: tD2H, meta: { bytes: d2h } },
    ];

    notes.push("Additional device traffic comes from partial reductions written per block.");

    return {
      timeline,
      bytes: { h2d, d2h, device },
      roofline_ms: tKernel,
      gridX,
      blockDimX,
      occupancy,
      notes,
      warpSize,
      smCount,
      maxThreadsPerSM,
      totalThreads,
      totalBlocks: gridX,
      totalDurationMs,
    };
  }

  if (input.kernel === "transpose") {
    const mode: TransposeMode = input.transposeMode ?? "coalesced";
    const deviceBase = 8 * N;
    const penalty = mode === "naive" ? 4 : 1;
    const device = deviceBase * penalty;
    const h2d = 4 * N;
    const d2h = 4 * N;
    const tH2D = msFor(h2d, pcie);
    const tKernel = msFor(device, hbm);
    const tD2H = msFor(d2h, pcie);
    const totalDurationMs = tH2D + tKernel + tD2H;
    const totalThreads = gridX * blockDimX;
    const timeline: SimEvent[] = [
      { name: "H2D", ts_ms: 0, dur_ms: tH2D, meta: { bytes: h2d } },
      {
        name: "kernel",
        ts_ms: tH2D,
        dur_ms: tKernel,
        meta: { deviceBytes: device, coalesced: mode === "coalesced" },
      },
      { name: "D2H", ts_ms: tH2D + tKernel, dur_ms: tD2H, meta: { bytes: d2h } },
    ];

    if (mode === "naive") {
      notes.push("Naive transpose penalized to reflect non-coalesced global accesses.");
    } else {
      notes.push("Coalesced transpose uses shared memory tiles to minimize penalties.");
    }

    return {
      timeline,
      bytes: { h2d, d2h, device },
      roofline_ms: tKernel,
      gridX,
      blockDimX,
      occupancy,
      notes,
      warpSize,
      smCount,
      maxThreadsPerSM,
      totalThreads,
      totalBlocks: gridX,
      totalDurationMs,
    };
  }

  const M = clampNumber(input.M ?? 1024, 1, 4096);
  const K = clampNumber(input.K ?? 1024, 1, 4096);
  const P = clampNumber(N, 1, 4096);
  const tile = clampNumber(input.tile ?? DEFAULTS.tile, 4, 128);
  const flops = 2 * M * K * P;
  const device = 4 * (M * K + K * P + M * P);
  const h2d = 4 * (M * K + K * P);
  const d2h = 4 * (M * P);
  const tH2D = msFor(h2d, pcie);
  const tKernel = msFor(device, hbm);
  const tD2H = msFor(d2h, pcie);
  const gridMatmul = Math.ceil(P / tile) * Math.ceil(M / tile);
  const totalThreads = gridMatmul * blockDimX;
  const totalDurationMs = tH2D + tKernel + tD2H;
  const timeline: SimEvent[] = [
    { name: "H2D", ts_ms: 0, dur_ms: tH2D, meta: { bytes: h2d } },
    {
      name: "kernel",
      ts_ms: tH2D,
      dur_ms: tKernel,
      meta: { flops, tile, gridBlocks: gridMatmul },
    },
    { name: "D2H", ts_ms: tH2D + tKernel, dur_ms: tD2H, meta: { bytes: d2h } },
  ];

  notes.push("Matmul estimate assumes ideal tiling reuse and is bandwidth limited.");

  return {
    timeline,
    bytes: { h2d, d2h, device },
    roofline_ms: tKernel,
    gridX: gridMatmul,
    blockDimX,
    occupancy,
    notes,
    warpSize,
    smCount,
    maxThreadsPerSM,
    totalThreads,
    totalBlocks: gridMatmul,
    totalDurationMs,
  };
}
