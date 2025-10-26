"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { KernelSnippets } from "./components/KernelSnippets";
import { GpuScene } from "./components/GpuScene";
import {
  type Kernel,
  type SimEvent,
  type SimResult,
  type TransposeMode,
  simulate,
} from "./lib/sim";

interface KernelOption {
  value: Kernel;
  label: string;
  description: string;
}

interface GpuPreset {
  id: string;
  name: string;
  hbmGBps: number;
  pcieGBps: number;
  warpSize: number;
  maxThreadsPerSM: number;
  maxBlocksPerSM: number;
  smCount: number;
  description: string;
}

const DEFAULT_N = 1_048_576; // 1M elements keeps 3D grid dense but readable

const KERNEL_OPTIONS: KernelOption[] = [
  {
    value: "vector_add",
    label: "Vector Add",
    description: "1D element-wise addition",
  },
  { value: "saxpy", label: "SAXPY", description: "a * X + Y (1D)" },
  {
    value: "reduce_sum",
    label: "Reduce Sum",
    description: "Parallel tree reduction",
  },
  {
    value: "transpose",
    label: "Transpose",
    description: "Naive vs. tiled shared memory",
  },
  {
    value: "matmul_tiled",
    label: "MatMul Tiled",
    description: "Tiled GEMM bandwidth estimate",
  },
];

const GPU_PRESETS: GpuPreset[] = [
  {
    id: "rtx4090",
    name: "GeForce RTX 4090",
    hbmGBps: 1008,
    pcieGBps: 32,
    warpSize: 32,
    maxThreadsPerSM: 1536,
    maxBlocksPerSM: 32,
    smCount: 128,
    description: "Ada Lovelace gaming GPU with GDDR6X",
  },
  {
    id: "a100",
    name: "NVIDIA A100 80GB",
    hbmGBps: 1555,
    pcieGBps: 32,
    warpSize: 32,
    maxThreadsPerSM: 2048,
    maxBlocksPerSM: 32,
    smCount: 108,
    description: "Data center GPU with HBM2e and PCIe Gen4",
  },
  {
    id: "h100",
    name: "NVIDIA H100 SXM",
    hbmGBps: 3000,
    pcieGBps: 64,
    warpSize: 32,
    maxThreadsPerSM: 2048,
    maxBlocksPerSM: 32,
    smCount: 132,
    description: "Hopper SXM with NVLink bandwidth headroom",
  },
];

const DEFAULT_GPU = GPU_PRESETS[1] ?? GPU_PRESETS[0];

const EVENT_COLORS: Record<SimEvent["name"], string> = {
  H2D: "#22c55e",
  kernel: "#22d3ee",
  D2H: "#f97316",
};

const EVENT_DESCRIPTIONS: Record<SimEvent["name"], string> = {
  H2D: "Copy inputs from host memory into device global memory via PCIe/NVLink.",
  kernel:
    "Execute the CUDA kernel across scheduled thread blocks on streaming multiprocessors.",
  D2H: "Transfer the resulting data back to host memory for CPU-side consumption.",
};

function formatNumber(value: number, digits = 1) {
  if (value === 0) return "0";
  if (value >= 1e9) return `${(value / 1e9).toFixed(digits)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(digits)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(digits)}K`;
  return value.toFixed(digits);
}

function formatBytes(bytes: number) {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(2)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(2)} KB`;
  return `${bytes.toFixed(0)} B`;
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function computeEventProgress(
  event: SimEvent | undefined,
  progressMs: number,
  totalDurationMs: number
) {
  if (!event) {
    return progressMs >= totalDurationMs ? 1 : 0;
  }
  const end = event.ts_ms + event.dur_ms;
  if (event.dur_ms <= 1e-6) {
    return progressMs >= end ? 1 : progressMs >= event.ts_ms ? 1 : 0;
  }
  if (progressMs <= event.ts_ms) return 0;
  if (progressMs >= end) return 1;
  return (progressMs - event.ts_ms) / event.dur_ms;
}

export default function Visualizer() {
  const [kernel, setKernel] = useState<Kernel>("vector_add");
  const [transposeMode, setTransposeMode] = useState<TransposeMode>("coalesced");
  const [gpuPresetId, setGpuPresetId] = useState<string>(DEFAULT_GPU.id);
  const activeGpu = useMemo(
    () => GPU_PRESETS.find((gpu) => gpu.id === gpuPresetId) ?? DEFAULT_GPU,
    [gpuPresetId]
  );

  const [N, setN] = useState(DEFAULT_N);
  const [M, setM] = useState(512);
  const [K, setK] = useState(512);
  const [blockDim, setBlockDim] = useState(256);
  const [tile, setTile] = useState(16);
  const [hbm, setHBM] = useState(activeGpu.hbmGBps);
  const [pcie, setPCIE] = useState(activeGpu.pcieGBps);

  useEffect(() => {
    setHBM(activeGpu.hbmGBps);
    setPCIE(activeGpu.pcieGBps);
  }, [activeGpu]);

  const [result, setResult] = useState<SimResult>(() =>
    simulate({
      kernel: "vector_add",
      N: DEFAULT_N,
      blockDimX: 256,
      hbmGBps: DEFAULT_GPU.hbmGBps,
      pcieGBps: DEFAULT_GPU.pcieGBps,
      maxThreadsPerSM: DEFAULT_GPU.maxThreadsPerSM,
      maxBlocksPerSM: DEFAULT_GPU.maxBlocksPerSM,
      warpSize: DEFAULT_GPU.warpSize,
      smCount: DEFAULT_GPU.smCount,
    })
  );

  const [isAnimating, setIsAnimating] = useState(false);
  const [animationProgress, setAnimationProgress] = useState(0);
  const rafRef = useRef<number>();

  const [showInputs, setShowInputs] = useState(true);
  const [showCode, setShowCode] = useState(true);

  const totalDuration = useMemo(() => {
    if (result.totalDurationMs > 0) {
      return result.totalDurationMs;
    }
    return result.timeline.reduce(
      (max, event) => Math.max(max, event.ts_ms + event.dur_ms),
      0
    );
  }, [result]);

  const activeStage = useMemo(() => {
    if (result.timeline.length === 0) return undefined;
    const current = result.timeline.find(
      (entry) =>
        animationProgress >= entry.ts_ms &&
        animationProgress < entry.ts_ms + entry.dur_ms
    );
    if (current) return current;
    if (animationProgress >= totalDuration) {
      return result.timeline[result.timeline.length - 1];
    }
    return undefined;
  }, [animationProgress, result, totalDuration]);

  const stageProgress = useMemo(
    () => computeEventProgress(activeStage, animationProgress, totalDuration),
    [activeStage, animationProgress, totalDuration]
  );

  const kernelEvent = useMemo(
    () => result.timeline.find((event) => event.name === "kernel"),
    [result]
  );

  const kernelProgress = useMemo(
    () => computeEventProgress(kernelEvent, animationProgress, totalDuration),
    [animationProgress, kernelEvent, totalDuration]
  );

  useEffect(() => {
    if (!isAnimating) return;
    const start = performance.now();
    const target = totalDuration + 360; // linger a moment after finishing

    const step = (now: number) => {
      const elapsed = now - start;
      const clamped = Math.min(elapsed, target);
      setAnimationProgress(clamped);
      if (clamped < target) {
        rafRef.current = requestAnimationFrame(step);
      } else {
        setIsAnimating(false);
      }
    };

    rafRef.current = requestAnimationFrame(step);

    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, [isAnimating, totalDuration]);

  useEffect(() => {
    setAnimationProgress(0);
    setIsAnimating(true);
  }, []);

  useEffect(() => {
    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, []);

  const runSimulation = () => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = undefined;
    }
    setIsAnimating(false);
    setAnimationProgress(0);

    const simResult = simulate({
      kernel,
      N,
      M,
      K,
      blockDimX: blockDim,
      tile,
      hbmGBps: hbm,
      pcieGBps: pcie,
      transposeMode,
      maxThreadsPerSM: activeGpu.maxThreadsPerSM,
      maxBlocksPerSM: activeGpu.maxBlocksPerSM,
      warpSize: activeGpu.warpSize,
      smCount: activeGpu.smCount,
    });

    setResult(simResult);
    requestAnimationFrame(() => setIsAnimating(true));
  };

  const infoCardData = [
    {
      title: "GPU preset",
      value: activeGpu.name,
      caption: `${activeGpu.hbmGBps} GB/s HBM · ${activeGpu.pcieGBps} GB/s PCIe/NVLink`,
    },
    {
      title: "Total threads",
      value: formatNumber(result.totalThreads, 1),
      caption: `${result.totalBlocks} blocks × ${result.blockDimX} threads`,
    },
    {
      title: "Occupancy",
      value: formatPercent(result.occupancy),
      caption: `${activeGpu.smCount} SMs, warp size ${result.warpSize}`,
    },
    {
      title: "Total duration",
      value: `${result.totalDurationMs.toFixed(2)} ms`,
      caption: "Simulated end-to-end latency",
    },
    {
      title: "Device bytes",
      value: formatBytes(result.bytes.device),
      caption: "Global + shared memory traffic",
    },
    {
      title: "Warps per block",
      value: formatNumber(Math.ceil(result.blockDimX / result.warpSize)),
      caption: `${result.blockDimX} threads ÷ warp size ${result.warpSize}`,
    },
  ];

  const kernelOption = KERNEL_OPTIONS.find((opt) => opt.value === kernel);

  const maxWarpsPerSM = result.maxThreadsPerSM / result.warpSize;
  const residentWarpsPerSM = maxWarpsPerSM * result.occupancy;
  const totalResidentWarps = residentWarpsPerSM * activeGpu.smCount;

  const gpuTour = [
    {
      heading: "Copy engines & PCIe/NVLink",
      detail: `${formatBytes(result.bytes.h2d)} H2D + ${formatBytes(
        result.bytes.d2h
      )} D2H over ${activeGpu.pcieGBps} GB/s links. Streams can overlap transfers with compute to hide ${EVENT_DESCRIPTIONS["H2D"].toLowerCase()}.`,
    },
    {
      heading: "HBM & L2 cache",
      detail: `${formatBytes(result.bytes.device)} touches global memory. Modern parts deliver ${activeGpu.hbmGBps} GB/s; keep hot tiles in shared memory (≈${formatNumber(
        Math.min(result.blockDimX * 4, 164_000)
      )} B/SM) for latency hiding.`,
    },
    {
      heading: "SMs, warps & occupancy",
      detail: `${formatNumber(result.totalBlocks)} blocks scheduled across ${activeGpu.smCount} SMs. Occupancy ${formatPercent(
        result.occupancy
      )} ⇒ ≈${formatNumber(totalResidentWarps)} resident warps (${formatNumber(
        residentWarpsPerSM,
        2
      )}/SM of max ${formatNumber(maxWarpsPerSM, 0)}).`,
    },
  ];

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="text-2xl font-semibold text-neutral-100">
            CUDA Visualizer — Mode A (Simulation)
          </h2>
          <p className="text-sm text-neutral-400">
            Pick a GPU preset, spin the 3D schematic freely, and watch host memory, copy engines,
            SMs, warps, and caches work together for each simulated kernel.
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setShowInputs((prev) => !prev)}
            className="rounded border border-neutral-700 px-3 py-1 text-sm text-neutral-200 transition hover:border-neutral-500"
          >
            {showInputs ? "Hide inputs" : "Show inputs"}
          </button>
          <button
            onClick={() => setShowCode((prev) => !prev)}
            className="rounded border border-neutral-700 px-3 py-1 text-sm text-neutral-200 transition hover:border-neutral-500"
          >
            {showCode ? "Hide code" : "Show code"}
          </button>
        </div>
      </div>

      <div className="flex min-h-[640px] flex-col gap-4 lg:flex-row">
        {showInputs && (
          <aside className="w-full max-w-xs shrink-0 space-y-4 rounded-lg border border-neutral-800 bg-neutral-900/60 p-4">
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wide text-neutral-500">
                GPU preset
              </label>
              <select
                value={gpuPresetId}
                onChange={(event) => setGpuPresetId(event.target.value)}
                className="mt-1 w-full rounded border border-neutral-700 bg-neutral-950/70 px-2 py-2 text-sm text-neutral-100 focus:border-sky-500 focus:outline-none"
              >
                {GPU_PRESETS.map((gpu) => (
                  <option key={gpu.id} value={gpu.id}>
                    {gpu.name}
                  </option>
                ))}
              </select>
              <p className="mt-1 text-xs text-neutral-500">{activeGpu.description}</p>
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wide text-neutral-500">
                Kernel preset
              </label>
              <select
                value={kernel}
                onChange={(event) => setKernel(event.target.value as Kernel)}
                className="mt-1 w-full rounded border border-neutral-700 bg-neutral-950/70 px-2 py-2 text-sm text-neutral-100 focus:border-sky-500 focus:outline-none"
              >
                {KERNEL_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              {kernelOption && (
                <p className="mt-1 text-xs text-neutral-500">{kernelOption.description}</p>
              )}
            </div>

            <div className="space-y-3">
              <label className="block text-xs font-semibold uppercase tracking-wide text-neutral-500">
                Problem size (N)
              </label>
              <input
                type="number"
                min={1}
                max={1_048_576}
                step={256}
                value={N}
                onChange={(event) => setN(Number.parseInt(event.target.value || "0", 10))}
                className="w-full rounded border border-neutral-700 bg-neutral-950/70 px-2 py-2 text-sm text-neutral-100 focus:border-sky-500 focus:outline-none"
              />
              <p className="text-xs text-neutral-500">
                Defaulted to one million elements so the block/warp lattice fills the scene without overwhelming it.
              </p>
            </div>

            {kernel === "matmul_tiled" && (
              <div className="grid grid-cols-2 gap-3 text-sm">
                <label className="space-y-1">
                  <span className="block text-xs font-semibold uppercase tracking-wide text-neutral-500">
                    M
                  </span>
                  <input
                    type="number"
                    min={64}
                    max={1024}
                    step={64}
                    value={M}
                    onChange={(event) =>
                      setM(Number.parseInt(event.target.value || "0", 10))
                    }
                    className="w-full rounded border border-neutral-700 bg-neutral-950/70 px-2 py-1.5 text-sm text-neutral-100 focus:border-sky-500 focus:outline-none"
                  />
                </label>
                <label className="space-y-1">
                  <span className="block text-xs font-semibold uppercase tracking-wide text-neutral-500">
                    K
                  </span>
                  <input
                    type="number"
                    min={64}
                    max={1024}
                    step={64}
                    value={K}
                    onChange={(event) =>
                      setK(Number.parseInt(event.target.value || "0", 10))
                    }
                    className="w-full rounded border border-neutral-700 bg-neutral-950/70 px-2 py-1.5 text-sm text-neutral-100 focus:border-sky-500 focus:outline-none"
                  />
                </label>
                <label className="space-y-1">
                  <span className="block text-xs font-semibold uppercase tracking-wide text-neutral-500">
                    Tile
                  </span>
                  <input
                    type="number"
                    min={8}
                    max={64}
                    step={8}
                    value={tile}
                    onChange={(event) =>
                      setTile(Number.parseInt(event.target.value || "0", 10))
                    }
                    className="w-full rounded border border-neutral-700 bg-neutral-950/70 px-2 py-1.5 text-sm text-neutral-100 focus:border-sky-500 focus:outline-none"
                  />
                </label>
              </div>
            )}

            {kernel === "transpose" && (
              <div className="space-y-2">
                <label className="block text-xs font-semibold uppercase tracking-wide text-neutral-500">
                  Transpose mode
                </label>
                <select
                  value={transposeMode}
                  onChange={(event) =>
                    setTransposeMode(event.target.value as TransposeMode)
                  }
                  className="mt-1 w-full rounded border border-neutral-700 bg-neutral-950/70 px-2 py-2 text-sm text-neutral-100 focus:border-sky-500 focus:outline-none"
                >
                  <option value="coalesced">Shared-memory tiled (coalesced)</option>
                  <option value="naive">Naive (strided)</option>
                </select>
                <p className="text-xs text-neutral-500">
                  Compare coalesced shared-memory tiles against the naive layout to feel global memory penalties instantly.
                </p>
              </div>
            )}

            <div className="grid grid-cols-2 gap-3 text-sm">
              <label className="space-y-1">
                <span className="block text-xs font-semibold uppercase tracking-wide text-neutral-500">
                  blockDim.x
                </span>
                <input
                  type="number"
                  min={32}
                  max={1024}
                  step={32}
                  value={blockDim}
                  onChange={(event) =>
                    setBlockDim(Number.parseInt(event.target.value || "0", 10))
                  }
                  className="w-full rounded border border-neutral-700 bg-neutral-950/70 px-2 py-1.5 text-sm text-neutral-100 focus:border-sky-500 focus:outline-none"
                />
              </label>
              <label className="space-y-1">
                <span className="block text-xs font-semibold uppercase tracking-wide text-neutral-500">
                  HBM (GB/s)
                </span>
                <input
                  type="number"
                  min={100}
                  max={4000}
                  step={50}
                  value={hbm}
                  onChange={(event) =>
                    setHBM(Number.parseInt(event.target.value || "0", 10))
                  }
                  className="w-full rounded border border-neutral-700 bg-neutral-950/70 px-2 py-1.5 text-sm text-neutral-100 focus:border-sky-500 focus:outline-none"
                />
              </label>
              <label className="space-y-1">
                <span className="block text-xs font-semibold uppercase tracking-wide text-neutral-500">
                  PCIe/NVLink (GB/s)
                </span>
                <input
                  type="number"
                  min={8}
                  max={128}
                  step={4}
                  value={pcie}
                  onChange={(event) =>
                    setPCIE(Number.parseInt(event.target.value || "0", 10))
                  }
                  className="w-full rounded border border-neutral-700 bg-neutral-950/70 px-2 py-1.5 text-sm text-neutral-100 focus:border-sky-500 focus:outline-none"
                />
              </label>
            </div>

            <div className="grid grid-cols-1 gap-2">
              <button
                onClick={runSimulation}
                className="w-full rounded bg-sky-500/80 py-2 text-sm font-semibold text-white shadow transition hover:bg-sky-500"
              >
                Run simulation
              </button>
              <button
                onClick={() => {
                  if (rafRef.current) {
                    cancelAnimationFrame(rafRef.current);
                    rafRef.current = undefined;
                  }
                  setHBM(activeGpu.hbmGBps);
                  setPCIE(activeGpu.pcieGBps);
                  setBlockDim(256);
                  setTile(16);
                  setM(512);
                  setK(512);
                  setN(DEFAULT_N);
                  setResult(
                    simulate({
                      kernel,
                      N: DEFAULT_N,
                      M: 512,
                      K: 512,
                      blockDimX: 256,
                      tile: 16,
                      hbmGBps: activeGpu.hbmGBps,
                      pcieGBps: activeGpu.pcieGBps,
                      transposeMode: "coalesced",
                      maxThreadsPerSM: activeGpu.maxThreadsPerSM,
                      maxBlocksPerSM: activeGpu.maxBlocksPerSM,
                      warpSize: activeGpu.warpSize,
                      smCount: activeGpu.smCount,
                    })
                  );
                  setTransposeMode("coalesced");
                  setIsAnimating(false);
                  setAnimationProgress(0);
                  requestAnimationFrame(() => setIsAnimating(true));
                }}
                className="w-full rounded border border-sky-700/60 bg-transparent py-2 text-sm font-semibold text-sky-300 transition hover:border-sky-500 hover:text-sky-200"
              >
                Reset to GPU defaults
              </button>
            </div>
          </aside>
        )}

        <section className="flex-1 space-y-4">
          <div className="overflow-hidden rounded-xl border border-neutral-800 bg-neutral-950/70">
            <GpuScene
              result={result}
              kernelProgress={kernelProgress}
              activeStage={activeStage}
              stageProgress={stageProgress}
            />
          </div>

          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-6">
            {infoCardData.map((card) => (
              <div
                key={card.title}
                className="rounded-lg border border-neutral-800 bg-neutral-900/60 px-4 py-3"
              >
                <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
                  {card.title}
                </p>
                <p className="mt-1 text-xl font-semibold text-neutral-100">{card.value}</p>
                <p className="text-xs text-neutral-500">{card.caption}</p>
              </div>
            ))}
          </div>

          <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)]">
            <div className="rounded-lg border border-neutral-800 bg-neutral-900/60">
              <div className="border-b border-neutral-800 px-4 py-2 text-sm font-semibold uppercase tracking-wide text-neutral-400">
                Execution timeline
              </div>
              <div className="space-y-4 px-4 py-4">
                {result.timeline.map((event) => {
                  const eventProgress = computeEventProgress(
                    event,
                    animationProgress,
                    totalDuration
                  );
                  return (
                    <div key={event.name} className="space-y-2">
                      <div className="flex items-center justify-between text-sm text-neutral-300">
                        <span className="flex items-center gap-2">
                          <span
                            className="inline-block h-2 w-2 rounded-full"
                            style={{ backgroundColor: EVENT_COLORS[event.name] }}
                          />
                          {event.name.toUpperCase()}
                        </span>
                        <span>{event.dur_ms.toFixed(2)} ms</span>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-neutral-800">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${Math.max(eventProgress * 100, 2)}%`,
                            backgroundColor: EVENT_COLORS[event.name],
                            opacity: 0.75,
                          }}
                        />
                      </div>
                      <p className="text-xs text-neutral-500">
                        {EVENT_DESCRIPTIONS[event.name]}
                      </p>
                    </div>
                  );
                })}
                <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-neutral-400">
                  <span>
                    Active stage: {activeStage ? activeStage.name.toUpperCase() : "—"} ·
                    {" "}
                    {(stageProgress * 100).toFixed(0)}%
                  </span>
                  <span>Total simulated duration: {result.totalDurationMs.toFixed(2)} ms</span>
                </div>
              </div>
            </div>

            {result.notes.length > 0 && (
              <div className="rounded-lg border border-neutral-800 bg-neutral-900/60 p-4">
                <h3 className="text-sm font-semibold uppercase tracking-wide text-neutral-400">
                  Model notes
                </h3>
                <ul className="mt-2 list-disc space-y-1 pl-5 text-xs text-neutral-500">
                  {result.notes.map((note) => (
                    <li key={note}>{note}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          <div className="rounded-lg border border-neutral-800 bg-neutral-900/60">
            <div className="border-b border-neutral-800 px-4 py-2 text-sm font-semibold uppercase tracking-wide text-neutral-400">
              GPU component tour
            </div>
            <div className="space-y-3 px-4 py-4 text-sm text-neutral-300">
              {gpuTour.map((item) => (
                <div key={item.heading} className="space-y-1">
                  <p className="font-semibold text-neutral-100">{item.heading}</p>
                  <p className="text-xs leading-relaxed text-neutral-400">{item.detail}</p>
                </div>
              ))}
              <p className="text-xs text-neutral-500">
                Tip: drag to orbit, scroll to zoom, and right-click to pan for a 360° walkthrough of the GPU schematic.
              </p>
            </div>
          </div>
        </section>

        {showCode && (
          <aside className="w-full max-w-sm shrink-0">
            <KernelSnippets kernel={kernel} transposeMode={transposeMode} />
          </aside>
        )}
      </div>
    </div>
  );
}
