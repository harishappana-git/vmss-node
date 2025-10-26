"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { KernelSnippets } from "./components/KernelSnippets";
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

const PCIe_PRESETS = [16, 32, 64];
const HBM_PRESETS = [600, 900, 1555];

function formatNumber(value: number, digits = 2) {
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
  const [transposeMode, setTransposeMode] =
    useState<TransposeMode>("coalesced");
  const [gpuPresetId, setGpuPresetId] = useState<string>(DEFAULT_GPU.id);
  const activeGpu = useMemo(
    () => GPU_PRESETS.find((gpu) => gpu.id === gpuPresetId) ?? DEFAULT_GPU,
    [gpuPresetId]
  );

  const [N, setN] = useState(1_000_000);
  const [M, setM] = useState(1024);
  const [K, setK] = useState(1024);
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
      N: 1_000_000,
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

  const effectiveResult = useMemo(() => result, [result]);
  const totalDuration = useMemo(() => {
    if (effectiveResult.totalDurationMs > 0) {
      return effectiveResult.totalDurationMs;
    }
    return effectiveResult.timeline.reduce(
      (max, event) => Math.max(max, event.ts_ms + event.dur_ms),
      0
    );
  }, [effectiveResult]);

  const activeStage = useMemo(() => {
    if (effectiveResult.timeline.length === 0) return undefined;
    const current = effectiveResult.timeline.find(
      (entry) =>
        animationProgress >= entry.ts_ms &&
        animationProgress < entry.ts_ms + entry.dur_ms
    );
    if (current) return current;
    if (animationProgress >= totalDuration) {
      return effectiveResult.timeline[effectiveResult.timeline.length - 1];
    }
    return undefined;
  }, [animationProgress, effectiveResult, totalDuration]);

  const stageProgress = useMemo(
    () => computeEventProgress(activeStage, animationProgress, totalDuration),
    [activeStage, animationProgress, totalDuration]
  );

  const approxActiveSMs = useMemo(
    () => Math.min(effectiveResult.totalBlocks ?? 0, effectiveResult.smCount ?? 0),
    [effectiveResult]
  );

  const threadsPerActiveSM = useMemo(() => {
    if (approxActiveSMs === 0) return effectiveResult.blockDimX;
    return Math.round(effectiveResult.totalThreads / approxActiveSMs);
  }, [approxActiveSMs, effectiveResult]);

  useEffect(() => {
    if (!isAnimating) return;
    const start = performance.now();
    const target = totalDuration + 320;

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

  const canvasRef = useRef<HTMLCanvasElement>(null);

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
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#09090b";
    ctx.fillRect(0, 0, width, height);
    ctx.font = "13px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#f4f4f5";

    const hostBox = { x: 40, y: 40, w: 180, h: 84 };
    const pcieBox = { x: 260, y: 40, w: 200, h: 84 };
    const gpuBox = { x: 500, y: 40, w: 380, h: 220 };

    const drawPanel = (
      x: number,
      y: number,
      w: number,
      h: number,
      label: string,
      sublabel?: string
    ) => {
      ctx.strokeStyle = "#27272a";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = "#e4e4e7";
      ctx.fillText(label, x + 12, y + 22);
      if (sublabel) {
        ctx.fillStyle = "#a1a1aa";
        ctx.fillText(sublabel, x + 12, y + 42);
      }
      ctx.fillStyle = "#f4f4f5";
      ctx.lineWidth = 1;
    };

    drawPanel(hostBox.x, hostBox.y, hostBox.w, hostBox.h, "Host", "CPU memory");
    drawPanel(
      pcieBox.x,
      pcieBox.y,
      pcieBox.w,
      pcieBox.h,
      "PCIe / NVLink",
      `${pcie.toFixed(0)} GB/s`
    );
    drawPanel(
      gpuBox.x,
      gpuBox.y,
      gpuBox.w,
      gpuBox.h,
      "GPU",
      `${activeGpu.name}`
    );

    const h2dEvent = effectiveResult.timeline.find((event) => event.name === "H2D");
    const kernelEvent = effectiveResult.timeline.find((event) => event.name === "kernel");
    const d2hEvent = effectiveResult.timeline.find((event) => event.name === "D2H");

    const h2dProgress = computeEventProgress(h2dEvent, animationProgress, totalDuration);
    const kernelProgress = computeEventProgress(
      kernelEvent,
      animationProgress,
      totalDuration
    );
    const d2hProgress = computeEventProgress(d2hEvent, animationProgress, totalDuration);

    const drawArrow = (
      x1: number,
      y1: number,
      x2: number,
      y2: number,
      text: string,
      color: string,
      progress: number
    ) => {
      const dx = x2 - x1;
      const dy = y2 - y1;
      const length = Math.sqrt(dx * dx + dy * dy) || 1;
      const unitX = dx / length;
      const unitY = dy / length;

      ctx.strokeStyle = "#27272a";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();

      const progressLength = Math.max(length * Math.min(Math.max(progress, 0), 1), 0);
      if (progressLength > 0) {
        const endX = x1 + unitX * progressLength;
        const endY = y1 + unitY * progressLength;
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(endX, endY);
        ctx.stroke();

        ctx.fillStyle = color;
        const angle = Math.atan2(dy, dx);
        ctx.beginPath();
        ctx.moveTo(endX, endY);
        ctx.lineTo(
          endX - 10 * Math.cos(angle - Math.PI / 6),
          endY - 10 * Math.sin(angle - Math.PI / 6)
        );
        ctx.lineTo(
          endX - 10 * Math.cos(angle + Math.PI / 6),
          endY - 10 * Math.sin(angle + Math.PI / 6)
        );
        ctx.closePath();
        ctx.fill();
      }

      ctx.fillStyle = "#cbd5f5";
      ctx.fillText(text, (x1 + x2) / 2 - text.length * 2.2, (y1 + y2) / 2 - 12);
      ctx.fillStyle = "#f4f4f5";
      ctx.lineWidth = 1;
    };

    drawArrow(
      hostBox.x + hostBox.w,
      hostBox.y + hostBox.h * 0.5,
      gpuBox.x,
      hostBox.y + hostBox.h * 0.5,
      `H2D ${formatBytes(effectiveResult.bytes.h2d)}`,
      EVENT_COLORS.H2D,
      h2dProgress
    );

    drawArrow(
      gpuBox.x + gpuBox.w,
      hostBox.y + hostBox.h + 32,
      hostBox.x,
      hostBox.y + hostBox.h + 32,
      `D2H ${formatBytes(effectiveResult.bytes.d2h)}`,
      EVENT_COLORS.D2H,
      d2hProgress
    );

    const memSectionWidth = (gpuBox.w - 16 * 2 - 12) / 2;
    const memSectionHeight = 56;
    const memRowY = gpuBox.y + 28;
    const memRow2Y = memRowY + memSectionHeight + 14;

    const drawMemSection = (
      x: number,
      y: number,
      label: string,
      baseColor: string,
      progress: number
    ) => {
      ctx.save();
      ctx.globalAlpha = 0.3 + 0.6 * Math.min(Math.max(progress, 0), 1);
      ctx.fillStyle = baseColor;
      ctx.fillRect(x, y, memSectionWidth, memSectionHeight);
      ctx.restore();
      ctx.strokeStyle = "#3f3f46";
      ctx.strokeRect(x, y, memSectionWidth, memSectionHeight);
      ctx.fillStyle = "#f4f4f5";
      ctx.fillText(label, x + 10, y + 22);
      ctx.fillStyle = "#a1a1aa";
      ctx.fillText(
        `${Math.round(Math.min(Math.max(progress, 0), 1) * 100)}% active`,
        x + 10,
        y + 40
      );
      ctx.fillStyle = "#f4f4f5";
    };

    drawMemSection(
      gpuBox.x + 16,
      memRowY,
      "Global Memory",
      "#0ea5e9",
      Math.max(h2dProgress, d2hProgress)
    );
    drawMemSection(
      gpuBox.x + 16 + memSectionWidth + 12,
      memRowY,
      "L2 Cache",
      "#a855f7",
      kernelProgress * 0.7
    );
    drawMemSection(
      gpuBox.x + 16,
      memRow2Y,
      "Shared Memory",
      "#22c55e",
      kernelProgress
    );
    drawMemSection(
      gpuBox.x + 16 + memSectionWidth + 12,
      memRow2Y,
      "Registers",
      "#f97316",
      kernelProgress
    );

    const smArea = { x: gpuBox.x + 16, y: gpuBox.y + 130, w: 110, h: 84 };
    const blockArea = { x: smArea.x + smArea.w + 12, y: smArea.y, w: 150, h: 84 };
    const threadArea = {
      x: blockArea.x + blockArea.w + 12,
      y: smArea.y,
      w: gpuBox.x + gpuBox.w - (blockArea.x + blockArea.w) - 28,
      h: 84,
    };

    const drawLabel = (x: number, y: number, text: string) => {
      ctx.fillStyle = "#a1a1aa";
      ctx.fillText(text, x, y);
      ctx.fillStyle = "#f4f4f5";
    };

    drawLabel(smArea.x, smArea.y - 10, "Streaming Multiprocessors");
    drawLabel(blockArea.x, blockArea.y - 10, "Blocks scheduled");
    drawLabel(threadArea.x, threadArea.y - 10, "Threads per block");

    const smDisplayCount = Math.min(effectiveResult.smCount, 48);
    const smCols = Math.ceil(Math.sqrt(smDisplayCount));
    const smRows = Math.ceil(smDisplayCount / smCols);
    const smCellW = Math.max((smArea.w - (smCols + 1) * 3) / smCols, 4);
    const smCellH = Math.max((smArea.h - (smRows + 1) * 3) / smRows, 6);
    const activeSm = approxActiveSMs;
    const smHighlight = Math.round(activeSm * kernelProgress);

    for (let i = 0; i < smDisplayCount; i += 1) {
      const row = Math.floor(i / smCols);
      const col = i % smCols;
      const x = smArea.x + 3 + col * (smCellW + 3);
      const y = smArea.y + 3 + row * (smCellH + 3);
      ctx.fillStyle = "#18181b";
      if (i < activeSm) {
        ctx.fillStyle = i < smHighlight ? "#22d3ee" : "#1d4ed8";
      }
      ctx.fillRect(x, y, smCellW, smCellH);
      ctx.strokeStyle = "#27272a";
      ctx.strokeRect(x, y, smCellW, smCellH);
    }

    const blockDisplayCount = Math.min(effectiveResult.totalBlocks, 160);
    const blockCols = Math.ceil(Math.sqrt(blockDisplayCount || 1));
    const blockRows = Math.ceil((blockDisplayCount || 1) / blockCols);
    const blockCellW = Math.max((blockArea.w - (blockCols + 1) * 3) / blockCols, 5);
    const blockCellH = Math.max((blockArea.h - (blockRows + 1) * 3) / blockRows, 5);
    const blockHighlight = Math.round(blockDisplayCount * kernelProgress);

    for (let i = 0; i < blockDisplayCount; i += 1) {
      const row = Math.floor(i / blockCols);
      const col = i % blockCols;
      const x = blockArea.x + 3 + col * (blockCellW + 3);
      const y = blockArea.y + 3 + row * (blockCellH + 3);
      ctx.fillStyle = i < blockHighlight ? "#0ea5e9" : "#1f2937";
      ctx.fillRect(x, y, blockCellW, blockCellH);
      ctx.strokeStyle = "#27272a";
      ctx.strokeRect(x, y, blockCellW, blockCellH);
    }

    if (effectiveResult.totalBlocks > blockDisplayCount) {
      ctx.fillStyle = "#a1a1aa";
      ctx.fillText(
        `+${formatNumber(effectiveResult.totalBlocks - blockDisplayCount, 1)} blocks`,
        blockArea.x + blockArea.w - 110,
        blockArea.y + blockArea.h + 14
      );
      ctx.fillStyle = "#f4f4f5";
    }

    const threadDisplayCount = Math.min(effectiveResult.blockDimX, 196);
    const threadCols = Math.ceil(Math.sqrt(threadDisplayCount || 1));
    const threadRows = Math.ceil((threadDisplayCount || 1) / threadCols);
    const threadCellW = Math.max((threadArea.w - (threadCols + 1) * 2) / threadCols, 4);
    const threadCellH = Math.max((threadArea.h - (threadRows + 1) * 2) / threadRows, 4);
    const threadHighlight = Math.round(threadDisplayCount * Math.min(kernelProgress + 0.1, 1));

    for (let i = 0; i < threadDisplayCount; i += 1) {
      const row = Math.floor(i / threadCols);
      const col = i % threadCols;
      const x = threadArea.x + 2 + col * (threadCellW + 2);
      const y = threadArea.y + 2 + row * (threadCellH + 2);
      ctx.fillStyle = i < threadHighlight ? "#22c55e" : "#1f2937";
      ctx.fillRect(x, y, threadCellW, threadCellH);
      ctx.strokeStyle = "#27272a";
      ctx.strokeRect(x, y, threadCellW, threadCellH);
    }

    const timelineTop = 310;
    const timelineHeight = 32;
    const gap = 26;
    const availableWidth = width - 80;
    const timeScale = totalDuration > 0 ? Math.min(6, availableWidth / totalDuration) : 6;

    ctx.fillStyle = "#a1a1aa";
    ctx.fillText("Execution timeline", 40, timelineTop - 18);
    let cursor = 40;
    const segments: { event: SimEvent; x: number; width: number; progress: number }[] = [];
    effectiveResult.timeline.forEach((event, index) => {
      const widthPx = Math.max(event.dur_ms * timeScale, 4);
      segments.push({
        event,
        x: cursor,
        width: widthPx,
        progress: computeEventProgress(event, animationProgress, totalDuration),
      });
      cursor += widthPx;
      if (index < effectiveResult.timeline.length - 1) {
        cursor += gap;
      }
    });
    const totalTimelineWidth = cursor - 40;

    segments.forEach(({ event, x, width: w, progress }) => {
      ctx.fillStyle = "#27272a";
      ctx.fillRect(x, timelineTop, w, timelineHeight);
      ctx.fillStyle = EVENT_COLORS[event.name];
      ctx.globalAlpha = 0.25 + 0.6 * progress;
      ctx.fillRect(x, timelineTop, w * Math.max(progress, 0.05), timelineHeight);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = EVENT_COLORS[event.name];
      ctx.strokeRect(x, timelineTop, w, timelineHeight);
      ctx.fillStyle = "#e4e4e7";
      ctx.fillText(
        `${event.name.toUpperCase()} ${event.dur_ms.toFixed(2)} ms`,
        x + 8,
        timelineTop + 20
      );
    });

    if (totalDuration > 0) {
      const progressRatio = Math.min(
        animationProgress / Math.max(totalDuration, 1),
        1
      );
      const progressX = 40 + totalTimelineWidth * progressRatio;
      ctx.strokeStyle = "#fde68a";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(progressX, timelineTop - 6);
      ctx.lineTo(progressX, timelineTop + timelineHeight + 6);
      ctx.stroke();
      ctx.lineWidth = 1;
      ctx.fillStyle = "#fde68a";
      ctx.beginPath();
      ctx.arc(progressX, timelineTop - 8, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#f4f4f5";
    }

    ctx.fillStyle = "#a1a1aa";
    ctx.fillText(
      `GridDim.x ≈ ${formatNumber(effectiveResult.gridX, 0)} | BlockDim.x = ${effectiveResult.blockDimX}`,
      40,
      timelineTop + timelineHeight + 40
    );
    ctx.fillText(
      `Threads launched ≈ ${formatNumber(effectiveResult.totalThreads, 0)}`,
      40,
      timelineTop + timelineHeight + 58
    );
    ctx.fillText(
      `Occupancy ≈ ${formatPercent(effectiveResult.occupancy)} | Warp size ${effectiveResult.warpSize}`,
      40,
      timelineTop + timelineHeight + 76
    );
  }, [
    activeGpu,
    animationProgress,
    approxActiveSMs,
    effectiveResult,
    hbm,
    pcie,
    totalDuration,
  ]);

  const stageName = activeStage?.name ?? (animationProgress >= totalDuration ? "complete" : "idle");
  const stageDescription =
    stageName === "complete"
      ? "Simulation complete."
      : stageName === "idle"
      ? "Ready to run."
      : EVENT_DESCRIPTIONS[stageName as SimEvent["name"]];

  const runDisabled = isAnimating && animationProgress < totalDuration;
  const kernelDescription = useMemo(
    () => KERNEL_OPTIONS.find((opt) => opt.value === kernel)?.description ?? "",
    [kernel]
  );

  const stageTitle =
    stageName === "complete"
      ? "Complete"
      : stageName === "idle"
      ? "Idle"
      : stageName.toUpperCase();
  const stagePercent =
    stageName === "complete"
      ? 100
      : stageName === "idle"
      ? 0
      : Math.round(stageProgress * 100);

  return (
    <div className="space-y-8">
      <header className="space-y-2">
        <p className="text-xs uppercase tracking-[0.2em] text-cyan-400">
          CUDA program visualizer (mode A)
        </p>
        <h1 className="text-3xl font-semibold text-white">
          Run a kernel preset and watch grids, blocks, and memory light up
        </h1>
        <p className="max-w-3xl text-sm text-neutral-400">
          Choose an NVIDIA GPU profile, tweak launch parameters, then press
          <span className="px-1 font-medium text-white">Run simulation</span> to
          animate host↔device transfers, SM scheduling, and timeline events.
        </p>
      </header>

      <section className="grid gap-6 xl:grid-cols-[2.25fr_1fr]">
        <div className="space-y-6 rounded-xl border border-neutral-800 bg-neutral-900/60 p-6 shadow-lg shadow-cyan-500/10">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            <label className="flex flex-col gap-1 text-sm">
              <span className="font-medium text-neutral-300">GPU preset</span>
              <select
                value={gpuPresetId}
                onChange={(event) => setGpuPresetId(event.target.value)}
                className="rounded-md border border-neutral-700 bg-neutral-950 px-3 py-2 text-neutral-100 focus:border-cyan-500 focus:outline-none"
              >
                {GPU_PRESETS.map((preset) => (
                  <option key={preset.id} value={preset.id}>
                    {preset.name}
                  </option>
                ))}
              </select>
              <span className="text-xs text-neutral-500">{activeGpu.description}</span>
            </label>

            <label className="flex flex-col gap-1 text-sm">
              <span className="font-medium text-neutral-300">Kernel preset</span>
              <select
                value={kernel}
                onChange={(event) => setKernel(event.target.value as Kernel)}
                className="rounded-md border border-neutral-700 bg-neutral-950 px-3 py-2 text-neutral-100 focus:border-cyan-500 focus:outline-none"
              >
                {KERNEL_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              <span className="text-xs text-neutral-500">{kernelDescription}</span>
            </label>

            <label className="flex flex-col gap-1 text-sm">
              <span className="font-medium text-neutral-300">Elements (N)</span>
              <input
                type="number"
                min={1}
                max={50_000_000}
                value={N}
                onChange={(event) => setN(Number(event.target.value) || 1)}
                className="rounded-md border border-neutral-700 bg-neutral-950 px-3 py-2 text-neutral-100 focus:border-cyan-500 focus:outline-none"
              />
            </label>

            <label className="flex flex-col gap-1 text-sm">
              <span className="font-medium text-neutral-300">blockDim.x</span>
              <input
                type="number"
                min={32}
                max={1024}
                step={32}
                value={blockDim}
                onChange={(event) => setBlockDim(Number(event.target.value) || 32)}
                className="rounded-md border border-neutral-700 bg-neutral-950 px-3 py-2 text-neutral-100 focus:border-cyan-500 focus:outline-none"
              />
              <span className="text-xs text-neutral-500">
                Keep multiples of the warp size ({activeGpu.warpSize}) for peak
                occupancy.
              </span>
            </label>

            {kernel === "transpose" && (
              <label className="flex flex-col gap-1 text-sm">
                <span className="font-medium text-neutral-300">Transpose mode</span>
                <select
                  value={transposeMode}
                  onChange={(event) =>
                    setTransposeMode(event.target.value as TransposeMode)
                  }
                  className="rounded-md border border-neutral-700 bg-neutral-950 px-3 py-2 text-neutral-100 focus:border-cyan-500 focus:outline-none"
                >
                  <option value="coalesced">Shared-memory coalesced</option>
                  <option value="naive">Naive (no tiling)</option>
                </select>
                <span className="text-xs text-neutral-500">
                  Shared-memory tiles highlight global vs shared reuse.
                </span>
              </label>
            )}

            {kernel === "matmul_tiled" && (
              <label className="flex flex-col gap-1 text-sm">
                <span className="font-medium text-neutral-300">Tile size</span>
                <input
                  type="number"
                  min={4}
                  max={128}
                  value={tile}
                  onChange={(event) => setTile(Number(event.target.value) || 16)}
                  className="rounded-md border border-neutral-700 bg-neutral-950 px-3 py-2 text-neutral-100 focus:border-cyan-500 focus:outline-none"
                />
              </label>
            )}

            {kernel === "matmul_tiled" && (
              <div className="grid gap-4 md:col-span-2 md:grid-cols-3">
                <label className="flex flex-col gap-1 text-sm">
                  <span className="font-medium text-neutral-300">Matrix M</span>
                  <input
                    type="number"
                    min={1}
                    max={4096}
                    value={M}
                    onChange={(event) => setM(Number(event.target.value) || 1)}
                    className="rounded-md border border-neutral-700 bg-neutral-950 px-3 py-2 text-neutral-100 focus:border-cyan-500 focus:outline-none"
                  />
                </label>
                <label className="flex flex-col gap-1 text-sm">
                  <span className="font-medium text-neutral-300">Matrix N</span>
                  <input
                    type="number"
                    min={1}
                    max={4096}
                    value={N}
                    onChange={(event) => setN(Number(event.target.value) || 1)}
                    className="rounded-md border border-neutral-700 bg-neutral-950 px-3 py-2 text-neutral-100 focus:border-cyan-500 focus:outline-none"
                  />
                </label>
                <label className="flex flex-col gap-1 text-sm">
                  <span className="font-medium text-neutral-300">Matrix K</span>
                  <input
                    type="number"
                    min={1}
                    max={4096}
                    value={K}
                    onChange={(event) => setK(Number(event.target.value) || 1)}
                    className="rounded-md border border-neutral-700 bg-neutral-950 px-3 py-2 text-neutral-100 focus:border-cyan-500 focus:outline-none"
                  />
                </label>
              </div>
            )}

            <label className="flex flex-col gap-1 text-sm">
              <span className="font-medium text-neutral-300">HBM bandwidth (GB/s)</span>
              <div className="flex gap-2">
                <input
                  type="number"
                  min={1}
                  max={3_500}
                  value={hbm}
                  onChange={(event) => setHBM(Number(event.target.value) || 1)}
                  className="w-full rounded-md border border-neutral-700 bg-neutral-950 px-3 py-2 text-neutral-100 focus:border-cyan-500 focus:outline-none"
                />
                <div className="flex gap-1">
                  {HBM_PRESETS.map((value) => (
                    <button
                      key={value}
                      type="button"
                      onClick={() => setHBM(value)}
                      className="rounded-md border border-neutral-700 px-2 text-xs text-neutral-300 hover:border-cyan-500 hover:text-cyan-400"
                    >
                      {value}
                    </button>
                  ))}
                </div>
              </div>
            </label>

            <label className="flex flex-col gap-1 text-sm">
              <span className="font-medium text-neutral-300">PCIe / NVLink (GB/s)</span>
              <div className="flex gap-2">
                <input
                  type="number"
                  min={1}
                  max={128}
                  value={pcie}
                  onChange={(event) => setPCIE(Number(event.target.value) || 1)}
                  className="w-full rounded-md border border-neutral-700 bg-neutral-950 px-3 py-2 text-neutral-100 focus:border-cyan-500 focus:outline-none"
                />
                <div className="flex gap-1">
                  {PCIe_PRESETS.map((value) => (
                    <button
                      key={value}
                      type="button"
                      onClick={() => setPCIE(value)}
                      className="rounded-md border border-neutral-700 px-2 text-xs text-neutral-300 hover:border-cyan-500 hover:text-cyan-400"
                    >
                      {value}
                    </button>
                  ))}
                </div>
              </div>
            </label>
          </div>

          <div className="flex flex-wrap items-center gap-4">
            <button
              type="button"
              onClick={runSimulation}
              disabled={runDisabled}
              className="inline-flex items-center gap-2 rounded-lg bg-cyan-500 px-4 py-2 text-sm font-semibold text-black shadow-lg shadow-cyan-500/30 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {runDisabled ? "Running…" : "Run simulation"}
            </button>
            <div className="flex flex-col">
              <span className="text-xs uppercase tracking-wide text-neutral-500">
                Active stage
              </span>
              <span className="text-sm font-medium text-neutral-100">
                {stageTitle} · {stagePercent}%
              </span>
              <span className="text-xs text-neutral-500 max-w-md">
                {stageDescription}
              </span>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-3">
            <div className="rounded-lg border border-neutral-800 bg-neutral-950/70 p-4">
              <p className="text-xs uppercase text-neutral-500">H2D bytes</p>
              <p className="mt-1 text-lg font-semibold text-cyan-300">
                {formatBytes(effectiveResult.bytes.h2d)}
              </p>
            </div>
            <div className="rounded-lg border border-neutral-800 bg-neutral-950/70 p-4">
              <p className="text-xs uppercase text-neutral-500">Kernel device bytes</p>
              <p className="mt-1 text-lg font-semibold text-cyan-300">
                {formatBytes(effectiveResult.bytes.device)}
              </p>
            </div>
            <div className="rounded-lg border border-neutral-800 bg-neutral-950/70 p-4">
              <p className="text-xs uppercase text-neutral-500">D2H bytes</p>
              <p className="mt-1 text-lg font-semibold text-cyan-300">
                {formatBytes(effectiveResult.bytes.d2h)}
              </p>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-4">
            <div className="rounded-lg border border-neutral-800 bg-neutral-950/70 p-4">
              <p className="text-xs uppercase text-neutral-500">Total blocks</p>
              <p className="mt-1 text-lg font-semibold text-neutral-100">
                {formatNumber(effectiveResult.totalBlocks, 0)}
              </p>
            </div>
            <div className="rounded-lg border border-neutral-800 bg-neutral-950/70 p-4">
              <p className="text-xs uppercase text-neutral-500">Threads launched</p>
              <p className="mt-1 text-lg font-semibold text-neutral-100">
                {formatNumber(effectiveResult.totalThreads, 0)}
              </p>
            </div>
            <div className="rounded-lg border border-neutral-800 bg-neutral-950/70 p-4">
              <p className="text-xs uppercase text-neutral-500">Active SMs</p>
              <p className="mt-1 text-lg font-semibold text-neutral-100">
                {approxActiveSMs}/{effectiveResult.smCount}
              </p>
              <p className="text-xs text-neutral-500">Threads/SM ≈ {formatNumber(threadsPerActiveSM, 0)}</p>
            </div>
            <div className="rounded-lg border border-neutral-800 bg-neutral-950/70 p-4">
              <p className="text-xs uppercase text-neutral-500">Occupancy</p>
              <p className="mt-1 text-lg font-semibold text-neutral-100">
                {formatPercent(effectiveResult.occupancy)}
              </p>
              <p className="text-xs text-neutral-500">Warp size {effectiveResult.warpSize}</p>
            </div>
          </div>

          <div className="rounded-xl border border-neutral-800 bg-neutral-900/70 p-4">
            <canvas
              ref={canvasRef}
              width={940}
              height={400}
              className="w-full"
              role="img"
              aria-label="Simulated GPU memory hierarchy, block scheduling, and timeline"
            />
          </div>

          {effectiveResult.notes.length > 0 && (
            <ul className="list-disc space-y-1 pl-6 text-sm text-neutral-400">
              {effectiveResult.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          )}
        </div>

        <div className="space-y-6">
          <KernelSnippets kernel={kernel} transposeMode={transposeMode} />

          <div className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-6">
            <h2 className="text-lg font-semibold text-white">Timeline events</h2>
            <div className="mt-4 space-y-3 text-sm">
              {effectiveResult.timeline.map((event) => {
                const isActive =
                  activeStage?.name === event.name && stageName !== "complete";
                const meta = event.meta as Record<string, unknown>;
                const metaBytes =
                  typeof meta.bytes === "number" ? meta.bytes : Number(meta.bytes ?? NaN);
                const metaGridX =
                  typeof meta.gridX === "number" ? meta.gridX : Number(meta.gridX ?? NaN);
                const metaGridBlocks =
                  typeof meta.gridBlocks === "number"
                    ? meta.gridBlocks
                    : Number(meta.gridBlocks ?? NaN);
                const metaBlockDim =
                  typeof meta.blockDimX === "number"
                    ? meta.blockDimX
                    : Number(meta.blockDimX ?? NaN);
                const metaTile =
                  typeof meta.tile === "number" ? meta.tile : Number(meta.tile ?? NaN);
                const metaFlops =
                  typeof meta.flops === "number" ? meta.flops : Number(meta.flops ?? NaN);
                const metaCoalesced =
                  typeof meta.coalesced === "boolean" ? meta.coalesced : undefined;
                return (
                  <div
                    key={`${event.name}-${event.ts_ms}`}
                    className={`rounded-lg border bg-neutral-950/60 p-3 transition ${
                      isActive
                        ? "border-cyan-500/70 bg-cyan-500/10 shadow shadow-cyan-500/20"
                        : "border-neutral-800"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-neutral-200">
                        {event.name.toUpperCase()}
                      </span>
                      <span className="text-xs text-neutral-500">
                        start {event.ts_ms.toFixed(3)} ms
                      </span>
                    </div>
                    <div className="mt-1 text-neutral-300">
                      Duration {event.dur_ms.toFixed(3)} ms
                    </div>
                    {Number.isFinite(metaBytes) && (
                      <div className="text-xs text-neutral-500">
                        {formatBytes(Number(metaBytes))}
                      </div>
                    )}
                    {Number.isFinite(metaGridX) && (
                      <div className="text-xs text-neutral-500">
                        gridDim.x ≈ {formatNumber(Number(metaGridX), 0)}
                      </div>
                    )}
                    {Number.isFinite(metaGridBlocks) && (
                      <div className="text-xs text-neutral-500">
                        grid blocks ≈ {formatNumber(Number(metaGridBlocks), 0)}
                      </div>
                    )}
                    {Number.isFinite(metaBlockDim) && (
                      <div className="text-xs text-neutral-500">
                        blockDim.x = {metaBlockDim}
                      </div>
                    )}
                    {Number.isFinite(metaTile) && (
                      <div className="text-xs text-neutral-500">tile = {metaTile}</div>
                    )}
                    {metaCoalesced !== undefined && (
                      <div className="text-xs text-neutral-500">
                        {metaCoalesced
                          ? "Coalesced shared-memory path"
                          : "Naive global-memory path"}
                      </div>
                    )}
                    {Number.isFinite(metaFlops) && (
                      <div className="text-xs text-neutral-500">
                        FLOPs ≈ {formatNumber(Number(metaFlops), 2)}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
