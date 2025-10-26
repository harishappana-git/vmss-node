"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { KernelSnippets } from "./components/KernelSnippets";
import {
  type Kernel,
  type SimResult,
  type TransposeMode,
  simulate,
} from "./lib/sim";

interface KernelOption {
  value: Kernel;
  label: string;
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

export default function Visualizer() {
  const [kernel, setKernel] = useState<Kernel>("vector_add");
  const [transposeMode, setTransposeMode] = useState<TransposeMode>("coalesced");
  const [N, setN] = useState(1_000_000);
  const [M, setM] = useState(1024);
  const [K, setK] = useState(1024);
  const [blockDim, setBlockDim] = useState(256);
  const [tile, setTile] = useState(16);
  const [hbm, setHBM] = useState(900);
  const [pcie, setPCIE] = useState(32);
  const [result, setResult] = useState<SimResult>(() =>
    simulate({
      kernel: "vector_add",
      N: 1_000_000,
      blockDimX: 256,
      hbmGBps: 900,
      pcieGBps: 32,
    })
  );

  const canvasRef = useRef<HTMLCanvasElement>(null);

  const effectiveResult = useMemo(() => result, [result]);

  useEffect(() => {
    const timer = setTimeout(() => {
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
      });
      setResult(simResult);
    }, 120);
    return () => clearTimeout(timer);
  }, [kernel, transposeMode, N, M, K, blockDim, tile, hbm, pcie]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, width, height);
    ctx.font = "13px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#f4f4f5";

    const box = (
      x: number,
      y: number,
      w: number,
      h: number,
      label: string,
      sublabel?: string
    ) => {
      ctx.strokeStyle = "#27272a";
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = "#e4e4e7";
      ctx.fillText(label, x + 12, y + 22);
      if (sublabel) {
        ctx.fillStyle = "#a1a1aa";
        ctx.fillText(sublabel, x + 12, y + 40);
      }
      ctx.fillStyle = "#f4f4f5";
    };

    const arrow = (
      x1: number,
      y1: number,
      x2: number,
      y2: number,
      text: string,
      color = "#38bdf8"
    ) => {
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      const angle = Math.atan2(y2 - y1, x2 - x1);
      ctx.beginPath();
      ctx.moveTo(x2, y2);
      ctx.lineTo(
        x2 - 10 * Math.cos(angle - Math.PI / 6),
        y2 - 10 * Math.sin(angle - Math.PI / 6)
      );
      ctx.lineTo(
        x2 - 10 * Math.cos(angle + Math.PI / 6),
        y2 - 10 * Math.sin(angle + Math.PI / 6)
      );
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
      ctx.fillStyle = "#cbd5f5";
      ctx.fillText(text, (x1 + x2) / 2 - text.length * 2.2, (y1 + y2) / 2 - 8);
      ctx.fillStyle = "#f4f4f5";
    };

    box(30, 28, 180, 72, "Host", "CPU memory");
    box(260, 28, 200, 72, "PCIe", `${pcie} GB/s`);
    box(520, 28, 220, 72, "GPU", `${hbm} GB/s HBM`);

    arrow(210, 64, 260, 64, `H2D ${formatBytes(effectiveResult.bytes.h2d)}`);
    arrow(520, 90, 320, 90, `D2H ${formatBytes(effectiveResult.bytes.d2h)}`, "#f97316");

    const timelineY = 220;
    const scale = 10; // px per ms
    ctx.fillStyle = "#a1a1aa";
    ctx.fillText("Timeline (ms)", 30, timelineY - 18);
    let cursor = 30;
    effectiveResult.timeline.forEach((event) => {
      const widthPx = Math.max(event.dur_ms * scale, 4);
      const color =
        event.name === "kernel"
          ? "#22d3ee"
          : event.name === "H2D"
          ? "#4ade80"
          : "#f97316";
      ctx.fillStyle = color;
      ctx.fillRect(cursor, timelineY, widthPx, 28);
      ctx.fillStyle = "#111827";
      ctx.fillText(
        `${event.name} ${event.dur_ms.toFixed(3)} ms`,
        cursor + 6,
        timelineY + 19
      );
      cursor += widthPx + 16;
    });

    ctx.fillStyle = "#cbd5f5";
    ctx.fillText(
      `Roofline estimate: ${effectiveResult.roofline_ms.toFixed(3)} ms`,
      30,
      timelineY + 52
    );
    ctx.fillText(
      `GridDim.x ≈ ${effectiveResult.gridX.toLocaleString()} | BlockDim.x = ${effectiveResult.blockDimX}`,
      30,
      timelineY + 72
    );
    ctx.fillText(
      `Occupancy ≈ ${formatPercent(effectiveResult.occupancy)}`,
      30,
      timelineY + 92
    );
  }, [effectiveResult, hbm, pcie]);

  const kernelDescription = useMemo(
    () => KERNEL_OPTIONS.find((opt) => opt.value === kernel)?.description ?? "",
    [kernel]
  );

  return (
    <div className="space-y-8">
      <header className="space-y-2">
        <p className="text-xs uppercase tracking-[0.2em] text-cyan-400">
          CUDA program visualizer (mode A)
        </p>
        <h1 className="text-3xl font-semibold text-white">
          See how launch parameters impact simulated GPU behavior
        </h1>
        <p className="max-w-2xl text-sm text-neutral-400">
          Adjust kernel presets, launch shapes, and bandwidth assumptions to
          instantly preview memory movement, timing, and occupancy heuristics –
          all computed safely in your browser.
        </p>
      </header>

      <section className="grid gap-6 lg:grid-cols-[2fr_1fr]">
        <div className="space-y-6 rounded-xl border border-neutral-800 bg-neutral-900/60 p-6 shadow-lg shadow-cyan-500/10">
          <div className="grid gap-4 md:grid-cols-2">
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
                Must be a multiple of warp size for ideal occupancy.
              </span>
            </label>

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
                  Coalesced mode uses shared tiles to reduce global penalties.
                </span>
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
                  max={1_000}
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
              <span className="font-medium text-neutral-300">PCIe bandwidth (GB/s)</span>
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

          <div className="rounded-xl border border-neutral-800 bg-neutral-900/70 p-4">
            <canvas
              ref={canvasRef}
              width={780}
              height={340}
              className="w-full"
              role="img"
              aria-label="Simulated memory transfers and timeline"
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
              {effectiveResult.timeline.map((event) => (
                <div
                  key={`${event.name}-${event.ts_ms}`}
                  className="rounded-lg border border-neutral-800 bg-neutral-950/60 p-3"
                >
                  {(() => {
                    const meta = event.meta as Record<string, unknown>;
                    const metaBytes =
                      typeof meta.bytes === "number"
                        ? meta.bytes
                        : Number(meta.bytes ?? NaN);
                    const metaGridX =
                      typeof meta.gridX === "number"
                        ? meta.gridX
                        : Number(meta.gridX ?? NaN);
                    const metaGridBlocks =
                      typeof meta.gridBlocks === "number"
                        ? meta.gridBlocks
                        : Number(meta.gridBlocks ?? NaN);
                    const metaBlockDim =
                      typeof meta.blockDimX === "number"
                        ? meta.blockDimX
                        : Number(meta.blockDimX ?? NaN);
                    const metaTile =
                      typeof meta.tile === "number"
                        ? meta.tile
                        : Number(meta.tile ?? NaN);
                    const metaFlops =
                      typeof meta.flops === "number"
                        ? meta.flops
                        : Number(meta.flops ?? NaN);
                    const metaCoalesced =
                      typeof meta.coalesced === "boolean"
                        ? meta.coalesced
                        : undefined;
                    return (
                      <>
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
                            {formatBytes(metaBytes)}
                          </div>
                        )}
                        {Number.isFinite(metaGridX) && (
                          <div className="text-xs text-neutral-500">
                            gridDim.x ≈ {formatNumber(metaGridX, 0)}
                          </div>
                        )}
                        {Number.isFinite(metaGridBlocks) && (
                          <div className="text-xs text-neutral-500">
                            grid blocks ≈ {formatNumber(metaGridBlocks, 0)}
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
                            FLOPs ≈ {formatNumber(metaFlops, 2)}
                          </div>
                        )}
                      </>
                    );
                  })()}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
