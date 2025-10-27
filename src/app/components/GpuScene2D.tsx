"use client";

import { useEffect, useRef } from "react";
import type { SimEvent, SimResult } from "../lib/sim";

const STAGE_COLORS: Record<SimEvent["name"], string> = {
  H2D: "#22c55e",
  kernel: "#22d3ee",
  D2H: "#f97316",
};

interface GpuScene2DProps {
  result: SimResult;
  activeStage: SimEvent | undefined;
  stageProgress: number;
  kernelProgress: number;
}

function formatBytes(bytes: number) {
  if (bytes >= 1e12) return `${(bytes / 1e12).toFixed(2)} TB`;
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(2)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(2)} KB`;
  return `${bytes.toFixed(0)} B`;
}

function formatNumber(value: number) {
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return value.toString();
}

export function GpuScene2D({ result, activeStage, stageProgress, kernelProgress }: GpuScene2DProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const { width: cssWidth, height: cssHeight } = canvas.getBoundingClientRect();
    const width = cssWidth || 960;
    const height = cssHeight || 520;
    const dpr = window.devicePixelRatio ?? 1;

    canvas.width = width * dpr;
    canvas.height = height * dpr;

    context.setTransform(1, 0, 0, 1, 0, 0);
    context.scale(dpr, dpr);

    context.clearRect(0, 0, width, height);
    context.fillStyle = "#0f172a";
    context.fillRect(0, 0, width, height);

    const margin = 32;
    const hostRect = {
      x: margin,
      y: margin,
      w: width * 0.23,
      h: height * 0.32,
    };
    const copyRect = {
      x: width * 0.4 - 50,
      y: margin + 20,
      w: 100,
      h: 80,
    };
    const gpuRect = {
      x: width - margin - width * 0.32,
      y: margin,
      w: width * 0.32,
      h: height * 0.6,
    };

    const currentStage = activeStage?.name;
    const h2dActive = currentStage === "H2D";
    const kernelActive = currentStage === "kernel";
    const d2hActive = currentStage === "D2H";

    const drawPanel = (
      rect: { x: number; y: number; w: number; h: number },
      title: string,
      subtitle: string,
      active: boolean,
      accentColor: string
    ) => {
      context.fillStyle = active ? `${accentColor}22` : "#111827cc";
      context.strokeStyle = active ? accentColor : "#1f2937";
      context.lineWidth = active ? 2 : 1.5;
      drawRoundedRect(context, rect.x, rect.y, rect.w, rect.h, 12);
      context.fill();
      context.stroke();
      context.font = "bold 14px 'Inter', system-ui";
      context.fillStyle = "#e2e8f0";
      context.fillText(title, rect.x + 16, rect.y + 26);
      context.font = "12px 'Inter', system-ui";
      context.fillStyle = "#94a3b8";
      context.fillText(subtitle, rect.x + 16, rect.y + 46);
    };

    drawPanel(
      hostRect,
      "Host memory",
      `${formatBytes(result.bytes.h2d)} upload · ${formatBytes(result.bytes.d2h)} download`,
      h2dActive || d2hActive,
      h2dActive ? STAGE_COLORS.H2D : STAGE_COLORS.D2H
    );

    drawPanel(
      copyRect,
      "Copy engines",
      "DMA over PCIe/NVLink",
      h2dActive || d2hActive,
      h2dActive ? STAGE_COLORS.H2D : STAGE_COLORS.D2H
    );

    drawPanel(
      gpuRect,
      "GPU device",
      `${result.smCount} SMs · ${formatBytes(result.bytes.device)} touched`,
      kernelActive,
      STAGE_COLORS.kernel
    );

    const drawArrow = (
      from: { x: number; y: number },
      to: { x: number; y: number },
      label: string,
      color: string,
      active: boolean
    ) => {
      const intensity = active ? 1 : 0.35;
      const stroke = active ? color : "#334155";
      context.strokeStyle = stroke;
      context.lineWidth = active ? 3 : 2;
      context.beginPath();
      context.moveTo(from.x, from.y);
      context.lineTo(to.x, to.y);
      context.stroke();
      const angle = Math.atan2(to.y - from.y, to.x - from.x);
      const arrowLength = 12;
      context.beginPath();
      context.moveTo(to.x, to.y);
      context.lineTo(
        to.x - arrowLength * Math.cos(angle - 0.35),
        to.y - arrowLength * Math.sin(angle - 0.35)
      );
      context.lineTo(
        to.x - arrowLength * Math.cos(angle + 0.35),
        to.y - arrowLength * Math.sin(angle + 0.35)
      );
      context.closePath();
      context.fillStyle = stroke;
      context.fill();
      context.font = "12px 'Inter', system-ui";
      context.fillStyle = `rgba(148, 163, 184, ${intensity})`;
      if (label) {
        const textWidth = context.measureText(label).width;
        context.fillText(label, (from.x + to.x) / 2 - textWidth / 2, (from.y + to.y) / 2 - 8);
      }
    };

    const hostCenter = { x: hostRect.x + hostRect.w, y: hostRect.y + hostRect.h / 2 };
    const copyCenter = { x: copyRect.x + copyRect.w / 2, y: copyRect.y + copyRect.h / 2 };
    const gpuCenter = { x: gpuRect.x, y: gpuRect.y + gpuRect.h / 2 };

    drawArrow(
      { x: hostCenter.x, y: hostCenter.y - 12 },
      { x: copyCenter.x, y: copyCenter.y - 12 },
      `H2D ${formatBytes(result.bytes.h2d)}`,
      STAGE_COLORS.H2D,
      h2dActive
    );
    drawArrow(
      { x: copyCenter.x, y: copyCenter.y - 12 },
      { x: gpuCenter.x, y: gpuCenter.y - gpuRect.h / 4 },
      "",
      STAGE_COLORS.H2D,
      h2dActive
    );

    drawArrow(
      { x: gpuCenter.x, y: gpuCenter.y + gpuRect.h / 4 },
      { x: copyCenter.x, y: copyCenter.y + 20 },
      "",
      STAGE_COLORS.D2H,
      d2hActive
    );
    drawArrow(
      { x: copyCenter.x, y: copyCenter.y + 20 },
      { x: hostCenter.x, y: hostCenter.y + 20 },
      `D2H ${formatBytes(result.bytes.d2h)}`,
      STAGE_COLORS.D2H,
      d2hActive
    );

    const smRect = {
      x: gpuRect.x + 20,
      y: gpuRect.y + 70,
      w: gpuRect.w - 40,
      h: gpuRect.h * 0.42,
    };
    context.fillStyle = "#0b172e";
    context.strokeStyle = "#1e293b";
    context.lineWidth = 1;
    drawRoundedRect(context, smRect.x, smRect.y, smRect.w, smRect.h, 10);
    context.fill();
    context.stroke();
    context.font = "12px 'Inter', system-ui";
    context.fillStyle = "#94a3b8";
    context.fillText("Thread blocks scheduled across SMs", smRect.x + 8, smRect.y + 18);

    const displayBlocks = Math.min(result.totalBlocks, 400);
    const cols = Math.ceil(Math.sqrt(displayBlocks));
    const rows = Math.ceil(displayBlocks / cols);
    const gap = 2;
    const blockSize = Math.min(
      (smRect.w - gap * (cols - 1) - 16) / cols,
      (smRect.h - gap * (rows - 1) - 26) / rows
    );
    const activeBlocks = Math.round(displayBlocks * Math.min(kernelProgress, 1));
    const highlightBlocks = kernelActive || kernelProgress >= 1;

    for (let i = 0; i < displayBlocks; i += 1) {
      const row = Math.floor(i / cols);
      const col = i % cols;
      const x = smRect.x + 8 + col * (blockSize + gap);
      const y = smRect.y + 24 + row * (blockSize + gap);
      const active = i < activeBlocks && highlightBlocks;
      context.fillStyle = active ? "#38bdf8" : "#1f2937";
      context.fillRect(x, y, blockSize, blockSize);
    }

    context.font = "11px 'Inter', system-ui";
    context.fillStyle = "#64748b";
    context.fillText(
      `${formatNumber(result.totalBlocks)} blocks · ${result.blockDimX} threads/block`,
      smRect.x + 8,
      smRect.y + smRect.h - 8
    );

    const memoryLayers = [
      {
        label: "Registers",
        detail: "Per-thread, fastest storage (~256 KB/SM)",
        color: "#38bdf8",
      },
      {
        label: "Shared memory / L1",
        detail: "Cooperative tile cache (up to ~160 KB/SM)",
        color: "#22d3ee",
      },
      {
        label: "L2 cache",
        detail: "Chip-wide cache (tens of MB)",
        color: "#6366f1",
      },
      {
        label: "HBM",
        detail: `${formatBytes(result.bytes.device)} touched this run`,
        color: "#22c55e",
      },
    ];

    const layerHeight = 26;
    const layerGap = 8;
    const layerStart = smRect.y + smRect.h + 32;
    memoryLayers.forEach((layer, index) => {
      const y = layerStart + index * (layerHeight + layerGap);
      context.fillStyle = `${layer.color}30`;
      context.strokeStyle = `${layer.color}80`;
      context.lineWidth = 1.5;
      drawRoundedRect(context, gpuRect.x + 20, y, gpuRect.w - 40, layerHeight, 6);
      context.fill();
      context.stroke();
      context.font = "12px 'Inter', system-ui";
      context.fillStyle = "#e2e8f0";
      context.fillText(layer.label, gpuRect.x + 28, y + 17);
      context.font = "11px 'Inter', system-ui";
      context.fillStyle = "#94a3b8";
      context.fillText(layer.detail, gpuRect.x + 150, y + 17);
    });

    const timelineHeight = 30;
    const timelineWidth = width - margin * 2;
    const timelineY = height - margin - timelineHeight;
    const totalDuration = result.totalDurationMs || 1;

    context.fillStyle = "#111827";
    drawRoundedRect(context, margin, timelineY, timelineWidth, timelineHeight, 12);
    context.fill();

    let cursor = margin;
    result.timeline.forEach((event) => {
      const eventWidth = (event.dur_ms / totalDuration) * timelineWidth;
      context.fillStyle = `${STAGE_COLORS[event.name]}55`;
      context.fillRect(cursor, timelineY, eventWidth, timelineHeight);
      if (currentStage === event.name) {
        context.fillStyle = `${STAGE_COLORS[event.name]}aa`;
        context.fillRect(cursor, timelineY, eventWidth * Math.min(stageProgress, 1), timelineHeight);
      }
      context.fillStyle = "#e2e8f0";
      context.font = "12px 'Inter', system-ui";
      context.fillText(
        `${event.name.toUpperCase()} · ${event.dur_ms.toFixed(2)} ms`,
        cursor + 12,
        timelineY + 20
      );
      cursor += eventWidth;
    });

    const tickX =
      margin + Math.min(animationPosition(result, stageProgress, currentStage), 1) * timelineWidth;
    context.strokeStyle = "#f8fafc";
    context.beginPath();
    context.moveTo(tickX, timelineY - 6);
    context.lineTo(tickX, timelineY + timelineHeight + 6);
    context.stroke();
  }, [result, activeStage, stageProgress, kernelProgress]);

  return <canvas ref={canvasRef} className="h-[520px] w-full" />;
}

function animationPosition(result: SimResult, stageProgress: number, currentStage: SimEvent["name"] | undefined) {
  if (!currentStage) {
    return stageProgress >= 1 ? 1 : 0;
  }
  const totalDuration = result.totalDurationMs || 1;
  let elapsed = 0;
  for (const event of result.timeline) {
    if (currentStage === event.name) {
      elapsed += event.dur_ms * Math.min(stageProgress, 1);
      break;
    }
    elapsed += event.dur_ms;
  }
  return totalDuration === 0 ? 0 : elapsed / totalDuration;
}

function drawRoundedRect(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number
) {
  const r = Math.max(radius, 0);
  context.beginPath();
  context.moveTo(x + r, y);
  context.lineTo(x + width - r, y);
  context.quadraticCurveTo(x + width, y, x + width, y + r);
  context.lineTo(x + width, y + height - r);
  context.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  context.lineTo(x + r, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - r);
  context.lineTo(x, y + r);
  context.quadraticCurveTo(x, y, x + r, y);
  context.closePath();
}
