"use client";

import { Html, OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { memo, useMemo } from "react";
import type { SimEvent, SimResult } from "../lib/sim";

type StageName = SimEvent["name"] | undefined;

interface GpuSceneProps {
  result: SimResult;
  kernelProgress: number;
  activeStage: SimEvent | undefined;
  stageProgress: number;
}

interface MemoryStackProps {
  position: [number, number, number];
  color: string;
  height: number;
  label: string;
  detail: string;
}

const MemoryStack = memo(function MemoryStack({
  position,
  color,
  height,
  label,
  detail,
}: MemoryStackProps) {
  const safeHeight = Math.max(height, 0.25);
  return (
    <group position={position}>
      <mesh position={[0, safeHeight / 2, 0]}>
        <boxGeometry args={[1.05, safeHeight, 1.05]} />
        <meshStandardMaterial color={color} roughness={0.35} metalness={0.2} />
      </mesh>
      <Html
        position={[0, safeHeight + 0.4, 0]}
        center
        style={{ pointerEvents: "none", textAlign: "center" }}
      >
        <div className="rounded bg-neutral-900/80 px-2 py-1 text-xs text-neutral-100 shadow">
          <div className="font-semibold">{label}</div>
          <div className="text-[10px] text-neutral-300">{detail}</div>
        </div>
      </Html>
    </group>
  );
});

interface BlockGridProps {
  count: number;
  highlightCount: number;
  stage: StageName;
}

const BlockGrid = memo(function BlockGrid({ count, highlightCount, stage }: BlockGridProps) {
  if (count === 0) return null;
  const cols = Math.ceil(Math.sqrt(count));
  const spacing = 1.1;
  const blocks: JSX.Element[] = [];
  for (let i = 0; i < count; i += 1) {
    const row = Math.floor(i / cols);
    const col = i % cols;
    const x = (col - cols / 2) * spacing;
    const z = (row - Math.ceil(count / cols) / 2) * spacing;
    const active = i < highlightCount;
    blocks.push(
      <mesh key={`block-${i}`} position={[x, active ? 0.65 : 0.5, z]}>
        <boxGeometry args={[0.9, active ? 1.0 : 0.8, 0.9]} />
        <meshStandardMaterial
          color={active ? "#0ea5e9" : "#1e293b"}
          emissive={active ? "#0284c7" : "#000000"}
          emissiveIntensity={active ? 0.25 : 0}
          roughness={0.4}
        />
      </mesh>
    );
  }
  return (
    <group position={[0, 0, 0]}>
      {blocks}
      <Html position={[0, 1.8, 0]} center style={{ pointerEvents: "none" }}>
        <div className="rounded bg-neutral-900/70 px-2 py-1 text-xs text-neutral-200 shadow">
          Blocks in flight ({highlightCount}/{count})
          {stage === "kernel" && <span className="ml-1 text-[10px] text-sky-300">kernel stage</span>}
        </div>
      </Html>
    </group>
  );
});

interface WarpClusterProps {
  warps: number;
  threadsPerWarp: number;
  progress: number;
}

const WarpCluster = memo(function WarpCluster({ warps, threadsPerWarp, progress }: WarpClusterProps) {
  if (warps === 0) return null;
  const maxWarps = Math.min(warps, 12);
  const spacing = 0.45;
  const threadCols = Math.min(threadsPerWarp, 8);
  const activeThreads = Math.round(threadsPerWarp * progress);

  return (
    <group position={[4.5, 0, 0]}>
      {Array.from({ length: maxWarps }).map((_, warpIndex) => {
        const x = (warpIndex % 4) * spacing;
        const z = Math.floor(warpIndex / 4) * spacing;
        return (
          <group key={`warp-${warpIndex}`} position={[x, 0.55, z]}>
            <mesh>
              <boxGeometry args={[0.4, 0.06, 0.4]} />
              <meshStandardMaterial color="#1f2937" roughness={0.3} />
            </mesh>
            <group position={[-0.18, 0.1, -0.18]}>
              {Array.from({ length: threadsPerWarp }).map((__, threadIndex) => {
                const col = threadIndex % threadCols;
                const row = Math.floor(threadIndex / threadCols);
                const tx = col * 0.09;
                const tz = row * 0.09;
                const active = threadIndex < activeThreads;
                return (
                  <mesh key={`warp-${warpIndex}-thread-${threadIndex}`} position={[tx, 0, tz]}>
                    <boxGeometry args={[0.06, 0.06, 0.06]} />
                    <meshStandardMaterial
                      color={active ? "#22c55e" : "#334155"}
                      emissive={active ? "#16a34a" : "#000"}
                      emissiveIntensity={active ? 0.35 : 0}
                    />
                  </mesh>
                );
              })}
            </group>
          </group>
        );
      })}
      <Html position={[0.6, 1.3, 0.6]} style={{ pointerEvents: "none" }}>
        <div className="rounded bg-neutral-900/70 px-2 py-1 text-xs text-neutral-200 shadow">
          Warp activity ({Math.min(maxWarps, warps)} shown)
        </div>
      </Html>
    </group>
  );
});

interface SmRackProps {
  smCount: number;
  activeCount: number;
}

const SmRack = memo(function SmRack({ smCount, activeCount }: SmRackProps) {
  const display = Math.min(smCount, 24);
  const cols = 6;
  const spacing = 0.7;
  return (
    <group position={[-5, 0, 0]}>
      {Array.from({ length: display }).map((_, index) => {
        const col = index % cols;
        const row = Math.floor(index / cols);
        const x = col * spacing;
        const z = row * spacing;
        const active = index < activeCount;
        return (
          <mesh key={`sm-${index}`} position={[x, 0.55, z]}>
            <boxGeometry args={[0.55, active ? 1.1 : 0.7, 0.55]} />
            <meshStandardMaterial
              color={active ? "#38bdf8" : "#111827"}
              emissive={active ? "#0ea5e9" : "#000"}
              emissiveIntensity={active ? 0.3 : 0}
              roughness={0.4}
            />
          </mesh>
        );
      })}
      <Html position={[0.9, 1.6, 0.9]} style={{ pointerEvents: "none" }}>
        <div className="rounded bg-neutral-900/70 px-2 py-1 text-xs text-neutral-200 shadow">
          Streaming multiprocessors ({activeCount}/{smCount})
        </div>
      </Html>
    </group>
  );
});

function normalizeHeight(bytes: number, reference: number) {
  if (bytes <= 0) return 0.25;
  const ratio = Math.log10(bytes + 10) / Math.log10(reference + 10);
  return Math.max(ratio * 3.5, 0.25);
}

export function GpuScene({ result, kernelProgress, activeStage, stageProgress }: GpuSceneProps) {
  const blockVisualCount = useMemo(
    () => Math.min(result.totalBlocks || 0, 64),
    [result.totalBlocks]
  );
  const highlightedBlocks = useMemo(
    () => Math.floor(blockVisualCount * kernelProgress),
    [blockVisualCount, kernelProgress]
  );

  const warpsPerBlock = useMemo(
    () => Math.max(1, Math.ceil(result.blockDimX / result.warpSize)),
    [result.blockDimX, result.warpSize]
  );

  const threadsPerWarp = useMemo(
    () => Math.min(result.warpSize, 32),
    [result.warpSize]
  );

  const activeSms = useMemo(
    () => Math.min(result.totalBlocks, result.smCount),
    [result.smCount, result.totalBlocks]
  );

  const memReference = useMemo(
    () =>
      Math.max(
        result.bytes.device,
        result.bytes.h2d + result.bytes.d2h,
        result.blockDimX * 4
      ),
    [result.blockDimX, result.bytes.d2h, result.bytes.device, result.bytes.h2d]
  );

  const stacks = useMemo(
    () => [
      {
        label: "Host RAM",
        detail: formatBytes(result.bytes.h2d + result.bytes.d2h),
        color: "#f59e0b",
        height: normalizeHeight(result.bytes.h2d + result.bytes.d2h, memReference),
        position: [-7, 0, -2],
      },
      {
        label: "Device global",
        detail: formatBytes(result.bytes.device),
        color: "#38bdf8",
        height: normalizeHeight(result.bytes.device, memReference),
        position: [-7, 0, 0],
      },
      {
        label: "Shared tiles",
        detail: `${result.blockDimX} threads/block`,
        color: "#a855f7",
        height: normalizeHeight(result.blockDimX * 4, memReference),
        position: [-7, 0, 2],
      },
    ],
    [memReference, result.blockDimX, result.bytes.d2h, result.bytes.device, result.bytes.h2d]
  );

  return (
    <Canvas
      camera={{ position: [8, 7, 10], fov: 45 }}
      style={{ width: "100%", height: "520px" }}
      dpr={[1, 2]}
    >
      <color attach="background" args={["#020617"]} />
      <ambientLight intensity={0.6} />
      <directionalLight position={[5, 10, 4]} intensity={0.75} />
      <spotLight position={[-12, 12, 6]} angle={0.35} penumbra={0.5} intensity={0.45} />
      <group position={[0, 0, 0]}>
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.51, 0]}>
          <planeGeometry args={[22, 14]} />
          <meshStandardMaterial color="#0b1120" />
        </mesh>
        <gridHelper args={[22, 22, "#1e293b", "#1e293b"]} position={[0, -0.5, 0]} />
        {stacks.map((stack) => (
          <MemoryStack key={stack.label} {...stack} />
        ))}
        <SmRack smCount={result.smCount} activeCount={activeSms} />
        <group position={[0, 0, 0]}>
          <BlockGrid
            count={blockVisualCount}
            highlightCount={highlightedBlocks}
            stage={activeStage?.name}
          />
        </group>
        <WarpCluster
          warps={warpsPerBlock}
          threadsPerWarp={threadsPerWarp}
          progress={activeStage?.name === "kernel" ? stageProgress : 0}
        />
        <Html position={[0, 3.5, -4]} style={{ pointerEvents: "none" }}>
          <div className="rounded bg-neutral-900/80 px-3 py-2 text-xs text-neutral-200 shadow">
            Active stage: {activeStage ? activeStage.name.toUpperCase() : "—"}
          </div>
        </Html>
      </group>
      <OrbitControls enablePan enableZoom makeDefault />
    </Canvas>
  );
}

function formatBytes(bytes: number) {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(2)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(2)} KB`;
  return `${bytes.toFixed(0)} B`;
}
