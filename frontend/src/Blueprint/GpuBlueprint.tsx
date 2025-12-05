import { useMemo } from 'react'
import type { ClusterSpec, GPUSpec, MemoryDescriptor, NodeSpec, RackSpec } from '../types'
import { useExplorerStore } from '../state/selectionStore'
import { useMetricsStore } from '../state/metricsStore'
import { verboseLog } from '../lib/logging'

const VIEWBOX_WIDTH = 720
const VIEWBOX_HEIGHT = 520
const CENTER_X = VIEWBOX_WIDTH / 2
const CENTER_Y = VIEWBOX_HEIGHT / 2 + 20

const DIE_WIDTH = 220
const DIE_HEIGHT = 220
const DIE_LAYER_HEIGHT = 40

const HBM_WIDTH = 70
const HBM_HEIGHT = 36
const HBM_RADIUS = 220

function buildHbmDescriptors(gpu: GPUSpec): MemoryDescriptor[] {
  const stacks = 8
  const perStackCapacity = gpu.memoryGB / stacks
  const perStackBandwidth = (gpu.hbmBandwidthTBs * 1024) / stacks
  return Array.from({ length: stacks }, (_, idx) => ({
    id: `memory:${gpu.uuid}:hbm${idx + 1}`,
    scope: 'gpu',
    parentId: gpu.uuid,
    label: `HBM Stack ${idx + 1}`,
    type: 'HBM3e',
    capacity: `≈${perStackCapacity.toFixed(1)} GB`,
    bandwidth: `≈${perStackBandwidth.toFixed(0)} GB/s`,
    description: 'HBM3e stack bonded on-package delivering ultra-wide bandwidth to Blackwell SMs.'
  }))
}

type GpuBlueprintProps = {
  gpu: GPUSpec
  node: NodeSpec
  cluster: ClusterSpec
  rack: RackSpec
}

export function GpuBlueprint({ gpu, node, cluster, rack }: GpuBlueprintProps) {
  const select = useExplorerStore((state) => state.select)
  const openMemoryBlueprint = useExplorerStore((state) => state.openMemoryBlueprint)
  const selection = useExplorerStore((state) => state.selection)
  const gpuMetrics = useMetricsStore((state) => state.gpu[gpu.uuid])

  const hbmDescriptors = useMemo(() => buildHbmDescriptors(gpu), [gpu])

  const l2Descriptor = useMemo<MemoryDescriptor>(
    () => ({
      id: `memory:${gpu.uuid}:l2`,
      scope: 'gpu',
      parentId: gpu.uuid,
      label: 'L2 Cache Ring',
      type: 'Unified L2 cache',
      capacity: gpu.l2CacheMB ? `${gpu.l2CacheMB} MB` : '≈126 MB class',
      bandwidth: '≈12 TB/s on-die fabric',
      description: 'High-capacity L2 cache linking SM partitions, NVLink, and HBM with low-latency access.'
    }),
    [gpu.l2CacheMB, gpu.uuid]
  )

  const sharedDescriptor = useMemo<MemoryDescriptor>(
    () => ({
      id: `memory:${gpu.uuid}:shared`,
      scope: 'gpu',
      parentId: gpu.uuid,
      label: 'SM Shared / L1 Cache',
      type: 'Configurable shared memory + L1',
      capacity: '≈256 KB per SM (configurable)',
      bandwidth: 'Tens of TB/s intra-SM',
      description: 'On-die scratchpad adjacent to each SM for warp-synchronous data reuse.'
    }),
    [gpu.uuid]
  )

  const registersDescriptor = useMemo<MemoryDescriptor>(
    () => ({
      id: `memory:${gpu.uuid}:registers`,
      scope: 'gpu',
      parentId: gpu.uuid,
      label: 'Register File',
      type: '64-bit register file',
      capacity: '≈256 KB per SM (aggregate multi-MB)',
      bandwidth: 'Single-cycle SM access',
      description: 'Per-thread register banks providing immediate operands for warp execution.'
    }),
    [gpu.uuid]
  )

  const tmaDescriptor = useMemo<MemoryDescriptor>(
    () => ({
      id: `memory:${gpu.uuid}:tma`,
      scope: 'gpu',
      parentId: gpu.uuid,
      label: 'Tensor Memory Accelerator',
      type: 'TMA staging SRAM',
      capacity: 'Multi-MB staging buffers',
      bandwidth: 'Optimized for asynchronous tensor moves',
      description: 'Dedicated fabric for async tensor transfers between HBM, shared memory, and registers.'
    }),
    [gpu.uuid]
  )

  const handleMemoryClick = (descriptor: MemoryDescriptor) => {
    verboseLog('gpu blueprint memory selected', {
      memoryId: descriptor.id,
      gpuId: gpu.uuid,
      nodeId: node.id,
      label: descriptor.label
    })
    select({ kind: 'memory', id: descriptor.id }, { memoryInfo: descriptor })
    openMemoryBlueprint({
      scope: 'gpu',
      descriptor,
      node,
      gpu,
      clusterName: cluster.name,
      rackName: rack.name
    })
  }

  return (
    <svg viewBox={`0 0 ${VIEWBOX_WIDTH} ${VIEWBOX_HEIGHT}`} className="gpu-blueprint" role="presentation">
      <rect className="gpu-blueprint__board" x={120} y={60} width={VIEWBOX_WIDTH - 240} height={VIEWBOX_HEIGHT - 140} rx={30} ry={30} />
      <text x={CENTER_X} y={50} textAnchor="middle" className="gpu-blueprint__title">
        {cluster.name} · {rack.name} · {node.hostname} · {gpu.name}
      </text>
      <text x={CENTER_X} y={VIEWBOX_HEIGHT - 20} textAnchor="middle" className="gpu-blueprint__metrics">
        {gpuMetrics
          ? `${(gpuMetrics.util * 100).toFixed(0)}% util · ${gpuMetrics.memUsedGB.toFixed(0)} GB used · ${gpuMetrics.nvlinkGBs.toFixed(0)} GB/s NVLink`
          : 'Live metrics pending'}
      </text>
      <text x={CENTER_X} y={90} textAnchor="middle" className="gpu-blueprint__subtitle">
        {gpu.memoryGB} GB HBM3e · {gpu.hbmBandwidthTBs.toFixed(1)} TB/s · NVLink {gpu.nvlinkTBs.toFixed(1)} TB/s
      </text>
      {hbmDescriptors.map((descriptor, index) => {
        const angle = (index / hbmDescriptors.length) * Math.PI * 2
        const x = CENTER_X + Math.cos(angle) * HBM_RADIUS - HBM_WIDTH / 2
        const y = CENTER_Y + Math.sin(angle) * HBM_RADIUS - HBM_HEIGHT / 2
        const isSelected = selection?.kind === 'memory' && selection.id === descriptor.id
        return (
          <g
            key={descriptor.id}
            transform={`translate(${x}, ${y})`}
            className={`gpu-blueprint__hbm${isSelected ? ' is-selected' : ''}`}
            data-blueprint-interactive="true"
          >
            <rect width={HBM_WIDTH} height={HBM_HEIGHT} rx={10} ry={10} onClick={() => handleMemoryClick(descriptor)}>
              <title>{descriptor.label} · {descriptor.capacity} · {descriptor.bandwidth}</title>
            </rect>
            <text x={HBM_WIDTH / 2} y={HBM_HEIGHT / 2 + 4} textAnchor="middle">
              {descriptor.label}
            </text>
          </g>
        )
      })}
      <g transform={`translate(${CENTER_X - DIE_WIDTH / 2}, ${CENTER_Y - DIE_HEIGHT / 2})`} className="gpu-blueprint__die">
        <rect width={DIE_WIDTH} height={DIE_HEIGHT} rx={20} ry={20} />
        <g transform={`translate(20, 30)`}>
          <g
            className={`gpu-blueprint__layer${selection?.kind === 'memory' && selection.id === l2Descriptor.id ? ' is-selected' : ''}`}
            transform={`translate(0, 0)`}
            data-blueprint-interactive="true"
          >
            <rect width={DIE_WIDTH - 40} height={DIE_LAYER_HEIGHT} rx={10} ry={10} onClick={() => handleMemoryClick(l2Descriptor)}>
              <title>{l2Descriptor.label}</title>
            </rect>
            <text x={(DIE_WIDTH - 40) / 2} y={DIE_LAYER_HEIGHT / 2 + 4} textAnchor="middle">
              {l2Descriptor.label}
            </text>
          </g>
          <g
            className={`gpu-blueprint__layer${selection?.kind === 'memory' && selection.id === sharedDescriptor.id ? ' is-selected' : ''}`}
            transform={`translate(0, ${DIE_LAYER_HEIGHT + 12})`}
            data-blueprint-interactive="true"
          >
            <rect width={DIE_WIDTH - 40} height={DIE_LAYER_HEIGHT} rx={10} ry={10} onClick={() => handleMemoryClick(sharedDescriptor)}>
              <title>{sharedDescriptor.label}</title>
            </rect>
            <text x={(DIE_WIDTH - 40) / 2} y={DIE_LAYER_HEIGHT / 2 + 4} textAnchor="middle">
              {sharedDescriptor.label}
            </text>
          </g>
          <g
            className={`gpu-blueprint__layer${selection?.kind === 'memory' && selection.id === registersDescriptor.id ? ' is-selected' : ''}`}
            transform={`translate(0, ${(DIE_LAYER_HEIGHT + 12) * 2})`}
            data-blueprint-interactive="true"
          >
            <rect width={DIE_WIDTH - 40} height={DIE_LAYER_HEIGHT} rx={10} ry={10} onClick={() => handleMemoryClick(registersDescriptor)}>
              <title>{registersDescriptor.label}</title>
            </rect>
            <text x={(DIE_WIDTH - 40) / 2} y={DIE_LAYER_HEIGHT / 2 + 4} textAnchor="middle">
              {registersDescriptor.label}
            </text>
          </g>
          <g
            className={`gpu-blueprint__layer${selection?.kind === 'memory' && selection.id === tmaDescriptor.id ? ' is-selected' : ''}`}
            transform={`translate(0, ${(DIE_LAYER_HEIGHT + 12) * 3})`}
            data-blueprint-interactive="true"
          >
            <rect width={DIE_WIDTH - 40} height={DIE_LAYER_HEIGHT} rx={10} ry={10} onClick={() => handleMemoryClick(tmaDescriptor)}>
              <title>{tmaDescriptor.label}</title>
            </rect>
            <text x={(DIE_WIDTH - 40) / 2} y={DIE_LAYER_HEIGHT / 2 + 4} textAnchor="middle">
              {tmaDescriptor.label}
            </text>
          </g>
        </g>
      </g>
      <text x={CENTER_X} y={CENTER_Y + DIE_HEIGHT / 2 + 60} textAnchor="middle" className="gpu-blueprint__nvlink">
        NVLink 5 pads · {gpu.nvlinkTBs.toFixed(1)} TB/s to NVSwitch fabric
      </text>
    </svg>
  )
}
