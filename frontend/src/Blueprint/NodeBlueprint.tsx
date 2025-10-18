import { useMemo } from 'react'
import type { ClusterSpec, MemoryDescriptor, NodeSpec, RackSpec } from '../types'
import { useExplorerStore } from '../state/selectionStore'
import { useMetricsStore } from '../state/metricsStore'

const VIEWBOX_WIDTH = 960
const VIEWBOX_HEIGHT = 600

const GPU_WIDTH = 150
const GPU_HEIGHT = 80
const GPU_COL_GAP = 40
const GPU_ROW_GAP = 110

const DIMM_WIDTH = 80
const DIMM_HEIGHT = 32

function buildMemoryDescriptors(node: NodeSpec): MemoryDescriptor[] {
  const modules = 16
  const perModuleCapacity = node.systemMemoryGB / modules
  return Array.from({ length: modules }, (_, idx) => ({
    id: `memory:${node.id}:dimm${idx + 1}`,
    scope: 'node',
    parentId: node.id,
    label: `DIMM ${idx + 1}`,
    type: 'DDR5-5600 DIMM',
    capacity: `${perModuleCapacity.toFixed(0)} GB`,
    bandwidth: 'Up to 5600 MT/s',
    description: 'Dual-rank DDR5 module feeding the dual Intel Xeon 8570 sockets.'
  }))
}

type NodeBlueprintProps = {
  node: NodeSpec
  cluster: ClusterSpec
  rack: RackSpec
}

export function NodeBlueprint({ node, cluster, rack }: NodeBlueprintProps) {
  const enterGpu = useExplorerStore((state) => state.enterGpu)
  const select = useExplorerStore((state) => state.select)
  const selection = useExplorerStore((state) => state.selection)
  const openMemoryBlueprint = useExplorerStore((state) => state.openMemoryBlueprint)
  const gpuMetrics = useMetricsStore((state) => state.gpu)
  const nodeMetrics = useMetricsStore((state) => state.node[node.id])

  const memoryDescriptors = useMemo(() => buildMemoryDescriptors(node), [node])

  const gpuPositions = useMemo(() => {
    return node.gpus.map((gpu, index) => {
      const column = index % 4
      const row = Math.floor(index / 4)
      const x = 220 + column * (GPU_WIDTH + GPU_COL_GAP)
      const y = 200 + row * (GPU_HEIGHT + GPU_ROW_GAP)
      return { gpu, x, y }
    })
  }, [node.gpus])

  const dimmPositions = useMemo(() => {
    const positions: { descriptor: MemoryDescriptor; x: number; y: number }[] = []
    memoryDescriptors.forEach((descriptor, index) => {
      const topRow = index < memoryDescriptors.length / 2
      const order = index % (memoryDescriptors.length / 2)
      const x = topRow ? 120 + order * (DIMM_WIDTH + 12) : 120 + order * (DIMM_WIDTH + 12)
      const y = topRow ? 90 : VIEWBOX_HEIGHT - 140
      positions.push({ descriptor, x, y })
    })
    return positions
  }, [memoryDescriptors])

  const handleGpuClick = (gpuId: string) => {
    const gpu = node.gpus.find((item) => item.uuid === gpuId)
    if (!gpu) return
    enterGpu(
      { kind: 'gpu', id: gpu.uuid },
      {
        node,
        clusterId: cluster.id,
        clusterLabel: cluster.name,
        rackId: rack.id,
        rackLabel: rack.name,
        gpuLabel: `${gpu.name} (${gpu.model})`
      }
    )
  }

  const handleMemoryClick = (descriptor: MemoryDescriptor) => {
    select({ kind: 'memory', id: descriptor.id }, { memoryInfo: descriptor })
    openMemoryBlueprint({
      scope: 'node',
      descriptor,
      node,
      clusterName: cluster.name,
      rackName: rack.name
    })
  }

  return (
    <svg viewBox={`0 0 ${VIEWBOX_WIDTH} ${VIEWBOX_HEIGHT}`} className="node-blueprint" role="presentation">
      <rect className="node-blueprint__chassis" x={80} y={60} width={VIEWBOX_WIDTH - 160} height={VIEWBOX_HEIGHT - 120} rx={24} ry={24} />
      <text x={VIEWBOX_WIDTH / 2} y={40} textAnchor="middle" className="node-blueprint__title">
        {cluster.name} · {rack.name} · {node.hostname}
      </text>
      <text x={120} y={150} className="node-blueprint__label">
        CPU 0 · Intel Xeon 8570
      </text>
      <text x={VIEWBOX_WIDTH - 340} y={150} className="node-blueprint__label" textAnchor="end">
        CPU 1 · Intel Xeon 8570
      </text>
      <rect className="node-blueprint__cpu" x={120} y={170} width={120} height={180} rx={12} ry={12} />
      <rect className="node-blueprint__cpu" x={VIEWBOX_WIDTH - 260} y={170} width={120} height={180} rx={12} ry={12} />
      <rect className="node-blueprint__nvswitch" x={VIEWBOX_WIDTH / 2 - 120} y={160} width={240} height={60} rx={12} ry={12} />
      <text x={VIEWBOX_WIDTH / 2} y={198} textAnchor="middle" className="node-blueprint__nvswitch-label">
        NVSwitch Hub · {node.nvlinkSwitchAggregateTBs.toFixed(1)} TB/s
      </text>
      <text x={VIEWBOX_WIDTH / 2} y={240} textAnchor="middle" className="node-blueprint__link-label">
        4× ConnectX-7 · 400 Gb/s · BlueField-3 DPUs
      </text>
      {nodeMetrics ? (
        <text x={VIEWBOX_WIDTH / 2} y={VIEWBOX_HEIGHT - 40} textAnchor="middle" className="node-blueprint__metrics">
          {(nodeMetrics.cpuUtil * 100).toFixed(0)}% CPU · {nodeMetrics.memoryUsedGB.toFixed(0)} GB RAM used · {nodeMetrics.ibUtilGbps.toFixed(0)} Gb/s IB
        </text>
      ) : (
        <text x={VIEWBOX_WIDTH / 2} y={VIEWBOX_HEIGHT - 40} textAnchor="middle" className="node-blueprint__metrics">
          Live metrics pending
        </text>
      )}
      {dimmPositions.map(({ descriptor, x, y }) => {
        const isSelected = selection?.kind === 'memory' && selection.id === descriptor.id
        return (
          <g key={descriptor.id} transform={`translate(${x}, ${y})`} className={`node-blueprint__dimm${isSelected ? ' is-selected' : ''}`}>
            <rect width={DIMM_WIDTH} height={DIMM_HEIGHT} rx={6} ry={6} onClick={() => handleMemoryClick(descriptor)}>
              <title>{descriptor.label} · {descriptor.capacity}</title>
            </rect>
            <text x={DIMM_WIDTH / 2} y={DIMM_HEIGHT / 2 + 4} textAnchor="middle">
              {descriptor.label}
            </text>
          </g>
        )
      })}
      {gpuPositions.map(({ gpu, x, y }) => {
        const metrics = gpuMetrics[gpu.uuid]
        const isSelected = selection?.kind === 'gpu' && selection.id === gpu.uuid
        return (
          <g key={gpu.uuid} transform={`translate(${x}, ${y})`} className={`node-blueprint__gpu${isSelected ? ' is-selected' : ''}`}>
            <rect width={GPU_WIDTH} height={GPU_HEIGHT} rx={14} ry={14} onClick={() => handleGpuClick(gpu.uuid)}>
              <title>{gpu.name} · {gpu.memoryGB} GB HBM3e</title>
            </rect>
            <text x={GPU_WIDTH / 2} y={26} textAnchor="middle" className="node-blueprint__gpu-title">
              {gpu.name}
            </text>
            <text x={GPU_WIDTH / 2} y={48} textAnchor="middle" className="node-blueprint__gpu-subtitle">
              {metrics
                ? `${(metrics.util * 100).toFixed(0)}% util · ${metrics.memUsedGB.toFixed(0)} GB`
                : `${gpu.memoryGB} GB HBM3e`}
            </text>
            <text x={GPU_WIDTH / 2} y={68} textAnchor="middle" className="node-blueprint__gpu-subtitle">
              NVLink {gpu.nvlinkTBs.toFixed(1)} TB/s
            </text>
          </g>
        )
      })}
    </svg>
  )
}
