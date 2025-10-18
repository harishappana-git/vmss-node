import { useMemo, useState } from 'react'
import type { Topology } from '../types'
import { useExplorerStore } from '../state/selectionStore'
import { useMetricsStore } from '../state/metricsStore'

const tabs = ['overview', 'links', 'json'] as const

type LeftPanelProps = {
  topology: Topology
}

function formatGbps(value: number) {
  return `${value.toLocaleString()} Gb/s`
}

function formatTbs(value: number) {
  return `${value.toFixed(1)} TB/s`
}

function formatPercent(value?: number) {
  if (value === undefined) return '—'
  return `${(value * 100).toFixed(0)}%`
}

function formatNumber(value?: number, unit: string) {
  if (value === undefined) return '—'
  return `${value.toFixed(1)} ${unit}`
}

export function LeftPanel({ topology }: LeftPanelProps) {
  const selection = useExplorerStore((state) => state.selection)
  const memoryInfo = useExplorerStore((state) => state.memoryInfo)
  const [tab, setTab] = useState<(typeof tabs)[number]>('overview')

  const context = useMemo(() => {
    if (!selection) return null
    for (const cluster of topology.clusters) {
      for (const rack of cluster.racks) {
        for (const node of rack.nodes) {
          if (selection.kind === 'node' && node.id === selection.id) {
            return { cluster, rack, node, gpu: undefined, memory: undefined }
          }
          if (selection.kind === 'gpu') {
            const gpu = node.gpus.find((g) => g.uuid === selection.id)
            if (gpu) {
              return { cluster, rack, node, gpu, memory: undefined }
            }
          }
          if (selection.kind === 'memory' && memoryInfo) {
            if (memoryInfo.scope === 'node' && node.id === memoryInfo.parentId) {
              return { cluster, rack, node, gpu: undefined, memory: memoryInfo }
            }
            if (memoryInfo.scope === 'gpu') {
              const gpu = node.gpus.find((g) => g.uuid === memoryInfo.parentId)
              if (gpu) {
                return { cluster, rack, node, gpu, memory: memoryInfo }
              }
            }
          }
        }
      }
    }
    return null
  }, [memoryInfo, selection, topology])

  const node = context?.node
  const gpu = context?.gpu
  const memory = context?.memory
  const nodeId = node?.id
  const gpuId = gpu?.uuid ?? (memory?.scope === 'gpu' ? memory.parentId : undefined)
  const nodeMetrics = useMetricsStore((state) => (nodeId ? state.node[nodeId] : undefined))
  const gpuMetrics = useMetricsStore((state) => (gpuId ? state.gpu[gpuId] : undefined))

  const ibLinks = useMemo(() => {
    if (!node) return []
    const cluster = context?.cluster
    if (!cluster) return []
    return cluster.links
      .filter((link) => link.type === 'IB' && (link.from.startsWith(node.id) || link.to.startsWith(node.id)))
      .map((link) => {
        const peer = link.from.startsWith(node.id) ? link.to : link.from
        return {
          id: link.id,
          type: 'InfiniBand',
          capacity: formatGbps(link.capacityGbps),
          peer
        }
      })
  }, [context, node])

  const nvLinks = useMemo(() => {
    if (!gpu || !node) return []
    return [
      {
        id: `${gpu.uuid}-nvlink`,
        type: 'NVLink 5',
        capacity: formatTbs(gpu.nvlinkTBs),
        peer: `${node.nvlinkSwitchId} (NVSwitch)`
      }
    ]
  }, [gpu, node])

  const linkMetrics = useMetricsStore((state) => {
    const result: Record<string, { bw?: number; rtt?: number }> = {}
    ibLinks.forEach((link) => {
      const metric = state.link[link.id]
      if (metric) {
        result[link.id] = { bw: metric.bwGbps, rtt: metric.rttUs }
      }
    })
    return result
  })

  const jsonPayload = useMemo(() => {
    if (memory) return memory
    if (gpu) return gpu
    if (node) return node
    return selection ?? null
  }, [gpu, memory, node, selection])

  if (!selection || (!node && !gpu && !memory)) {
    return (
      <aside className="left-panel">
        <header>
          <h2>Selection</h2>
        </header>
        <p>Select a node, GPU, or memory component to inspect Blackwell defaults.</p>
      </aside>
    )
  }

  const renderOverview = () => {
    if (memory && node) {
      const isGpuMemory = memory.scope === 'gpu'
      return (
        <ul>
          <li>
            <strong>Component</strong> {memory.label}
          </li>
          <li>
            <strong>Type</strong> {memory.type}
          </li>
          <li>
            <strong>Capacity</strong> {memory.capacity}
          </li>
          {memory.bandwidth && (
            <li>
              <strong>Bandwidth</strong> {memory.bandwidth}
            </li>
          )}
          <li>
            <strong>Belongs To</strong> {isGpuMemory && gpu ? `${gpu.name} (${gpu.model})` : node.hostname}
          </li>
          <li>
            <strong>Description</strong> {memory.description}
          </li>
          {isGpuMemory && gpuMetrics && (
            <>
              <li>
                <strong>HBM Used</strong> {`${gpuMetrics.memUsedGB.toFixed(1)} GB / ${(gpu?.memoryGB ?? 0).toFixed(0)} GB`}
              </li>
              <li>
                <strong>Utilization</strong> {formatPercent(gpuMetrics.util)}
              </li>
              <li>
                <strong>Temp</strong> {`${gpuMetrics.tempC.toFixed(1)} °C`}
              </li>
              <li>
                <strong>Power</strong> {`${gpuMetrics.powerW.toFixed(0)} W`}
              </li>
              <li>
                <strong>NVLink</strong> {formatNumber(gpuMetrics.nvlinkGBs, 'GB/s')}
              </li>
            </>
          )}
          {!isGpuMemory && nodeMetrics && (
            <>
              <li>
                <strong>System RAM Used</strong> {`${nodeMetrics.memoryUsedGB.toFixed(1)} GB / ${node.systemMemoryGB} GB`}
              </li>
              <li>
                <strong>CPU Util</strong> {formatPercent(nodeMetrics.cpuUtil)}
              </li>
              <li>
                <strong>InfiniBand</strong> {formatNumber(nodeMetrics.ibUtilGbps, 'Gb/s')}
              </li>
              <li>
                <strong>Active Jobs</strong> {nodeMetrics.jobsRunning}
              </li>
            </>
          )}
        </ul>
      )
    }

    if (gpu && node) {
      return (
        <ul>
          <li>
            <strong>Model</strong> B200 (Blackwell, SXM)
          </li>
          <li>
            <strong>HBM</strong> ≈ {gpu.memoryGB} GB HBM3e ({formatTbs(gpu.hbmBandwidthTBs)})
          </li>
          <li>
            <strong>NVLink</strong> NVLink 5 ≈ {formatTbs(gpu.nvlinkTBs)} per GPU
          </li>
          <li>
            <strong>MIG</strong> {gpu.migGuide}
          </li>
          <li>
            <strong>L2 Cache</strong> {gpu.l2CacheMB ? `${gpu.l2CacheMB} MB (Blackwell class reference)` : 'Blackwell enlarged L2 cache'}
          </li>
          <li>
            <strong>Utilization</strong> {formatPercent(gpuMetrics?.util)}
          </li>
          <li>
            <strong>HBM Used</strong> {gpuMetrics ? `${gpuMetrics.memUsedGB.toFixed(1)} GB` : '—'}
          </li>
          <li>
            <strong>NVLink</strong> {gpuMetrics ? formatNumber(gpuMetrics.nvlinkGBs, 'GB/s') : '—'}
          </li>
          <li>
            <strong>Temp</strong> {gpuMetrics ? `${gpuMetrics.tempC.toFixed(1)} °C` : '—'}
          </li>
          <li>
            <strong>Power</strong> {gpuMetrics ? `${gpuMetrics.powerW.toFixed(0)} W` : '—'}
          </li>
        </ul>
      )
    }

    if (node) {
      const gpuCount = node.gpus.length
      const perGpuMemory = node.gpus[0]?.memoryGB ?? 0
      const totalHBM = gpuCount * perGpuMemory
      return (
        <ul>
          <li>
            <strong>Model</strong> DGX B200 (8× B200)
          </li>
          <li>
            <strong>Hostname</strong> {node.hostname}
          </li>
          <li>
            <strong>CPU</strong> 2× Intel® Xeon® 8570 ({node.cpu.coresTotal} cores total)
          </li>
          <li>
            <strong>System RAM</strong> Up to 4 TB
          </li>
          <li>
            <strong>GPUs</strong> {gpuCount}× B200 — {(totalHBM / 1024).toFixed(2)} TB HBM3e total
          </li>
          <li>
            <strong>NVSwitch/NVLink</strong> Gen5 NVLink · {node.nvlinkSwitchAggregateTBs.toFixed(1)} TB/s aggregate
          </li>
          <li>
            <strong>Fabric</strong> 4× 400 Gb/s ConnectX-7 · BlueField-3 DPUs
          </li>
          <li>
            <strong>Storage</strong> {node.storage.os}
          </li>
          <li>
            <strong>CPU Util</strong> {formatPercent(nodeMetrics?.cpuUtil)}
          </li>
          <li>
            <strong>System RAM Used</strong> {nodeMetrics ? `${nodeMetrics.memoryUsedGB.toFixed(1)} GB` : '—'}
          </li>
          <li>
            <strong>InfiniBand</strong> {nodeMetrics ? formatNumber(nodeMetrics.ibUtilGbps, 'Gb/s') : '—'}
          </li>
          <li>
            <strong>Active Jobs</strong> {nodeMetrics ? nodeMetrics.jobsRunning : '—'}
          </li>
        </ul>
      )
    }

    return null
  }

  const renderLinks = () => {
    if (gpu) {
      return (
        <table>
          <thead>
            <tr>
              <th>Type</th>
              <th>Capacity</th>
              <th>Peer</th>
            </tr>
          </thead>
          <tbody>
            {nvLinks.map((link) => (
              <tr key={link.id}>
                <td>{link.type}</td>
                <td>{link.capacity}</td>
                <td>{link.peer}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )
    }

    return (
      <table>
        <thead>
          <tr>
            <th>Type</th>
            <th>Capacity</th>
            <th>Peer</th>
            <th>Live BW</th>
            <th>Latency</th>
          </tr>
        </thead>
        <tbody>
          {ibLinks.map((link) => {
            const metric = linkMetrics[link.id]
            return (
              <tr key={link.id}>
                <td>{link.type}</td>
                <td>{link.capacity}</td>
                <td>{link.peer}</td>
                <td>{metric?.bw !== undefined ? `${metric.bw.toFixed(1)} Gb/s` : '—'}</td>
                <td>{metric?.rtt !== undefined ? `${metric.rtt.toFixed(2)} µs` : '—'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    )
  }

  return (
    <aside className="left-panel">
      <header>
        <h2>{memory ? memory.label : gpu ? gpu.name : node?.hostname}</h2>
        <p>
          {memory
            ? memory.scope === 'gpu'
              ? 'GPU memory component detail'
              : 'Node memory component detail'
            : gpu
            ? 'B200 internal view'
            : 'DGX B200 node summary'}
        </p>
      </header>
      <nav>
        {tabs.map((item) => (
          <button key={item} className={item === tab ? 'active' : ''} onClick={() => setTab(item)}>
            {item.toUpperCase()}
          </button>
        ))}
      </nav>
      <section className="panel-body">
        {tab === 'overview' && renderOverview()}
        {tab === 'links' && renderLinks()}
        {tab === 'json' && <pre>{JSON.stringify(jsonPayload, null, 2)}</pre>}
      </section>
    </aside>
  )
}
