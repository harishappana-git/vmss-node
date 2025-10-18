import { useMemo, useState } from 'react'
import type { Topology } from '../types'
import { useExplorerStore } from '../state/selectionStore'

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

export function LeftPanel({ topology }: LeftPanelProps) {
  const selection = useExplorerStore((state) => state.selection)
  const [tab, setTab] = useState<(typeof tabs)[number]>('overview')

  const context = useMemo(() => {
    if (!selection) return null
    for (const cluster of topology.clusters) {
      for (const rack of cluster.racks) {
        for (const node of rack.nodes) {
          if (selection.kind === 'node' && node.id === selection.id) {
            return { cluster, rack, node, gpu: undefined }
          }
          if (selection.kind === 'gpu') {
            const gpu = node.gpus.find((g) => g.uuid === selection.id)
            if (gpu) {
              return { cluster, rack, node, gpu }
            }
          }
        }
      }
    }
    return null
  }, [selection, topology])

  const node = context?.node
  const gpu = context?.gpu

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

  const jsonPayload = useMemo(() => {
    if (gpu) return gpu
    if (node) return node
    return selection ?? null
  }, [gpu, node, selection])

  if (!selection || (!node && !gpu)) {
    return (
      <aside className="left-panel">
        <header>
          <h2>Selection</h2>
        </header>
        <p>Select a node or GPU to inspect Blackwell defaults.</p>
      </aside>
    )
  }

  const renderOverview = () => {
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
          </tr>
        </thead>
        <tbody>
          {ibLinks.map((link) => (
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
    <aside className="left-panel">
      <header>
        <h2>{gpu ? gpu.name : node?.hostname}</h2>
        <p>{gpu ? 'B200 internal view' : 'DGX B200 node summary'}</p>
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
