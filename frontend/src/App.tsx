import { useEffect, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useExplorerStore } from './state/selectionStore'
import { fetchTopology } from './api/client'
import type { ClusterSpec, GPUSpec, NodeSpec, RackSpec, Topology } from './types'
import { ZoomControls } from './ui/ZoomControls'
import { LeftPanel } from './Panels/LeftPanel'
import { useMetricsStream } from './hooks/useMetricsStream'
import { MemoryBlueprintOverlay } from './ui/MemoryBlueprintOverlay'
import { BlueprintViewport } from './Blueprint/BlueprintViewport'
import { BlueprintRoot } from './Blueprint/BlueprintRoot'
import { useViewportStore } from './lib/viewport'

export default function App() {
  const { data } = useQuery<Topology>({ queryKey: ['topology'], queryFn: fetchTopology })
  useMetricsStream()
  const breadcrumbs = useExplorerStore((state) => state.breadcrumbs)
  const goHome = useExplorerStore((state) => state.goHome)
  const goToBreadcrumb = useExplorerStore((state) => state.goToBreadcrumb)
  const selection = useExplorerStore((state) => state.selection)
  const memoryInfo = useExplorerStore((state) => state.memoryInfo)
  const view = useExplorerStore((state) => state.view)
  const resetViewport = useViewportStore((state) => state.reset)
  const fitViewport = useViewportStore((state) => state.fit)

  const breadcrumbNodes = useMemo(() => {
    if (!breadcrumbs.length) {
      return (
        <button type="button" onClick={goHome} className="breadcrumb-home">
          Cluster View
        </button>
      )
    }

    return (
      <>
        <button type="button" onClick={goHome} className="breadcrumb-home">
          Cluster View
        </button>
        {breadcrumbs.map((crumb, index) => (
          <span key={crumb.id}>
            <span className="breadcrumb-sep">›</span>
            <button type="button" onClick={() => goToBreadcrumb(index)}>
              {crumb.label}
            </button>
          </span>
        ))}
      </>
    )
  }, [breadcrumbs, goHome, goToBreadcrumb])

  const selectionLabel = useMemo(() => {
    if (!selection) return 'None'
    if (!data) {
      if (selection.kind === 'memory' && memoryInfo) {
        return `Memory – ${memoryInfo.label}`
      }
      return `${selection.kind.toUpperCase()} – ${selection.id}`
    }

    if (selection.kind === 'cluster') {
      const cluster = data.clusters.find((item) => item.id === selection.id)
      if (cluster) {
        const rackCount = cluster.racks.length
        const nodeCount = cluster.racks.reduce((sum, rack) => sum + rack.nodes.length, 0)
        const gpuCount = cluster.racks.reduce(
          (sum, rack) => sum + rack.nodes.reduce((gpuSum, node) => gpuSum + node.gpus.length, 0),
          0
        )
        return `Cluster – ${cluster.name} · ${rackCount} racks · ${nodeCount} nodes · ${gpuCount} GPUs`
      }
    }

    let clusterMatch: ClusterSpec | undefined
    let rackMatch: RackSpec | undefined
    let nodeMatch: NodeSpec | undefined
    let gpuMatch: GPUSpec | undefined

    for (const cluster of data.clusters) {
      for (const rack of cluster.racks) {
        for (const node of rack.nodes) {
          if (!nodeMatch && selection.kind === 'node' && node.id === selection.id) {
            clusterMatch = cluster
            rackMatch = rack
            nodeMatch = node
          }
          if (!gpuMatch && selection.kind === 'gpu') {
            const match = node.gpus.find((gpu) => gpu.uuid === selection.id)
            if (match) {
              clusterMatch = cluster
              rackMatch = rack
              nodeMatch = node
              gpuMatch = match
            }
          }
          if (selection.kind === 'memory' && memoryInfo) {
            if (memoryInfo.scope === 'node' && memoryInfo.parentId === node.id) {
              clusterMatch = cluster
              rackMatch = rack
              nodeMatch = node
            }
            if (memoryInfo.scope === 'gpu') {
              const match = node.gpus.find((gpu) => gpu.uuid === memoryInfo.parentId)
              if (match) {
                clusterMatch = cluster
                rackMatch = rack
                nodeMatch = node
                gpuMatch = match
              }
            }
          }
        }
      }
    }

    if (selection.kind === 'rack') {
      const match = data.clusters
        .flatMap((cluster) => cluster.racks.map((rack) => ({ cluster, rack })))
        .find((entry) => entry.rack.id === selection.id)
      if (match) {
        const nodeCount = match.rack.nodes.length
        const gpuCount = match.rack.nodes.reduce((sum, node) => sum + node.gpus.length, 0)
        return `Rack – ${match.rack.name} · ${nodeCount} nodes · ${gpuCount} GPUs · Cluster ${match.cluster.name}`
      }
    }

    if (selection.kind === 'node' && nodeMatch) {
      const gpuCount = nodeMatch.gpus.length
      const totalHbm = nodeMatch.gpus.reduce((sum, gpu) => sum + gpu.memoryGB, 0)
      const gpuModel = nodeMatch.gpus[0]?.model ?? 'GPU'
      return `Node – ${nodeMatch.hostname} · ${nodeMatch.model} · ${gpuCount}× ${gpuModel} (${totalHbm.toFixed(0)} GB HBM3e)`
    }

    if (selection.kind === 'gpu' && gpuMatch) {
      return `GPU – ${gpuMatch.name} (${gpuMatch.model}) · ${gpuMatch.memoryGB} GB HBM3e · NVLink ${gpuMatch.nvlinkTBs.toFixed(1)} TB/s`
    }

    if (selection.kind === 'memory' && memoryInfo) {
      const parentLabel = memoryInfo.scope === 'gpu' && gpuMatch
        ? `${gpuMatch.name} (${gpuMatch.model})`
        : nodeMatch
        ? nodeMatch.hostname
        : undefined
      const suffix = parentLabel ? ` · Parent ${parentLabel}` : ''
      return `Memory – ${memoryInfo.label} (${memoryInfo.type}, ${memoryInfo.capacity})${suffix}`
    }

    if (selection.kind === 'rack' && rackMatch && clusterMatch) {
      const nodeCount = rackMatch.nodes.length
      const gpuCount = rackMatch.nodes.reduce((sum, node) => sum + node.gpus.length, 0)
      return `Rack – ${rackMatch.name} · ${nodeCount} nodes · ${gpuCount} GPUs · Cluster ${clusterMatch.name}`
    }

    return `${selection.kind.toUpperCase()} – ${selection.id}`
  }, [data, memoryInfo, selection])

  useEffect(() => {
    if (view === 'cluster') {
      resetViewport()
    } else {
      fitViewport()
    }
  }, [fitViewport, resetViewport, view])

  if (!data) return null

  return (
    <div className="app">
      <header className="app__header">
        <div className="title-block">
          <h1>Blackwell Physical Explorer</h1>
          <p className="subtitle">DGX B200 defaults · realistic NVLink &amp; InfiniBand capacity</p>
        </div>
        <nav className="breadcrumbs">{breadcrumbNodes}</nav>
      </header>
      <main className="app__main">
        <BlueprintViewport>
          <BlueprintRoot topology={data} />
        </BlueprintViewport>
        <ZoomControls />
        <LeftPanel topology={data} />
        <MemoryBlueprintOverlay />
      </main>
      <footer className="app__footer">
        <p>Selection {selectionLabel}</p>
      </footer>
    </div>
  )
}
