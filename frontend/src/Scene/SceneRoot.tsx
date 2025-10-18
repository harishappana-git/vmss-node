import { useEffect, useMemo } from 'react'
import type { Topology } from '../types'
import { ClusterScene } from './ClusterScene'
import { NodeScene } from './NodeScene'
import { GPUInternals } from './GPUInternals'
import { useExplorerStore } from '../state/selectionStore'
import { focusOn } from '../lib/camera'

type SceneRootProps = {
  topology: Topology
}

export function SceneRoot({ topology }: SceneRootProps) {
  const view = useExplorerStore((state) => state.view)
  const focusedNodeId = useExplorerStore((state) => state.focusedNodeId)
  const focusedGpuId = useExplorerStore((state) => state.focusedGpuId)

  useEffect(() => {
    if (view === 'cluster') {
      focusOn([32, 26, 32], [0, 0, 0], 45)
    } else if (view === 'node') {
      focusOn([14, 10, 14], [0, 0, 0], 42)
    } else if (view === 'gpu') {
      focusOn([8, 6, 8], [0, 0, 0], 38)
    }
  }, [view, focusedNodeId, focusedGpuId])

  const focus = useMemo(() => {
    if (!focusedNodeId) return null
    for (const cluster of topology.clusters) {
      for (const rack of cluster.racks) {
        const node = rack.nodes.find((n) => n.id === focusedNodeId)
        if (node) {
          return { cluster, rack, node }
        }
      }
    }
    return null
  }, [focusedNodeId, topology])

  if (view === 'node' && focus) {
    return <NodeScene node={focus.node} cluster={focus.cluster} rack={focus.rack} />
  }

  if (view === 'gpu' && focus && focusedGpuId) {
    const gpu = focus.node.gpus.find((g) => g.uuid === focusedGpuId)
    if (gpu) {
      return <GPUInternals gpu={gpu} node={focus.node} cluster={focus.cluster} rack={focus.rack} />
    }
  }

  return <ClusterScene topology={topology} />
}
