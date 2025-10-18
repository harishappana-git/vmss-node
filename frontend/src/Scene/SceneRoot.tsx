import { useMemo } from 'react'
import type { Topology } from '../types'
import { ClusterScene } from './ClusterScene'
import { NodeScene } from './NodeScene'
import { GPUInternals } from './GPUInternals'
import { useExplorerStore } from '../state/selectionStore'

type SceneRootProps = {
  topology: Topology
}

export function SceneRoot({ topology }: SceneRootProps) {
  const view = useExplorerStore((state) => state.view)
  const focusedNodeId = useExplorerStore((state) => state.focusedNodeId)
  const focusedGpuId = useExplorerStore((state) => state.focusedGpuId)

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
