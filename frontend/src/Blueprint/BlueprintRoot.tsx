import type { Topology } from '../types'
import { useExplorerStore } from '../state/selectionStore'
import { ClusterBlueprint } from './ClusterBlueprint'
import { NodeBlueprint } from './NodeBlueprint'
import { GpuBlueprint } from './GpuBlueprint'

export function BlueprintRoot({ topology }: { topology: Topology }) {
  const view = useExplorerStore((state) => state.view)
  const focusedNodeId = useExplorerStore((state) => state.focusedNodeId)
  const focusedGpuId = useExplorerStore((state) => state.focusedGpuId)

  if (view === 'cluster') {
    return <ClusterBlueprint topology={topology} />
  }

  for (const cluster of topology.clusters) {
    for (const rack of cluster.racks) {
      for (const node of rack.nodes) {
        if (view === 'node' && node.id === focusedNodeId) {
          return <NodeBlueprint key={node.id} node={node} cluster={cluster} rack={rack} />
        }
        if (view === 'gpu' && node.id === focusedNodeId) {
          const gpu = node.gpus.find((item) => item.uuid === focusedGpuId)
          if (gpu) {
            return <GpuBlueprint key={gpu.uuid} node={node} gpu={gpu} cluster={cluster} rack={rack} />
          }
        }
      }
    }
  }

  return <ClusterBlueprint topology={topology} />
}
