import { create } from 'zustand'
import type {
  Breadcrumb,
  GPUSpec,
  MemoryDescriptor,
  NodeSpec,
  Selection,
  SelectionKind
} from '../types'
import { verboseLog } from '../lib/logging'

type ViewLevel = 'cluster' | 'node' | 'gpu'

type ExplorerState = {
  view: ViewLevel
  selection: Selection | null
  breadcrumbs: Breadcrumb[]
  focusedNodeId?: string
  focusedGpuId?: string
  memoryInfo?: MemoryDescriptor
  memoryBlueprint: MemoryBlueprint | null
  select: (selection: Selection | null, options?: { memoryInfo?: MemoryDescriptor }) => void
  enterNode: (node: NodeSpec, context: { clusterId: string; clusterLabel: string; rackId: string; rackLabel: string }) => void
  enterGpu: (
    gpu: Selection,
    context: {
      node: NodeSpec
      clusterId: string
      clusterLabel: string
      rackId: string
      rackLabel: string
      gpuLabel: string
    }
  ) => void
  goHome: () => void
  goToBreadcrumb: (index: number) => void
  openMemoryBlueprint: (blueprint: MemoryBlueprint) => void
  closeMemoryBlueprint: () => void
}

export type MemoryBlueprint =
  | {
      scope: 'node'
      descriptor: MemoryDescriptor
      node: NodeSpec
      clusterName: string
      rackName: string
    }
  | {
      scope: 'gpu'
      descriptor: MemoryDescriptor
      node: NodeSpec
      gpu: GPUSpec
      clusterName: string
      rackName: string
    }

function makeBreadcrumb(label: string, kind: SelectionKind | 'rack', id: string): Breadcrumb {
  return { label, kind, id }
}

export const useExplorerStore = create<ExplorerState>((set, get) => ({
  view: 'cluster',
  selection: null,
  breadcrumbs: [],
  focusedNodeId: undefined,
  focusedGpuId: undefined,
  memoryInfo: undefined,
  memoryBlueprint: null,
  select: (nextSelection, options) =>
    set((state) => {
      if (nextSelection) {
        verboseLog('selection updated', {
          kind: nextSelection.kind,
          id: nextSelection.id,
          memoryDescriptor: options?.memoryInfo?.label
        })
      } else {
        verboseLog('selection cleared')
      }
      return {
        selection: nextSelection,
        memoryInfo: options?.memoryInfo,
        memoryBlueprint: nextSelection?.kind === 'memory' ? state.memoryBlueprint : null
      }
    }),
  enterNode: (node, context) => {
    verboseLog('entering node view', {
      nodeId: node.id,
      hostname: node.hostname,
      rackId: context.rackId,
      clusterId: context.clusterId
    })
    set({
      view: 'node',
      selection: { kind: 'node', id: node.id },
      focusedNodeId: node.id,
      focusedGpuId: undefined,
      memoryInfo: undefined,
      memoryBlueprint: null,
      breadcrumbs: [
        makeBreadcrumb(context.clusterLabel, 'cluster', context.clusterId),
        makeBreadcrumb(context.rackLabel, 'rack', context.rackId),
        makeBreadcrumb(node.hostname, 'node', node.id)
      ]
    })
  },
  enterGpu: (gpu, context) => {
    verboseLog('entering gpu view', {
      gpuId: gpu.id,
      nodeId: context.node.id,
      clusterId: context.clusterId,
      rackId: context.rackId
    })
    set({
      view: 'gpu',
      selection: { kind: 'gpu', id: gpu.id },
      focusedNodeId: context.node.id,
      focusedGpuId: gpu.id,
      memoryInfo: undefined,
      memoryBlueprint: null,
      breadcrumbs: [
        makeBreadcrumb(context.clusterLabel, 'cluster', context.clusterId),
        makeBreadcrumb(context.rackLabel, 'rack', context.rackId),
        makeBreadcrumb(context.node.hostname, 'node', context.node.id),
        makeBreadcrumb(context.gpuLabel, 'gpu', gpu.id)
      ]
    })
  },
  goHome: () => {
    verboseLog('returning to cluster view')
    set({
      view: 'cluster',
      selection: null,
      focusedNodeId: undefined,
      focusedGpuId: undefined,
      memoryInfo: undefined,
      memoryBlueprint: null,
      breadcrumbs: []
    })
  },
  goToBreadcrumb: (index) => {
    const crumbs = get().breadcrumbs
    const crumb = crumbs[index]
    if (!crumb) return
    if (crumb.kind === 'cluster') {
      verboseLog('navigating via breadcrumb', { target: 'cluster', id: crumb.id })
      set({
        view: 'cluster',
        selection: { kind: 'cluster', id: crumb.id },
        focusedNodeId: undefined,
        focusedGpuId: undefined,
        memoryInfo: undefined,
        memoryBlueprint: null,
        breadcrumbs: crumbs.slice(0, index + 1)
      })
      return
    }
    if (crumb.kind === 'rack') {
      verboseLog('navigating via breadcrumb', { target: 'rack', id: crumb.id })
      set({
        view: 'cluster',
        selection: { kind: 'rack', id: crumb.id },
        focusedNodeId: undefined,
        focusedGpuId: undefined,
        memoryInfo: undefined,
        memoryBlueprint: null,
        breadcrumbs: crumbs.slice(0, index + 1)
      })
      return
    }
    if (crumb.kind === 'node') {
      verboseLog('navigating via breadcrumb', { target: 'node', id: crumb.id })
      set({
        view: 'node',
        selection: { kind: 'node', id: crumb.id },
        focusedNodeId: crumb.id,
        focusedGpuId: undefined,
        memoryInfo: undefined,
        memoryBlueprint: null,
        breadcrumbs: crumbs.slice(0, index + 1)
      })
      return
    }
    if (crumb.kind === 'gpu') {
      const priorNode = [...crumbs.slice(0, index)].reverse().find((item) => item.kind === 'node')
      verboseLog('navigating via breadcrumb', {
        target: 'gpu',
        id: crumb.id,
        nodeId: priorNode?.id
      })
      set({
        view: 'gpu',
        selection: { kind: 'gpu', id: crumb.id },
        focusedNodeId: priorNode?.id,
        focusedGpuId: crumb.id,
        memoryInfo: undefined,
        memoryBlueprint: null,
        breadcrumbs: crumbs.slice(0, index + 1)
      })
    }
  },
  openMemoryBlueprint: (blueprint) => {
    verboseLog('opening memory blueprint', {
      scope: blueprint.scope,
      descriptor: blueprint.descriptor.label
    })
    set({ memoryBlueprint: blueprint })
  },
  closeMemoryBlueprint: () => {
    verboseLog('closing memory blueprint')
    set({ memoryBlueprint: null })
  }
}))
