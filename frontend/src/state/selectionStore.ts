import { create } from 'zustand'
import type {
  Breadcrumb,
  GPUSpec,
  MemoryDescriptor,
  NodeSpec,
  Selection,
  SelectionKind
} from '../types'

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
    set((state) => ({
      selection: nextSelection,
      memoryInfo: options?.memoryInfo,
      memoryBlueprint: nextSelection?.kind === 'memory' ? state.memoryBlueprint : null
    })),
  enterNode: (node, context) => {
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
  openMemoryBlueprint: (blueprint) => set({ memoryBlueprint: blueprint }),
  closeMemoryBlueprint: () => set({ memoryBlueprint: null })
}))
