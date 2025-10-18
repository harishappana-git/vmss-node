import { create } from 'zustand'
import type { GPUFrame, LinkFrame, NodeFrame } from '../types'

type FrameMessage = {
  topic: string
  t: number
  data: Record<string, unknown>
}

type MetricsState = {
  gpu: Record<string, GPUFrame>
  node: Record<string, NodeFrame>
  link: Record<string, LinkFrame>
  ingest: (payload: FrameMessage) => void
}

function normalizeGPU(topic: string) {
  return topic.replace(/^gpu\./, '')
}

function normalizeNode(topic: string) {
  return topic.replace(/^node\./, '')
}

function normalizeLink(topic: string) {
  return topic.replace(/^link\./, '')
}

export const useMetricsStore = create<MetricsState>((set) => ({
  gpu: {},
  node: {},
  link: {},
  ingest: (payload) =>
    set((state) => {
      if (payload.topic.startsWith('gpu.')) {
        const id = normalizeGPU(payload.topic)
        return {
          ...state,
          gpu: { ...state.gpu, [id]: { ...(payload.data as GPUFrame), t: payload.t } }
        }
      }
      if (payload.topic.startsWith('node.')) {
        const id = normalizeNode(payload.topic)
        return {
          ...state,
          node: { ...state.node, [id]: { ...(payload.data as NodeFrame), t: payload.t } }
        }
      }
      if (payload.topic.startsWith('link.')) {
        const id = normalizeLink(payload.topic)
        return {
          ...state,
          link: { ...state.link, [id]: { ...(payload.data as LinkFrame), t: payload.t } }
        }
      }
      return state
    })
}))
