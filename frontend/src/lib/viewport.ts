import { create } from 'zustand'

type Point = { x: number; y: number }

type ViewportState = {
  scale: number
  translate: Point
  setTranslate: (translate: Point) => void
  applyZoom: (factor: number, origin: Point) => void
  reset: () => void
  fit: () => void
}

const MIN_SCALE = 0.4
const MAX_SCALE = 2.8

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max)
}

export const useViewportStore = create<ViewportState>((set, get) => ({
  scale: 1,
  translate: { x: 0, y: 0 },
  setTranslate: (translate) => set({ translate }),
  applyZoom: (factor, origin) => {
    const { scale, translate } = get()
    const nextScale = clamp(scale * factor, MIN_SCALE, MAX_SCALE)
    const clampedFactor = nextScale / scale
    const offsetX = origin.x - translate.x
    const offsetY = origin.y - translate.y
    const nextTranslate: Point = {
      x: origin.x - offsetX * clampedFactor,
      y: origin.y - offsetY * clampedFactor
    }
    set({ scale: nextScale, translate: nextTranslate })
  },
  reset: () => set({ scale: 1, translate: { x: 0, y: 0 } }),
  fit: () => set({ scale: 0.9, translate: { x: 20, y: 20 } })
}))

export function useViewportTools() {
  const applyZoom = useViewportStore((state) => state.applyZoom)
  const reset = useViewportStore((state) => state.reset)
  const fit = useViewportStore((state) => state.fit)

  return {
    zoomIn: () => applyZoom(1.2, { x: window.innerWidth / 2, y: window.innerHeight / 2 }),
    zoomOut: () => applyZoom(1 / 1.2, { x: window.innerWidth / 2, y: window.innerHeight / 2 }),
    zoomHome: reset,
    zoomFit: fit
  }
}
