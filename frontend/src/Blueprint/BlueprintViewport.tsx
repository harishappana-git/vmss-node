import { useCallback, useRef, useState } from 'react'
import { useViewportStore } from '../lib/viewport'

type BlueprintViewportProps = {
  children: React.ReactNode
}

export function BlueprintViewport({ children }: BlueprintViewportProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const scale = useViewportStore((state) => state.scale)
  const translate = useViewportStore((state) => state.translate)
  const setTranslate = useViewportStore((state) => state.setTranslate)
  const applyZoom = useViewportStore((state) => state.applyZoom)
  const [isPanning, setIsPanning] = useState(false)
  const [panStart, setPanStart] = useState<{ x: number; y: number } | null>(null)
  const [dragged, setDragged] = useState(false)

  const handlePointerDown = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId)
    setIsPanning(true)
    setDragged(false)
    setPanStart({ x: event.clientX - translate.x, y: event.clientY - translate.y })
  }, [translate.x, translate.y])

  const handlePointerMove = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (!isPanning || !panStart) return
    const nextTranslate = { x: event.clientX - panStart.x, y: event.clientY - panStart.y }
    if (Math.abs(nextTranslate.x - translate.x) > 2 || Math.abs(nextTranslate.y - translate.y) > 2) {
      setDragged(true)
    }
    setTranslate(nextTranslate)
  }, [isPanning, panStart, setTranslate, translate.x, translate.y])

  const handlePointerUp = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
    setIsPanning(false)
    setPanStart(null)
  }, [])

  const handleWheel = useCallback((event: React.WheelEvent<HTMLDivElement>) => {
    event.preventDefault()
    const container = containerRef.current
    if (!container) return
    const rect = container.getBoundingClientRect()
    const origin = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top
    }
    const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12
    applyZoom(factor, origin)
  }, [applyZoom])

  return (
    <div
      ref={containerRef}
      className="blueprint-viewport"
      onWheel={handleWheel}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerUp}
      onClickCapture={(event) => {
        if (dragged) {
          event.stopPropagation()
          event.preventDefault()
          setDragged(false)
        }
      }}
    >
      <div
        className={`blueprint-viewport__content${isPanning ? ' is-panning' : ''}`}
        style={{ transform: `translate(${translate.x}px, ${translate.y}px) scale(${scale})` }}
      >
        {children}
      </div>
    </div>
  )
}
