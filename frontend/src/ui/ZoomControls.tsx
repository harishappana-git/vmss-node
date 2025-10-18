import { useCameraTools } from '../lib/camera'

export function ZoomControls() {
  const { zoomIn, zoomOut, zoomHome, zoomFit } = useCameraTools()
  return (
    <div className="zoom-controls">
      <button type="button" onClick={zoomIn} aria-label="Zoom in">
        ＋
      </button>
      <button type="button" onClick={zoomOut} aria-label="Zoom out">
        −
      </button>
      <button type="button" onClick={zoomHome} aria-label="Go home">
        ⟳
      </button>
      <button type="button" onClick={zoomFit} aria-label="Fit view">
        ⤢
      </button>
    </div>
  )
}
