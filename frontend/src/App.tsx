import { useMemo, useRef } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera, StatsGl } from '@react-three/drei'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { useQuery } from '@tanstack/react-query'
import { useExplorerStore } from './state/selectionStore'
import { fetchTopology } from './api/client'
import type { Topology } from './types'
import { SceneRoot } from './Scene/SceneRoot'
import { ZoomControls } from './ui/ZoomControls'
import { LeftPanel } from './Panels/LeftPanel'
import { useMetricsStream } from './hooks/useMetricsStream'
import { CameraRig } from './lib/camera'

export default function App() {
  const { data } = useQuery<Topology>({ queryKey: ['topology'], queryFn: fetchTopology })
  useMetricsStream()
  const breadcrumbs = useExplorerStore((state) => state.breadcrumbs)
  const goHome = useExplorerStore((state) => state.goHome)
  const goToBreadcrumb = useExplorerStore((state) => state.goToBreadcrumb)
  const selection = useExplorerStore((state) => state.selection)
  const controlsRef = useRef<OrbitControlsImpl>(null)

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
        <Canvas shadows frameloop="always" className="scene-canvas">
          <color attach="background" args={[0.02, 0.02, 0.05]} />
          <hemisphereLight intensity={0.45} groundColor={0x111111} />
          <directionalLight position={[18, 28, 18]} intensity={1.4} castShadow />
          <PerspectiveCamera makeDefault position={[32, 26, 32]} fov={45} />
          <OrbitControls ref={controlsRef} enablePan enableZoom enableRotate />
          <CameraRig controls={controlsRef} />
          <SceneRoot topology={data} />
          <StatsGl className="stats" />
        </Canvas>
        <ZoomControls />
        <LeftPanel topology={data} />
      </main>
      <footer className="app__footer">
        <p>Selection: {selection ? `${selection.kind.toUpperCase()} – ${selection.id}` : 'None'}</p>
      </footer>
    </div>
  )
}
