import { useMemo, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera, StatsGl } from '@react-three/drei'
import { useQuery } from '@tanstack/react-query'
import { PhysicalScene } from './components/PhysicalScene'
import { CudaScene } from './components/CudaScene'
import { useMetricsStream } from './hooks/useMetricsStream'
import { useSelectionStore } from './state/selectionStore'
import { fetchTopology } from './api/client'
import { MetricsOverlay } from './components/MetricsOverlay'
import type { Topology } from './types'

const modes = ['physical', 'cuda'] as const

export default function App() {
  const [mode, setMode] = useState<(typeof modes)[number]>('physical')
  const { data } = useQuery<Topology>({ queryKey: ['topology'], queryFn: fetchTopology })
  useMetricsStream()
  const selected = useSelectionStore((state) => state.selected)

  const scene = useMemo(() => {
    if (!data) return null
    return mode === 'physical' ? <PhysicalScene topology={data} /> : <CudaScene topology={data} />
  }, [data, mode])

  return (
    <div className="app">
      <header className="app__header">
        <h1>3D GPU Cluster &amp; CUDA Explorer</h1>
        <div className="app__controls">
          {modes.map((m) => (
            <button key={m} className={m === mode ? 'active' : ''} onClick={() => setMode(m)}>
              {m.toUpperCase()}
            </button>
          ))}
        </div>
      </header>
      <main className="app__main">
        <Canvas shadows frameloop="always" className="scene-canvas">
          <color attach="background" args={[0.02, 0.02, 0.05]} />
          <hemisphereLight intensity={0.35} groundColor={0x111111} />
          <directionalLight position={[10, 20, 10]} intensity={1.3} castShadow />
          <PerspectiveCamera makeDefault position={[18, 14, 18]} fov={50} />
          <OrbitControls enablePan enableZoom enableRotate />
          {scene}
          <StatsGl className="stats" />
        </Canvas>
        {selected && <MetricsOverlay />}
      </main>
    </div>
  )
}
