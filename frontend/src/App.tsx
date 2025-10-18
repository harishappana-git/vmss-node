import { useEffect, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useExplorerStore } from './state/selectionStore'
import { fetchTopology } from './api/client'
import type { Topology } from './types'
import { ZoomControls } from './ui/ZoomControls'
import { LeftPanel } from './Panels/LeftPanel'
import { useMetricsStream } from './hooks/useMetricsStream'
import { MemoryBlueprintOverlay } from './ui/MemoryBlueprintOverlay'
import { BlueprintViewport } from './Blueprint/BlueprintViewport'
import { BlueprintRoot } from './Blueprint/BlueprintRoot'
import { useViewportStore } from './lib/viewport'

export default function App() {
  const { data } = useQuery<Topology>({ queryKey: ['topology'], queryFn: fetchTopology })
  useMetricsStream()
  const breadcrumbs = useExplorerStore((state) => state.breadcrumbs)
  const goHome = useExplorerStore((state) => state.goHome)
  const goToBreadcrumb = useExplorerStore((state) => state.goToBreadcrumb)
  const selection = useExplorerStore((state) => state.selection)
  const memoryInfo = useExplorerStore((state) => state.memoryInfo)
  const view = useExplorerStore((state) => state.view)
  const resetViewport = useViewportStore((state) => state.reset)
  const fitViewport = useViewportStore((state) => state.fit)

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

  useEffect(() => {
    if (view === 'cluster') {
      resetViewport()
    } else {
      fitViewport()
    }
  }, [fitViewport, resetViewport, view])

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
        <BlueprintViewport>
          <BlueprintRoot topology={data} />
        </BlueprintViewport>
        <ZoomControls />
        <LeftPanel topology={data} />
        <MemoryBlueprintOverlay />
      </main>
      <footer className="app__footer">
        <p>
          Selection:{' '}
          {selection
            ? selection.kind === 'memory' && memoryInfo
              ? `${selection.kind.toUpperCase()} – ${memoryInfo.label}`
              : `${selection.kind.toUpperCase()} – ${selection.id}`
            : 'None'}
        </p>
      </footer>
    </div>
  )
}
