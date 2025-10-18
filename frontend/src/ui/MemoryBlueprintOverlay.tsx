import { useMemo, type MouseEvent } from 'react'
import { useExplorerStore, type MemoryBlueprint } from '../state/selectionStore'

function NodeMemoryDiagram({ blueprint }: { blueprint: Extract<MemoryBlueprint, { scope: 'node' }> }) {
  const dimmMatch = blueprint.descriptor.id.match(/dimm(\d+)/i)
  const highlightIndex = dimmMatch ? Number.parseInt(dimmMatch[1], 10) - 1 : undefined
  const leftDimms = useMemo(() => Array.from({ length: 8 }, (_, idx) => idx), [])
  const rightDimms = useMemo(() => Array.from({ length: 8 }, (_, idx) => idx + 8), [])

  return (
    <div className="memory-blueprint__diagram memory-blueprint__diagram--node">
      <div className="node-blueprint__column">
        <div className="node-blueprint__cpu">CPU 0<br />Intel Xeon 8570<br />56 cores</div>
        <div className="node-blueprint__dimms">
          {leftDimms.map((index) => (
            <div
              key={`left-${index}`}
              className={`node-blueprint__dimm${highlightIndex === index ? ' is-active' : ''}`}
            >
              DIMM {index + 1}
            </div>
          ))}
        </div>
      </div>
      <div className="node-blueprint__center">
        <div className="node-blueprint__nvlink">NVSwitch Hub<br />Gen5 · 14.4&nbsp;TB/s</div>
        <div className="node-blueprint__gpus">
          {blueprint.node.gpus.map((gpu) => (
            <div key={gpu.uuid} className="node-blueprint__gpu">
              {gpu.name}
              <span>{gpu.memoryGB} GB HBM3e</span>
            </div>
          ))}
        </div>
        <div className="node-blueprint__io">4× ConnectX-7 · 400 Gb/s<br />BlueField-3 DPUs</div>
      </div>
      <div className="node-blueprint__column">
        <div className="node-blueprint__cpu">CPU 1<br />Intel Xeon 8570<br />56 cores</div>
        <div className="node-blueprint__dimms">
          {rightDimms.map((index) => (
            <div
              key={`right-${index}`}
              className={`node-blueprint__dimm${highlightIndex === index ? ' is-active' : ''}`}
            >
              DIMM {index + 1}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function GpuMemoryDiagram({ blueprint }: { blueprint: Extract<MemoryBlueprint, { scope: 'gpu' }> }) {
  const hbmMatch = blueprint.descriptor.id.match(/hbm(\d+)/i)
  const highlightedStack = hbmMatch ? Number.parseInt(hbmMatch[1], 10) - 1 : undefined
  const highlightL2 = blueprint.descriptor.id.includes(':l2')
  const highlightShared = blueprint.descriptor.id.includes(':shared')
  const highlightRegisters = blueprint.descriptor.id.includes(':registers')
  const highlightTma = blueprint.descriptor.id.includes(':tma')

  return (
    <div className="memory-blueprint__diagram memory-blueprint__diagram--gpu">
      <div className="gpu-blueprint__board">
        <div className={`gpu-blueprint__die${highlightL2 || highlightShared || highlightRegisters || highlightTma ? ' is-active' : ''}`}>
          <div className={`gpu-blueprint__layer gpu-blueprint__layer--l2${highlightL2 ? ' is-active' : ''}`}>L2 Cache Ring</div>
          <div className={`gpu-blueprint__layer gpu-blueprint__layer--shared${highlightShared ? ' is-active' : ''}`}>
            Shared / L1 per SM
          </div>
          <div className={`gpu-blueprint__layer gpu-blueprint__layer--registers${highlightRegisters ? ' is-active' : ''}`}>
            Register File
          </div>
          <div className={`gpu-blueprint__layer gpu-blueprint__layer--tma${highlightTma ? ' is-active' : ''}`}>TMA Staging</div>
        </div>
        <div className="gpu-blueprint__hbms">
          {Array.from({ length: 8 }, (_, idx) => (
            <div
              key={`hbm-${idx}`}
              className={`gpu-blueprint__hbm${highlightedStack === idx ? ' is-active' : ''}`}
            >
              HBM {idx + 1}
            </div>
          ))}
        </div>
        <div className="gpu-blueprint__nvlink">NVLink 5 Pads<br />1.8&nbsp;TB/s per GPU</div>
      </div>
      <dl className="gpu-blueprint__meta">
        <div>
          <dt>HBM3e Total</dt>
          <dd>{blueprint.gpu.memoryGB} GB · {blueprint.gpu.hbmBandwidthTBs.toFixed(1)} TB/s</dd>
        </div>
        <div>
          <dt>SM Shared/L1</dt>
          <dd>≈256 KB per SM configurable</dd>
        </div>
        <div>
          <dt>Register File</dt>
          <dd>≈256 KB per SM · single-cycle</dd>
        </div>
        <div>
          <dt>TMA</dt>
          <dd>Async tensor copy fabric</dd>
        </div>
      </dl>
    </div>
  )
}

export function MemoryBlueprintOverlay() {
  const blueprint = useExplorerStore((state) => state.memoryBlueprint)
  const close = useExplorerStore((state) => state.closeMemoryBlueprint)

  if (!blueprint) return null

  const handleBackdropClick = (event: MouseEvent<HTMLDivElement>) => {
    if (event.target === event.currentTarget) {
      close()
    }
  }

  return (
    <div className="memory-blueprint-overlay" role="dialog" aria-modal="true" onClick={handleBackdropClick}>
      <div className="memory-blueprint">
        <header className="memory-blueprint__header">
          <div>
            <h2>{blueprint.scope === 'node' ? 'DGX B200 System Memory Blueprint' : 'B200 Memory Hierarchy Blueprint'}</h2>
            <p className="memory-blueprint__subtitle">
              {blueprint.clusterName} · {blueprint.rackName} · {blueprint.node.hostname}
              {blueprint.scope === 'gpu' ? ` · ${blueprint.gpu.name}` : ''}
            </p>
          </div>
          <button type="button" onClick={close} className="memory-blueprint__close" aria-label="Close memory blueprint">
            ×
          </button>
        </header>
        {blueprint.scope === 'node' ? <NodeMemoryDiagram blueprint={blueprint} /> : <GpuMemoryDiagram blueprint={blueprint} />}
        <section className="memory-blueprint__details">
          <h3>{blueprint.descriptor.label}</h3>
          <p>{blueprint.descriptor.description}</p>
          <ul>
            <li>
              <strong>Type:</strong> {blueprint.descriptor.type}
            </li>
            <li>
              <strong>Capacity:</strong> {blueprint.descriptor.capacity}
            </li>
            {blueprint.descriptor.bandwidth && (
              <li>
                <strong>Bandwidth:</strong> {blueprint.descriptor.bandwidth}
              </li>
            )}
            {blueprint.scope === 'gpu' && (
              <li>
                <strong>NVLink:</strong> {blueprint.gpu.nvlinkTBs.toFixed(1)} TB/s to NVSwitch fabric
              </li>
            )}
          </ul>
        </section>
        <footer className="memory-blueprint__footer">Double-click another component or press close to exit the 2D blueprint view.</footer>
      </div>
    </div>
  )
}
