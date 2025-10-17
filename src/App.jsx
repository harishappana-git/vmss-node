import React, { useEffect, useMemo, useState } from 'react';
import TopologyScene from './components/TopologyScene.jsx';
import SunburstView from './components/SunburstView.jsx';
import { buildTopology, deriveMetrics, topologyToCsv, LEVEL_ORDER } from './topology.js';

const formatNumber = (value) => {
  if (value >= 1_000_000_000_000) return `${(value / 1_000_000_000_000).toFixed(1)}T`;
  if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(1)}B`;
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return value.toString();
};

const INITIAL_CONFIG = {
  datasetSize: 4096,
  batchSize: 128,
  gpus: 4,
  nodes: 2
};

const App = () => {
  const [config, setConfig] = useState(INITIAL_CONFIG);
  const topology = useMemo(() => buildTopology(config), [config]);
  const [focusPath, setFocusPath] = useState([topology.id]);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [viewMode, setViewMode] = useState('spatial');

  useEffect(() => {
    setFocusPath([topology.id]);
  }, [topology]);

  useEffect(() => {
    setHoveredNode(null);
  }, [viewMode, focusPath.join('>')]);

  const activeNode = useMemo(() => {
    let current = topology;
    for (let i = 1; i < focusPath.length; i += 1) {
      const next = current.children?.find((child) => child.id === focusPath[i]);
      if (next) {
        current = next;
      } else {
        break;
      }
    }
    return current;
  }, [focusPath, topology]);

  const metrics = useMemo(() => deriveMetrics(config), [config]);

  const breadcrumbs = useMemo(() => {
    const crumbs = [{ id: topology.id, name: topology.name }];
    let current = topology;
    focusPath.slice(1).forEach((id) => {
      const next = current.children?.find((child) => child.id === id);
      if (next) {
        crumbs.push({ id: next.id, name: next.name });
        current = next;
      }
    });
    return crumbs;
  }, [focusPath, topology]);

  const activeDescription = useMemo(() => {
    const base = activeNode.meta?.explanation ?? '';
    const details = Object.entries(activeNode.meta ?? {})
      .filter(([key]) => key !== 'explanation')
      .map(([key, value]) => ({ key, value }));
    return { base, details };
  }, [activeNode]);

  const handleInputChange = (field, value) => {
    setConfig((prev) => ({
      ...prev,
      [field]: Number.isNaN(Number(value)) ? prev[field] : Math.max(1, Math.round(Number(value)))
    }));
  };

  const handleSelectNode = (node) => {
    if (!node || node.id === activeNode.id) return;
    setFocusPath((prev) => {
      const exists = prev.includes(node.id);
      if (exists) {
        return prev.slice(0, prev.indexOf(node.id) + 1);
      }
      return [...prev, node.id];
    });
  };

  const handleBreadcrumbClick = (crumbId) => {
    setFocusPath((prev) => {
      const index = prev.indexOf(crumbId);
      if (index === -1) return prev;
      return prev.slice(0, index + 1);
    });
  };

  const handleResetView = () => {
    setFocusPath([topology.id]);
  };

  const handleExportCsv = () => {
    const csv = topologyToCsv(topology);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'ai-distributed-topology.csv';
    link.click();
    URL.revokeObjectURL(link.href);
  };

  return (
    <div className="app-shell">
      <header>
        <h1>AI Distributed Training Stack Explorer</h1>
        <p className="intro">
          Interactively inspect how dataset sizing, GPU count, and distributed infrastructure shape the CUDA execution
          hierarchy. Click components to zoom into finer-grained levels such as grids, blocks, warps, and threads.
        </p>
      </header>
      <div className="main-content">
        <aside className="controls-panel">
          <h2>Simulation Inputs</h2>
          <label htmlFor="datasetSize">Dataset size (samples)</label>
          <input
            id="datasetSize"
            type="number"
            min="1"
            value={config.datasetSize}
            onChange={(event) => handleInputChange('datasetSize', event.target.value)}
          />
          <label htmlFor="batchSize">Batch size</label>
          <input
            id="batchSize"
            type="number"
            min="1"
            value={config.batchSize}
            onChange={(event) => handleInputChange('batchSize', event.target.value)}
          />
          <label htmlFor="gpus">Total GPUs</label>
          <input
            id="gpus"
            type="number"
            min="1"
            value={config.gpus}
            onChange={(event) => handleInputChange('gpus', event.target.value)}
          />
          <label htmlFor="nodes">Server nodes</label>
          <input
            id="nodes"
            type="number"
            min="1"
            value={config.nodes}
            onChange={(event) => handleInputChange('nodes', event.target.value)}
          />

          <div className="metrics">
            <h3>Derived topology</h3>
            <ul>
              <li>
                <span>Servers</span>
                <span>{metrics.nodes}</span>
              </li>
              <li>
                <span>GPUs</span>
                <span>{metrics.gpus}</span>
              </li>
              <li>
                <span>Grids</span>
                <span>{metrics.grids}</span>
              </li>
              <li>
                <span>Blocks</span>
                <span>{metrics.blocks}</span>
              </li>
              <li>
                <span>Warps</span>
                <span>{metrics.warps}</span>
              </li>
              <li>
                <span>Threads</span>
                <span>{formatNumber(metrics.threads)}</span>
              </li>
            </ul>
          </div>

          <div className="performance">
            <h3>Performance snapshot</h3>
            <ul>
              <li>
                <span>Throughput</span>
                <span>{formatNumber(metrics.flops)} FLOPs/s</span>
              </li>
              <li>
                <span>Memory</span>
                <span>{formatNumber(metrics.memoryGb)} GB</span>
              </li>
              <li>
                <span>Interconnect</span>
                <span>{formatNumber(metrics.bandwidthGb)} GB/s</span>
              </li>
              <li>
                <span>Epoch time</span>
                <span>{metrics.epochSeconds.toFixed(2)} s</span>
              </li>
            </ul>
          </div>

          <button type="button" className="export-button" onClick={handleExportCsv}>
            Export topology CSV
          </button>
        </aside>
        <div className="visual-wrapper">
          <div className="visual-header">
            <div className="breadcrumb">
              {breadcrumbs.map((crumb, index) => (
                <React.Fragment key={crumb.id}>
                  {index > 0 && <span className="crumb-sep">›</span>}
                  <button
                    type="button"
                    onClick={() => handleBreadcrumbClick(crumb.id)}
                    className={`crumb ${index === breadcrumbs.length - 1 ? 'is-active' : ''}`}
                  >
                    {crumb.name}
                  </button>
                </React.Fragment>
              ))}
            </div>
            <div className="visual-actions">
              <div className="view-toggle" role="tablist" aria-label="Visualisation mode">
                <button
                  type="button"
                  className={viewMode === 'spatial' ? 'is-selected' : ''}
                  onClick={() => setViewMode('spatial')}
                >
                  3D spatial
                </button>
                <button
                  type="button"
                  className={viewMode === 'sunburst' ? 'is-selected' : ''}
                  onClick={() => setViewMode('sunburst')}
                >
                  Sunburst
                </button>
              </div>
              <button type="button" className="reset-button" onClick={handleResetView}>
                Show full system
              </button>
            </div>
          </div>
          <div className="context-outline">
            {LEVEL_ORDER.map((level) => {
              const label = level
                .replace('distributed-system', 'System')
                .replace(/-/g, ' ')
                .replace(/\b\w/g, (char) => char.toUpperCase());
              const crumb = breadcrumbs.find((item) => item.id.includes(level));
              return (
                <div key={level} className={`context-item ${crumb ? 'is-present' : ''}`}>
                  <span className="context-dot" data-level={level} />
                  <span className="context-label">{label}</span>
                  <span className="context-value">{crumb ? crumb.name : '—'}</span>
                </div>
              );
            })}
          </div>
          <div className={`canvas-wrapper ${viewMode}`}>
            {viewMode === 'spatial' ? (
              <TopologyScene
                topology={topology}
                focusPath={focusPath}
                onSelectNode={handleSelectNode}
                onHoverNode={setHoveredNode}
              />
            ) : (
              <SunburstView
                topology={topology}
                focusPath={focusPath}
                onSelectNode={handleSelectNode}
                onHoverNode={setHoveredNode}
              />
            )}
            <div className="canvas-overlay">
              <h3>Focus details</h3>
              {activeDescription.base && <p>{activeDescription.base}</p>}
              {activeDescription.details.length > 0 && (
                <ul>
                  {activeDescription.details.map((item) => (
                    <li key={item.key}>
                      <strong>{item.key}</strong>: {item.value}
                    </li>
                  ))}
                </ul>
              )}
            </div>
            {hoveredNode && (
              <div className="hover-card">
                <h4>{hoveredNode.name}</h4>
                <p>{hoveredNode.meta?.explanation}</p>
                <ul>
                  {Object.entries(hoveredNode.meta ?? {})
                    .filter(([key]) => key !== 'explanation')
                    .map(([key, value]) => (
                      <li key={key}>
                        <strong>{key}</strong>: {value}
                      </li>
                    ))}
                </ul>
              </div>
            )}
          </div>
        </div>
        <aside className="legend-panel">
          <section>
            <h2>Legend</h2>
            <ul className="legend-list">
              <li data-level="server">Servers (blue)</li>
              <li data-level="device">GPUs (green)</li>
              <li data-level="grid">CUDA grids (orange)</li>
              <li data-level="block">Blocks (yellow)</li>
              <li data-level="warp">Warps (purple)</li>
              <li data-level="thread">Threads (gray)</li>
            </ul>
          </section>
          <section>
            <h2>Concept glossary</h2>
            <dl className="glossary">
              <dt>Grid</dt>
              <dd>Collection of thread blocks launched for a CUDA kernel invocation.</dd>
              <dt>Block</dt>
              <dd>Group of threads that share fast memory and synchronize on an SM.</dd>
              <dt>Warp</dt>
              <dd>32 threads executed in lock-step by the SIMT scheduler.</dd>
              <dt>Thread</dt>
              <dd>Smallest unit of CUDA execution, typically handling one data element.</dd>
            </dl>
          </section>
        </aside>
      </div>
      <footer className="footer">
        Built with React, Vite, and react-three-fiber/Three.js. Use the controls to map distributed infrastructure to CUDA
        execution and export the derived hierarchy for offline analysis.
      </footer>
    </div>
  );
};

export default App;
