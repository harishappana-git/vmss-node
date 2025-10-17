import React, { useEffect, useMemo, useState } from 'react';
import TopologyScene from './components/TopologyScene.jsx';
import { buildTopology, deriveMetrics } from './topology.js';

const formatNumber = (value) => {
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

  useEffect(() => {
    setFocusPath([topology.id]);
  }, [topology.id]);

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

  const focusDepth = focusPath.length - 1;

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

  return (
    <div className="app-shell">
      <header>
        <h1>AI Distributed Training Stack Explorer</h1>
        <p style={{ margin: '0.35rem 0 0', color: '#8b949e', maxWidth: '720px' }}>
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
          <div style={{ marginTop: '1.5rem', fontSize: '0.8rem', color: '#6e7681' }}>
            Tip: start at the distributed system level and progressively drill into servers, GPUs, grids, blocks, warps,
            and threads to understand how CUDA maps work to hardware.
          </div>
        </aside>
        <div className="canvas-wrapper">
          <div className="canvas-overlay">
            <h3>Focus</h3>
            <p style={{ marginBottom: '0.5rem' }}>
              {breadcrumbs.map((crumb, index) => (
                <React.Fragment key={crumb.id}>
                  {index > 0 && <span style={{ margin: '0 0.25rem' }}>›</span>}
                  <button
                    type="button"
                    onClick={() => handleBreadcrumbClick(crumb.id)}
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: index === breadcrumbs.length - 1 ? '#58a6ff' : '#e6edf3',
                      cursor: 'pointer',
                      padding: 0,
                      font: 'inherit'
                    }}
                  >
                    {crumb.name}
                  </button>
                </React.Fragment>
              ))}
            </p>
            {activeDescription.base && <p style={{ marginBottom: '0.5rem' }}>{activeDescription.base}</p>}
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
          <TopologyScene
            topology={topology}
            activeNode={activeNode}
            focusDepth={focusDepth}
            onSelectNode={handleSelectNode}
          />
        </div>
      </div>
      <footer className="footer">
        Built with React, Vite, and react-three-fiber/Three.js to emphasise spatial reasoning about distributed AI
        training infrastructure.
      </footer>
    </div>
  );
};

export default App;
