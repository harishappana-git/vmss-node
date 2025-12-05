import React, { useCallback, useEffect, useMemo, useState } from 'react';
import TopologyScene from './components/TopologyScene.jsx';
import SunburstView from './components/SunburstView.jsx';
import {
  buildTopology,
  deriveMetrics,
  topologyToCsv,
  LEVEL_ORDER,
  LEVEL_LABELS,
  findNodeById
} from './topology.js';

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

const LEVEL_EXPORT_OPTIONS = LEVEL_ORDER.map((level) => ({
  value: level,
  label: LEVEL_LABELS[level] ?? level
}));

const MAX_COMPARISON_ITEMS = 4;

const GLOSSARY = {
  'Distributed system': 'End-to-end training stack spanning servers, interconnect, and CUDA hierarchy.',
  Server: 'Host machine coordinating GPUs and handling data ingestion.',
  GPU: 'Accelerator device executing CUDA kernels across streaming multiprocessors.',
  Grid: 'CUDA launch dimension containing many thread blocks for a kernel.',
  Block: 'Thread group sharing fast memory and synchronization primitives.',
  Warp: 'SIMT execution group of 32 threads scheduled together.',
  Thread: 'Finest-grained CUDA execution lane operating on individual data.'
};

const getPathToNode = (node, targetId, trail = []) => {
  if (!node) return null;
  const nextTrail = [...trail, node.id];
  if (node.id === targetId) return nextTrail;
  for (const child of node.children ?? []) {
    const path = getPathToNode(child, targetId, nextTrail);
    if (path) return path;
  }
  return null;
};

const collectMetaPairs = (meta) =>
  Object.entries(meta ?? {})
    .filter(([key]) => key !== 'explanation')
    .map(([key, value]) => ({ key, value }));

const App = () => {
  const [config, setConfig] = useState(INITIAL_CONFIG);
  const topology = useMemo(() => buildTopology(config), [config]);
  const [focusPath, setFocusPath] = useState([topology.id]);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [viewMode, setViewMode] = useState('spatial');
  const [showTablePane, setShowTablePane] = useState(true);
  const [comparisonIds, setComparisonIds] = useState([]);
  const [exportTargetLevel, setExportTargetLevel] = useState('device');

  useEffect(() => {
    setFocusPath([topology.id]);
  }, [topology]);

  useEffect(() => {
    setHoveredNode(null);
  }, [viewMode, focusPath.join('>')]);

  useEffect(() => {
    setComparisonIds((prev) => prev.filter((id) => findNodeById(topology, id)));
  }, [topology]);

  const focusNodes = useMemo(() => {
    const nodes = [topology];
    let current = topology;
    focusPath.slice(1).forEach((id) => {
      const next = current.children?.find((child) => child.id === id);
      if (next) {
        nodes.push(next);
        current = next;
      }
    });
    return nodes;
  }, [focusPath, topology]);

  const activeNode = focusNodes[focusNodes.length - 1];

  const metrics = useMemo(() => deriveMetrics(config), [config]);

  const breadcrumbs = useMemo(
    () => focusNodes.map((node) => ({ id: node.id, name: node.name })),
    [focusNodes]
  );

  const activeDescription = useMemo(
    () => ({
      base: activeNode.meta?.explanation ?? '',
      details: collectMetaPairs(activeNode.meta)
    }),
    [activeNode]
  );

  const dataFlowTrail = useMemo(
    () =>
      focusNodes.map((node, index) => ({
        node,
        id: node.id,
        name: node.name,
        type: node.type,
        order: index,
        level: LEVEL_LABELS[node.type] ?? node.type,
        meta: collectMetaPairs(node.meta)
      })),
    [focusNodes]
  );

  const tableRows = useMemo(
    () =>
      (activeNode.children ?? []).map((child) => ({
        node: child,
        id: child.id,
        name: child.name,
        type: child.type,
        level: LEVEL_LABELS[child.type] ?? child.type,
        meta: collectMetaPairs(child.meta)
      })),
    [activeNode]
  );

  const comparisonDetails = useMemo(
    () =>
      comparisonIds
        .map((id) => findNodeById(topology, id))
        .filter(Boolean)
        .map((node) => ({
          node,
          id: node.id,
          name: node.name,
          type: node.type,
          level: LEVEL_LABELS[node.type] ?? node.type,
          meta: collectMetaPairs(node.meta)
        })),
    [comparisonIds, topology]
  );

  const handleInputChange = (field, value) => {
    setConfig((prev) => ({
      ...prev,
      [field]: Number.isNaN(Number(value)) ? prev[field] : Math.max(1, Math.round(Number(value)))
    }));
  };

  const handleSelectNode = useCallback(
    (node) => {
      if (!node) return;
      const path = getPathToNode(topology, node.id);
      if (path) {
        setFocusPath(path);
      }
    },
    [topology]
  );

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

  const handleMiniMapSelect = useCallback(
    (nodeId) => {
      const path = getPathToNode(topology, nodeId);
      if (path) {
        setFocusPath(path);
      }
    },
    [topology]
  );

  const handleToggleTablePane = useCallback(() => {
    setShowTablePane((prev) => !prev);
  }, []);

  const handleToggleCompare = useCallback((nodeId) => {
    setComparisonIds((prev) => {
      const exists = prev.includes(nodeId);
      if (exists) {
        return prev.filter((id) => id !== nodeId);
      }
      const next = [...prev, nodeId];
      if (next.length > MAX_COMPARISON_ITEMS) {
        next.shift();
      }
      return next;
    });
  }, []);

  const handleRemoveComparison = useCallback((nodeId) => {
    setComparisonIds((prev) => prev.filter((id) => id !== nodeId));
  }, []);

  const handleClearComparison = useCallback(() => {
    setComparisonIds([]);
  }, []);

  const triggerCsvDownload = useCallback((csv, filename) => {
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
    URL.revokeObjectURL(link.href);
  }, []);

  const handleExportFullTopology = useCallback(() => {
    const csv = topologyToCsv(topology);
    triggerCsvDownload(csv, 'ai-distributed-topology.csv');
  }, [topology, triggerCsvDownload]);

  const handleExportFocusedSubtree = useCallback(() => {
    const csv = topologyToCsv(topology, { rootId: activeNode.id });
    const safeName = activeNode.name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/(^-|-$)/g, '');
    triggerCsvDownload(csv, `${safeName || 'focus'}-topology.csv`);
  }, [activeNode, topology, triggerCsvDownload]);

  const handleExportSelectedLevel = useCallback(() => {
    const csv = topologyToCsv(topology, { levelType: exportTargetLevel });
    triggerCsvDownload(csv, `ai-distributed-${exportTargetLevel}.csv`);
  }, [exportTargetLevel, topology, triggerCsvDownload]);

  const miniMapPathSet = useMemo(() => new Set(focusPath), [focusPath]);
  const glossaryEntries = useMemo(() => Object.entries(GLOSSARY), []);
  const focusLevelLabel = LEVEL_LABELS[activeNode.type] ?? activeNode.type;

  const renderMiniMapNode = (node) => {
    const isInPath = miniMapPathSet.has(node.id);
    const isCurrent = node.id === activeNode.id;
    const label = `${LEVEL_LABELS[node.type] ?? node.type}: ${node.name}`;
    const showChildren = isInPath && node.children?.length;
    return (
      <li
        key={node.id}
        className={`mini-map__item ${isInPath ? 'is-path' : ''} ${isCurrent ? 'is-current' : ''}`.trim()}
      >
        <button type="button" onClick={() => handleMiniMapSelect(node.id)} className="mini-map__button">
          {label}
        </button>
        {showChildren && (
          <ul className="mini-map__children">
            {node.children.map((child) => renderMiniMapNode(child))}
          </ul>
        )}
      </li>
    );
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

          <div className="export-group">
            <h3>Export data</h3>
            <button type="button" onClick={handleExportFullTopology}>
              Full system CSV
            </button>
            <button type="button" onClick={handleExportFocusedSubtree}>
              Current focus CSV
            </button>
            <label htmlFor="layerExport">Layer CSV</label>
            <div className="export-level-row">
              <select
                id="layerExport"
                value={exportTargetLevel}
                onChange={(event) => setExportTargetLevel(event.target.value)}
              >
                {LEVEL_EXPORT_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              <button type="button" onClick={handleExportSelectedLevel}>
                Export layer
              </button>
            </div>
          </div>
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
              <button
                type="button"
                className={`table-toggle ${showTablePane ? 'is-active' : ''}`}
                onClick={handleToggleTablePane}
                aria-pressed={showTablePane}
              >
                {showTablePane ? 'Hide table' : 'Show table'}
              </button>
              <button type="button" className="reset-button" onClick={handleResetView}>
                Show full system
              </button>
            </div>
          </div>
          <div className="visual-secondary">
            <div className="mini-map">
              <h3>Hierarchy mini-map</h3>
              <ul className="mini-map__tree">{renderMiniMapNode(topology)}</ul>
            </div>
            <div className="level-outline">
              {LEVEL_ORDER.map((level) => {
                const label = LEVEL_LABELS[level] ?? level;
                const crumb = breadcrumbs.find((item) => item.id.includes(level));
                return (
                  <button
                    type="button"
                    key={level}
                    className={`level-chip ${crumb ? 'is-present' : ''}`}
                    onClick={() => crumb && handleMiniMapSelect(crumb.id)}
                  >
                    <span className="level-chip__dot" data-level={level} />
                    <span className="level-chip__label">{label}</span>
                    <span className="level-chip__value">{crumb ? crumb.name : '—'}</span>
                  </button>
                );
              })}
            </div>
          </div>
          <div className={`visual-content ${showTablePane ? 'has-table' : 'is-single'}`}>
            <div className={`canvas-wrapper ${viewMode}`}>
              {viewMode === 'spatial' ? (
                <TopologyScene
                  topology={topology}
                  focusPath={focusPath}
                  comparisonIds={comparisonIds}
                  onSelectNode={handleSelectNode}
                  onHoverNode={setHoveredNode}
                  onToggleCompare={handleToggleCompare}
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
              <div className="dataflow-trail">
                <h4>Batch journey</h4>
                <ol>
                  {dataFlowTrail.map((item, index) => (
                    <li key={item.id}>
                      <span className="trail-step" data-level={item.type}>
                        {item.level}: {item.name}
                      </span>
                      {index < dataFlowTrail.length - 1 && <span className="trail-arrow" aria-hidden="true">➜</span>}
                    </li>
                  ))}
                </ol>
              </div>
            </div>
            {hoveredNode && (
              <div className="hover-card">
                <h4>{hoveredNode.name}</h4>
                <p>{hoveredNode.meta?.explanation}</p>
                <ul>
                  {collectMetaPairs(hoveredNode.meta).map((item) => (
                    <li key={item.key}>
                      <strong>{item.key}</strong>
                      <em>{item.value}</em>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            </div>
            {showTablePane && (
              <div className="table-pane" aria-live="polite">
                <h3>{focusLevelLabel} breakdown</h3>
                <p className="table-pane__intro">
                  Toggle ↕ inside the 3D matrices or here to compare layers, and click any row to focus that component.
                </p>
                <table>
                  <thead>
                    <tr>
                      <th scope="col">Component</th>
                      <th scope="col">Inline metrics</th>
                      <th scope="col" className="table-actions">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tableRows.length === 0 ? (
                      <tr>
                        <td colSpan={3} className="empty-cell">
                          No deeper layers available for this selection.
                        </td>
                      </tr>
                    ) : (
                      tableRows.map((row) => (
                        <tr key={row.id} data-level={row.type}>
                          <th scope="row">
                            <span className="table-name">{row.name}</span>
                            <span className="table-level">{row.level}</span>
                          </th>
                          <td>
                            <div className="table-metrics">
                              {row.meta.length === 0 ? (
                                <span className="table-metrics__empty">No metrics available</span>
                              ) : (
                                row.meta.map((item) => (
                                  <span key={item.key}>
                                    <strong>{item.key}</strong>
                                    <em>{item.value}</em>
                                  </span>
                                ))
                              )}
                            </div>
                          </td>
                          <td className="table-actions">
                            <button type="button" onClick={() => handleSelectNode(row.node)}>
                              Focus
                            </button>
                            <button type="button" onClick={() => handleToggleCompare(row.id)}>
                              {comparisonIds.includes(row.id) ? 'Remove' : 'Compare'}
                            </button>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
        <aside className="legend-panel">
          <section>
            <h2>Legend</h2>
            <ul className="legend-list">
              <li data-level="server">
                <span className="legend-chip" data-tooltip={GLOSSARY.Server} title={GLOSSARY.Server} tabIndex={0}>
                  Servers (blue)
                </span>
              </li>
              <li data-level="device">
                <span className="legend-chip" data-tooltip={GLOSSARY.GPU} title={GLOSSARY.GPU} tabIndex={0}>
                  GPUs (green)
                </span>
              </li>
              <li data-level="grid">
                <span className="legend-chip" data-tooltip={GLOSSARY.Grid} title={GLOSSARY.Grid} tabIndex={0}>
                  CUDA grids (orange)
                </span>
              </li>
              <li data-level="block">
                <span className="legend-chip" data-tooltip={GLOSSARY.Block} title={GLOSSARY.Block} tabIndex={0}>
                  Blocks (yellow)
                </span>
              </li>
              <li data-level="warp">
                <span className="legend-chip" data-tooltip={GLOSSARY.Warp} title={GLOSSARY.Warp} tabIndex={0}>
                  Warps (purple)
                </span>
              </li>
              <li data-level="thread">
                <span className="legend-chip" data-tooltip={GLOSSARY.Thread} title={GLOSSARY.Thread} tabIndex={0}>
                  Threads (gray)
                </span>
              </li>
            </ul>
          </section>
          <section>
            <h2>Compare selections</h2>
            {comparisonDetails.length === 0 ? (
              <p className="empty-message">Use the ↕ control to pin components for side-by-side review.</p>
            ) : (
              <div className="comparison-panel">
                {comparisonDetails.map((item) => (
                  <article key={item.id} className="comparison-card" data-level={item.type}>
                    <header>
                      <span>{item.level}</span>
                      <button type="button" onClick={() => handleRemoveComparison(item.id)} aria-label="Remove from comparison">
                        ×
                      </button>
                    </header>
                    <h3>{item.name}</h3>
                    <ul>
                      {item.meta.map((meta) => (
                        <li key={meta.key}>
                          <strong>{meta.key}</strong>
                          <em>{meta.value}</em>
                        </li>
                      ))}
                    </ul>
                  </article>
                ))}
                <button type="button" className="clear-button" onClick={handleClearComparison}>
                  Clear all
                </button>
              </div>
            )}
          </section>
          <section>
            <h2>Concept glossary</h2>
            <ul className="glossary-tooltip-list">
              {glossaryEntries.map(([term, description]) => (
                <li key={term}>
                  <span className="glossary-chip" data-tooltip={description} title={description} tabIndex={0}>
                    {term}
                  </span>
                </li>
              ))}
            </ul>
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
