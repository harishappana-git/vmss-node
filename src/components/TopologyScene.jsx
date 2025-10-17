import React, { useMemo } from 'react';
import PropTypes from 'prop-types';
import { Canvas } from '@react-three/fiber';
import { Html, OrbitControls, PerspectiveCamera } from '@react-three/drei';

import { LEVEL_ORDER, LEVEL_LABELS } from '../topology.js';

const NODE_COLORS = {
  'distributed-system': '#0b7285',
  server: '#1f6feb',
  device: '#2ea043',
  grid: '#f0883e',
  block: '#f2cc60',
  warp: '#bf7af0',
  thread: '#8b949e'
};

const collectFocusNodes = (root, focusPath) => {
  const nodes = [root];
  let current = root;
  for (let i = 1; i < focusPath.length; i += 1) {
    const next = current.children?.find((child) => child.id === focusPath[i]);
    if (next) {
      nodes.push(next);
      current = next;
    } else {
      break;
    }
  }
  return nodes;
};

const collectDescendantIds = (node, set = new Set()) => {
  set.add(node.id);
  (node.children ?? []).forEach((child) => collectDescendantIds(child, set));
  return set;
};

const collectLevels = (root, focusNodes) => {
  const levels = [];
  LEVEL_ORDER.forEach((type, index) => {
    if (index === 0) {
      levels.push({ type, nodes: [root], parentId: null });
      return;
    }
    const parent = focusNodes[Math.min(index - 1, focusNodes.length - 1)];
    const nodes = parent?.children ?? [];
    if (nodes.length > 0) {
      levels.push({ type, nodes, parentId: parent?.id ?? null });
    }
  });
  return levels;
};

const MatrixCell = ({
  node,
  isActive,
  isContext,
  isDescendant,
  isFlow,
  isCompared,
  onSelect,
  onHover,
  onToggleCompare
}) => (
  <div
    role="button"
    tabIndex={0}
    className={`matrix-cell matrix-${node.type} ${isActive ? 'is-active' : ''} ${
      isDescendant ? 'is-descendant' : ''
    } ${isContext ? 'is-context' : ''} ${isFlow ? 'is-flow' : ''} ${
      isCompared ? 'is-compared' : ''
    }`.trim()}
    style={{ '--node-color': NODE_COLORS[node.type] ?? '#6e7681' }}
    onClick={(event) => {
      event.stopPropagation();
      onSelect(node);
    }}
    onKeyDown={(event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        onSelect(node);
      }
    }}
    onMouseEnter={() => onHover(node)}
    onFocus={() => onHover(node)}
    onMouseLeave={() => onHover(null)}
    onBlur={() => onHover(null)}
  >
    <header className="matrix-cell__header">
      <span className="matrix-cell__title">{node.name}</span>
      <button
        type="button"
        className={`matrix-cell__compare ${isCompared ? 'is-selected' : ''}`}
        onClick={(event) => {
          event.stopPropagation();
          onToggleCompare(node.id);
        }}
        aria-pressed={isCompared}
        aria-label={isCompared ? 'Remove from comparison' : 'Add to comparison'}
      >
        ↕
      </button>
    </header>
    <span className="matrix-cell__meta">{node.meta?.explanation}</span>
    {node.meta && (
      <div className="matrix-cell__stats">
        {Object.entries(node.meta)
          .filter(([key]) => key !== 'explanation')
          .slice(0, 4)
          .map(([key, value]) => (
            <span key={key}>
              <strong>{key}</strong>
              <em>{value}</em>
            </span>
          ))}
      </div>
    )}
  </div>
);

MatrixCell.propTypes = {
  node: PropTypes.object.isRequired,
  isActive: PropTypes.bool,
  isContext: PropTypes.bool,
  isDescendant: PropTypes.bool,
  isFlow: PropTypes.bool,
  isCompared: PropTypes.bool,
  onSelect: PropTypes.func.isRequired,
  onHover: PropTypes.func.isRequired,
  onToggleCompare: PropTypes.func.isRequired
};

MatrixCell.defaultProps = {
  isActive: false,
  isContext: false,
  isDescendant: false,
  isFlow: false,
  isCompared: false
};

const LevelMatrix = ({
  level,
  nodes,
  focusSet,
  descendantSet,
  flowSet,
  comparisonSet,
  activeId,
  parentId,
  onSelect,
  onHover,
  onToggleCompare
}) => (
  <div className="matrix">
    <header className="matrix__header">
      <span className="matrix__title">{LEVEL_LABELS[level]} matrix</span>
      <span className="matrix__subtitle">{nodes.length} {LEVEL_LABELS[level].toLowerCase()} visible</span>
    </header>
    <div className="matrix__grid">
      {nodes.map((node) => {
        const isActive = node.id === activeId;
        const isContext = !isActive && parentId !== null && node.id !== activeId;
        const isDescendant = descendantSet.has(node.id) || focusSet.has(node.id);
        const isFlow = flowSet.has(node.id);
        const isCompared = comparisonSet.has(node.id);
        return (
          <MatrixCell
            key={node.id}
            node={node}
            isActive={isActive}
            isContext={isContext && !isDescendant}
            isDescendant={isDescendant}
            isFlow={isFlow}
            isCompared={isCompared}
            onSelect={onSelect}
            onHover={onHover}
            onToggleCompare={onToggleCompare}
          />
        );
      })}
    </div>
  </div>
);

LevelMatrix.propTypes = {
  level: PropTypes.string.isRequired,
  nodes: PropTypes.arrayOf(PropTypes.object).isRequired,
  focusSet: PropTypes.instanceOf(Set).isRequired,
  descendantSet: PropTypes.instanceOf(Set).isRequired,
  flowSet: PropTypes.instanceOf(Set).isRequired,
  comparisonSet: PropTypes.instanceOf(Set).isRequired,
  activeId: PropTypes.string,
  parentId: PropTypes.string,
  onSelect: PropTypes.func.isRequired,
  onHover: PropTypes.func.isRequired,
  onToggleCompare: PropTypes.func.isRequired
};

LevelMatrix.defaultProps = {
  activeId: null,
  parentId: null
};

const LevelPlate = ({ levelIndex, children }) => {
  const z = -levelIndex * 6;
  const y = 0;
  const width = 18;
  const depth = 10;
  return (
    <group position={[0, y, z]}>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.6, 0]}
        scale={[1, 1, 1]}
      >
        <planeGeometry args={[width, depth]} />
        <meshStandardMaterial color="#0d1117" opacity={0.45} transparent />
      </mesh>
      <Html
        transform
        distanceFactor={18}
        position={[0, 0.1, 0]}
        style={{ width: `${width * 22}px`, maxWidth: '720px' }}
      >
        <div className="matrix-wrapper">{children}</div>
      </Html>
    </group>
  );
};

LevelPlate.propTypes = {
  levelIndex: PropTypes.number.isRequired,
  children: PropTypes.node.isRequired
};

const Scene = ({ topology, focusPath, comparisonIds, onSelectNode, onHoverNode, onToggleCompare }) => {
  const focusNodes = useMemo(() => collectFocusNodes(topology, focusPath), [topology, focusPath]);
  const levels = useMemo(() => collectLevels(topology, focusNodes), [topology, focusNodes]);
  const focusSet = useMemo(() => new Set(focusPath), [focusPath]);
  const activeNode = focusNodes[focusNodes.length - 1];
  const descendantSet = useMemo(() => collectDescendantIds(activeNode), [activeNode]);
  const flowSet = focusSet;
  const comparisonSet = useMemo(() => new Set(comparisonIds), [comparisonIds]);

  return (
    <Canvas dpr={[1, 2]} shadows>
      <PerspectiveCamera makeDefault position={[0, 12, 32]} fov={52} />
      <color attach="background" args={[0x04070d]} />
      <ambientLight intensity={0.6} />
      <directionalLight position={[18, 22, 18]} intensity={1.1} />
      <pointLight position={[-10, 12, -14]} intensity={0.4} color="#58a6ff" />
      {levels.map((level, index) => (
        <LevelPlate key={level.type} levelIndex={index}>
          <LevelMatrix
            level={level.type}
            nodes={level.nodes}
            focusSet={focusSet}
            descendantSet={descendantSet}
            flowSet={flowSet}
            comparisonSet={comparisonSet}
            activeId={focusPath[index] ?? null}
            parentId={level.parentId}
            onSelect={onSelectNode}
            onHover={onHoverNode}
            onToggleCompare={onToggleCompare}
          />
        </LevelPlate>
      ))}
      <OrbitControls
        enablePan
        enableDamping
        dampingFactor={0.15}
        minDistance={12}
        maxDistance={68}
        minPolarAngle={0.2 * Math.PI}
        maxPolarAngle={0.85 * Math.PI}
      />
    </Canvas>
  );
};

Scene.propTypes = {
  topology: PropTypes.object.isRequired,
  focusPath: PropTypes.arrayOf(PropTypes.string).isRequired,
  comparisonIds: PropTypes.arrayOf(PropTypes.string).isRequired,
  onSelectNode: PropTypes.func.isRequired,
  onHoverNode: PropTypes.func.isRequired,
  onToggleCompare: PropTypes.func.isRequired
};

const TopologyScene = ({
  topology,
  focusPath,
  comparisonIds,
  onSelectNode,
  onHoverNode,
  onToggleCompare
}) => {
  const memoisedScene = useMemo(
    () => (
      <Scene
        key={focusPath.join('>')}
        topology={topology}
        focusPath={focusPath}
        comparisonIds={comparisonIds}
        onSelectNode={onSelectNode}
        onHoverNode={onHoverNode}
        onToggleCompare={onToggleCompare}
      />
    ),
    [topology, focusPath, comparisonIds, onSelectNode, onHoverNode, onToggleCompare]
  );

  return memoisedScene;
};

TopologyScene.propTypes = {
  topology: PropTypes.object.isRequired,
  focusPath: PropTypes.arrayOf(PropTypes.string).isRequired,
  comparisonIds: PropTypes.arrayOf(PropTypes.string).isRequired,
  onSelectNode: PropTypes.func.isRequired,
  onHoverNode: PropTypes.func.isRequired,
  onToggleCompare: PropTypes.func.isRequired
};

export default TopologyScene;
