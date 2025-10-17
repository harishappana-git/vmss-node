import React, { useMemo } from 'react';
import PropTypes from 'prop-types';
import { hierarchy, partition } from 'd3-hierarchy';
import { arc } from 'd3-shape';

const NODE_COLORS = {
  'distributed-system': '#0b7285',
  server: '#1f6feb',
  device: '#2ea043',
  grid: '#f0883e',
  block: '#f2cc60',
  warp: '#bf7af0',
  thread: '#8b949e'
};

const SunburstView = ({ topology, focusPath, onSelectNode, onHoverNode }) => {
  const radius = 210;
  const focusSet = useMemo(() => new Set(focusPath), [focusPath]);

  const arcs = useMemo(() => {
    const root = hierarchy(topology)
      .sum((d) => (d.children?.length ? 0 : 1))
      .sort((a, b) => a.depth - b.depth);
    const partitionLayout = partition().size([2 * Math.PI, radius]);
    const data = partitionLayout(root);
    const arcGenerator = arc()
      .startAngle((d) => d.x0)
      .endAngle((d) => d.x1)
      .innerRadius((d) => d.y0)
      .outerRadius((d) => d.y1);
    return data.descendants().map((node) => ({
      node: node.data,
      depth: node.depth,
      path: arcGenerator(node),
      x0: node.x0,
      x1: node.x1,
      y0: node.y0,
      y1: node.y1
    }));
  }, [topology, radius]);

  return (
    <svg className="sunburst" viewBox={`0 0 ${radius * 2} ${radius * 2}`} role="presentation">
      <g transform={`translate(${radius}, ${radius})`}>
        {arcs.map(({ node, depth, path }, index) => {
          if (!path || depth === 0) return null;
          const color = NODE_COLORS[node.type] ?? '#6e7681';
          const isFocused = focusSet.has(node.id);
          return (
            <path
              key={`${node.id}-${index}`}
              className={`sunburst__arc ${isFocused ? 'is-flow' : ''}`}
              d={path}
              fill={color}
              fillOpacity={isFocused ? 0.9 : 0.55}
              stroke="#0d1117"
              strokeWidth={isFocused ? 2 : 1}
              data-level={node.type}
              onMouseEnter={() => onHoverNode(node)}
              onMouseLeave={() => onHoverNode(null)}
              onClick={() => onSelectNode(node)}
            >
              <title>{`${node.name}\n${node.meta?.explanation ?? ''}`}</title>
            </path>
          );
        })}
      </g>
      <text x="50%" y="50%" textAnchor="middle" className="sunburst__label">
        Sunburst hierarchy
      </text>
    </svg>
  );
};

SunburstView.propTypes = {
  topology: PropTypes.object.isRequired,
  focusPath: PropTypes.arrayOf(PropTypes.string).isRequired,
  onSelectNode: PropTypes.func.isRequired,
  onHoverNode: PropTypes.func.isRequired
};

export default SunburstView;
