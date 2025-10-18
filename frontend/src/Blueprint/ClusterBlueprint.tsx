import { useMemo } from 'react'
import type { KeyboardEvent as ReactKeyboardEvent, MouseEvent as ReactMouseEvent } from 'react'
import type { ClusterSpec, NodeSpec, RackSpec, Topology } from '../types'
import { useExplorerStore } from '../state/selectionStore'
import { useMetricsStore } from '../state/metricsStore'

const NODE_WIDTH = 150
const NODE_HEIGHT = 90
const RACK_GAP_X = 220
const NODE_GAP_Y = 140
const CLUSTER_PADDING = 120

export function ClusterBlueprint({ topology }: { topology: Topology }) {
  const enterNode = useExplorerStore((state) => state.enterNode)
  const selection = useExplorerStore((state) => state.selection)
  const nodeMetrics = useMetricsStore((state) => state.node)

  const activateNode = (
    node: NodeSpec,
    rack: RackSpec,
    cluster: ClusterSpec,
    event: ReactMouseEvent<SVGGElement> | ReactKeyboardEvent<SVGGElement>
  ) => {
    if ('key' in event) {
      if (event.key !== 'Enter' && event.key !== ' ') {
        return
      }
      event.preventDefault()
    }
    enterNode(node, {
      clusterId: cluster.id,
      clusterLabel: cluster.name,
      rackId: rack.id,
      rackLabel: rack.name
    })
  }

  const layouts = useMemo(() => {
    return topology.clusters.map((cluster) => {
      const nodePositions = new Map<string, { x: number; y: number; rackName: string; rackId: string }>()
      cluster.racks.forEach((rack, rackIndex) => {
        const baseX = CLUSTER_PADDING + rackIndex * RACK_GAP_X
        rack.nodes.forEach((node, nodeIndex) => {
          const x = baseX
          const y = CLUSTER_PADDING + nodeIndex * NODE_GAP_Y
          nodePositions.set(node.id, { x, y, rackName: rack.name, rackId: rack.id })
        })
      })

      const width = Math.max(cluster.racks.length * RACK_GAP_X + CLUSTER_PADDING * 2, 640)
      const height = Math.max(
        (Math.max(...cluster.racks.map((rack) => rack.nodes.length)) || 1) * NODE_GAP_Y + CLUSTER_PADDING * 2,
        480
      )

      return { cluster, nodePositions, width, height }
    })
  }, [topology.clusters])

  return (
    <div className="cluster-blueprint">
      {layouts.map(({ cluster, nodePositions, width, height }) => (
        <section key={cluster.id} className="cluster-blueprint__cluster">
          <header>
            <h2>{cluster.name}</h2>
            <p>{cluster.racks.length} racks · {cluster.racks.reduce((acc, rack) => acc + rack.nodes.length, 0)} nodes</p>
          </header>
          <svg viewBox={`0 0 ${width} ${height}`} role="presentation">
            <defs>
              <marker id="arrow" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="6" markerHeight="6" orient="auto">
                <path d="M0,0 L6,3 L0,6 z" fill="#4aa9ff" />
              </marker>
            </defs>
            {cluster.links.map((link) => {
              const fromId = link.from.split(':')[0]
              const toId = link.to.split(':')[0]
              const from = nodePositions.get(fromId)
              const to = nodePositions.get(toId)
              if (!from || !to) return null
              const path = `M ${from.x + NODE_WIDTH / 2} ${from.y + NODE_HEIGHT / 2} L ${to.x + NODE_WIDTH / 2} ${to.y + NODE_HEIGHT / 2}`
              return (
                <path
                  key={link.id}
                  d={path}
                  className="cluster-blueprint__link"
                  markerEnd="url(#arrow)"
                  strokeWidth={link.type === 'IB' ? 4 : 2}
                >
                  <title>
                    {link.type === 'IB' ? 'InfiniBand ' : 'NVLink '}
                    {link.type === 'IB' ? `${link.capacityGbps} Gb/s` : `${link.capacityGBs.toFixed(1)} GB/s`}
                  </title>
                </path>
              )
            })}
            {cluster.racks.map((rack) => (
              <g key={rack.id}>
                {rack.nodes.map((node) => {
                  const position = nodePositions.get(node.id)
                  if (!position) return null
                  const metrics = nodeMetrics[node.id]
                  const isSelected = selection?.kind === 'node' && selection.id === node.id
                  return (
                    <g
                      key={node.id}
                      className={`cluster-blueprint__node${isSelected ? ' is-selected' : ''}`}
                      transform={`translate(${position.x}, ${position.y})`}
                      role="button"
                      tabIndex={0}
                      aria-label={`${node.hostname} node in rack ${rack.name}`}
                      onClick={(event) => activateNode(node, rack, cluster, event)}
                      onKeyDown={(event) => activateNode(node, rack, cluster, event)}
                    >
                      <rect
                        width={NODE_WIDTH}
                        height={NODE_HEIGHT}
                        rx={10}
                        ry={10}
                      >
                        <title>{`${node.hostname} — 8× B200 · 4 TB RAM`}</title>
                      </rect>
                      <text x={NODE_WIDTH / 2} y={24} textAnchor="middle" className="cluster-blueprint__node-title">
                        {node.hostname}
                      </text>
                      <text x={NODE_WIDTH / 2} y={46} textAnchor="middle" className="cluster-blueprint__node-subtitle">
                        8× B200 · NVSwitch {node.nvlinkSwitchAggregateTBs.toFixed(1)} TB/s
                      </text>
                      <text x={NODE_WIDTH / 2} y={68} textAnchor="middle" className="cluster-blueprint__node-metrics">
                        {metrics
                          ? `${(metrics.cpuUtil * 100).toFixed(0)}% CPU · ${metrics.ibUtilGbps.toFixed(0)} Gb/s IB`
                          : 'Live metrics pending'}
                      </text>
                    </g>
                  )
                })}
              </g>
            ))}
          </svg>
        </section>
      ))}
    </div>
  )
}
