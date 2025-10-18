import { useMemo } from 'react'
import { Line } from '@react-three/drei'
import type { ClusterSpec, NodeSpec, Topology } from '../types'
import { useExplorerStore } from '../state/selectionStore'
import type { ThreeEvent } from '@react-three/fiber'

type Props = {
  topology: Topology
}

type PositionedNode = {
  node: NodeSpec
  position: [number, number, number]
  rackId: string
  cluster: ClusterSpec
}

function computeNodePosition(clusterIndex: number, rackIndex: number, slot: number): [number, number, number] {
  const clusterSpacing = 36
  const rackSpacing = 16
  const slotSpacing = 6
  const x = clusterIndex * clusterSpacing
  const z = rackIndex * rackSpacing - rackSpacing
  const y = slot * slotSpacing - 4
  return [x, y, z]
}

function NodeChassis({ positioned }: { positioned: PositionedNode }) {
  const select = useExplorerStore((state) => state.select)
  const enterNode = useExplorerStore((state) => state.enterNode)
  const selection = useExplorerStore((state) => state.selection)
  const isSelected = selection?.kind === 'node' && selection.id === positioned.node.id

  const onClick = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation()
    select({ kind: 'node', id: positioned.node.id })
  }

  const onDoubleClick = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation()
    select({ kind: 'node', id: positioned.node.id })
    enterNode(positioned.node, {
      clusterId: positioned.cluster.id,
      clusterLabel: positioned.cluster.name,
      rackId: positioned.rackId,
      rackLabel: `Rack ${positioned.rackId}`
    })
  }

  return (
    <group position={positioned.position}>
      <mesh onClick={onClick} onDoubleClick={onDoubleClick} castShadow receiveShadow>
        <boxGeometry args={[8, 2.4, 4]} />
        <meshStandardMaterial
          color={isSelected ? '#2f7bff' : '#162236'}
          emissive={isSelected ? '#264dff' : '#0e1320'}
          metalness={0.4}
          roughness={0.55}
        />
      </mesh>
      <mesh position={[0, 1.4, 0]}>
        <cylinderGeometry args={[1.4, 1.4, 0.2, 32]} />
        <meshStandardMaterial color="#3a9efd" emissive="#1c4bd6" roughness={0.35} metalness={0.6} />
      </mesh>
      <group position={[-2.6, 0.4, -1.2]}>
        {positioned.node.gpus.map((gpu, idx) => {
          const column = idx % 4
          const row = Math.floor(idx / 4)
          const x = column * 1.6
          const z = row * 1.6
          const isGpuSelected = selection?.kind === 'gpu' && selection.id === gpu.uuid
          return (
            <mesh
              key={gpu.uuid}
              position={[x, 0, z]}
              onClick={(event) => {
                event.stopPropagation()
                select({ kind: 'gpu', id: gpu.uuid })
              }}
            >
              <boxGeometry args={[1.1, 0.3, 1.1]} />
              <meshStandardMaterial
                color={isGpuSelected ? '#9ad7ff' : '#1d8cf2'}
                emissive={isGpuSelected ? '#6297ff' : '#0a2c78'}
                roughness={0.4}
              />
            </mesh>
          )
        })}
      </group>
    </group>
  )
}

export function ClusterScene({ topology }: Props) {
  const positionedNodes = useMemo(() => {
    const nodes: PositionedNode[] = []
    topology.clusters.forEach((cluster, clusterIdx) => {
      cluster.racks.forEach((rack, rackIdx) => {
        rack.nodes.forEach((node) => {
          nodes.push({
            node,
            position: computeNodePosition(clusterIdx, rackIdx, node.rackPosition),
            rackId: rack.id,
            cluster
          })
        })
      })
    })
    return nodes
  }, [topology])

  const linkGeometries = useMemo(() => {
    return topology.clusters.flatMap((cluster, clusterIdx) => {
      return cluster.links
        .filter((link) => link.type === 'IB')
        .map((link) => {
          const [fromNode] = link.from.split(':')
          const [toNode] = link.to.split(':')
          const from = positionedNodes.find((n) => n.node.id === fromNode)
          const to = positionedNodes.find((n) => n.node.id === toNode)
          if (!from || !to) return null
          return {
            clusterIdx,
            link,
            start: from.position,
            end: to.position
          }
        })
        .filter(Boolean) as {
        clusterIdx: number
        link: ClusterSpec['links'][number]
        start: [number, number, number]
        end: [number, number, number]
      }[]
    })
  }, [positionedNodes, topology])

  return (
    <group>
      {positionedNodes.map((positioned) => (
        <NodeChassis key={positioned.node.id} positioned={positioned} />
      ))}
      {linkGeometries.map(({ link, start, end }) => (
        <Line
          key={link.id}
          points={[start, [(start[0] + end[0]) / 2, Math.max(start[1], end[1]) + 6, (start[2] + end[2]) / 2], end]}
          color="#4da6ff"
          lineWidth={2.5}
          transparent
          opacity={0.85}
        />
      ))}
    </group>
  )
}
