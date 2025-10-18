import { useMemo } from 'react'
import { Line } from '@react-three/drei'
import type { Topology, Node, Link } from '../types'
import { useSelectionStore } from '../state/selectionStore'

type Props = {
  topology: Topology
}

type PositionedNode = {
  node: Node
  position: [number, number, number]
}

type PositionedLink = {
  link: Link
  start: [number, number, number]
  end: [number, number, number]
}

function computeNodePosition(clusterIdx: number, rackIdx: number, nodeIdx: number): [number, number, number] {
  const clusterSpacing = 28
  const rackSpacing = 10
  const nodeSpacing = 2.6
  const clusterX = clusterIdx * clusterSpacing
  const rackZ = (rackIdx - 1.5) * rackSpacing
  const y = (nodeIdx - 1.5) * nodeSpacing
  return [clusterX, y, rackZ]
}

function computeLinkEndpoints(nodes: PositionedNode[], link: Link): PositionedLink | null {
  const [fromNodeId] = link.from.split(':')
  const [toNodeId] = link.to.split(':')
  const from = nodes.find((n) => n.node.id === fromNodeId)
  const to = nodes.find((n) => n.node.id === toNodeId)
  if (!from || !to) return null
  const start: [number, number, number] = [from.position[0], from.position[1], from.position[2]]
  const end: [number, number, number] = [to.position[0], to.position[1], to.position[2]]
  return { link, start, end }
}

export function PhysicalScene({ topology }: Props) {
  const select = useSelectionStore((state) => state.select)
  const layout = useMemo(() => {
    const nodes: PositionedNode[] = []
    const links: PositionedLink[] = []

    topology.clusters.forEach((cluster, cIdx) => {
      cluster.racks.forEach((rack, rIdx) => {
        rack.nodes.forEach((node, nIdx) => {
          const position = computeNodePosition(cIdx, rIdx, nIdx)
          nodes.push({ node, position })
        })
      })
      cluster.links.forEach((link) => {
        const positioned = computeLinkEndpoints(nodes, link)
        if (positioned) {
          links.push(positioned)
        }
      })
    })

    return { nodes, links }
  }, [topology])

  return (
    <group>
      {layout.nodes.map(({ node, position }) => (
        <group key={node.id} position={position}>
          <mesh
            onClick={(event) => {
              event.stopPropagation()
              select({ kind: 'node', id: node.id })
            }}
          >
            <boxGeometry args={[4.2, 1.6, 2.4]} />
            <meshStandardMaterial color={0x1b2845} emissive={0x0f1624} metalness={0.3} roughness={0.6} />
          </mesh>
          <group position={[-1.6, 0.9, 0]}>
            {node.gpus.map((gpu, idx) => {
              const col = idx % 4
              const row = Math.floor(idx / 4)
              const x = col * 0.8
              const z = row * 0.8 - 0.3
              return (
                <mesh
                  key={gpu.uuid}
                  position={[x, 0, z]}
                  onClick={(event) => {
                    event.stopPropagation()
                    select({ kind: 'gpu', id: gpu.uuid })
                  }}
                >
                  <boxGeometry args={[0.6, 0.2, 0.6]} />
                  <meshStandardMaterial color={0x2d91ff} emissive={0x04387a} />
                </mesh>
              )
            })}
          </group>
        </group>
      ))}
      {layout.links.map(({ link, start, end }) => (
        <Line
          key={link.id}
          points={[start, [(start[0] + end[0]) / 2, start[1] + 1.5, (start[2] + end[2]) / 2], end]}
          color={link.type === 'IB' ? '#4da6ff' : '#f6c255'}
          lineWidth={2.4}
          onClick={(event) => {
            event.stopPropagation()
            select({ kind: 'link', id: link.id })
          }}
        />
      ))}
    </group>
  )
}
