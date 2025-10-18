import { useEffect, useMemo } from 'react'
import { Text } from '@react-three/drei'
import { Color, QuadraticBezierCurve3, TubeGeometry, Vector3 } from 'three'
import type { ClusterSpec, NodeSpec, RackSpec } from '../types'
import { useExplorerStore } from '../state/selectionStore'
import { focusOn } from '../lib/camera'
import type { ThreeEvent } from '@react-three/fiber'

const nvlinkColor = new Color('#f6c255')
const nvlinkEmissive = new Color('#a36f1f')

function NVLinkArc({ from, to, capacity }: { from: Vector3; to: Vector3; capacity: number }) {
  const curve = useMemo(() => {
    const mid = from.clone().lerp(to, 0.5)
    mid.y += 1.6
    return new QuadraticBezierCurve3(from.clone(), mid, to.clone())
  }, [from, to])

  const geometry = useMemo(() => new TubeGeometry(curve, 24, Math.min(0.22, capacity / 800), 12, false), [capacity, curve])

  useEffect(() => () => geometry.dispose(), [geometry])

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial color={nvlinkColor} emissive={nvlinkEmissive} roughness={0.35} metalness={0.5} />
    </mesh>
  )
}

type NodeSceneProps = {
  node: NodeSpec
  cluster: ClusterSpec
  rack: RackSpec
}

const temp = new Vector3()

export function NodeScene({ node, cluster, rack }: NodeSceneProps) {
  const select = useExplorerStore((state) => state.select)
  const enterGpu = useExplorerStore((state) => state.enterGpu)
  const selection = useExplorerStore((state) => state.selection)
  const nodeLabel = `${cluster.name} / ${rack.name} / ${node.hostname}`

  const nvSwitchPosition = new Vector3(0, 0.6, 0)

  const gpuPositions = useMemo(() => {
    const radius = 6
    return node.gpus.map((gpu, idx) => {
      const angle = (idx / node.gpus.length) * Math.PI * 2
      const x = Math.cos(angle) * radius
      const z = Math.sin(angle) * radius
      return { gpu, position: new Vector3(x, 0, z), angle }
    })
  }, [node.gpus])

  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow position={[0, -0.2, 0]}>
        <cylinderGeometry args={[9.5, 9.5, 0.6, 64]} />
        <meshStandardMaterial color="#121824" roughness={0.7} />
      </mesh>
      <mesh position={nvSwitchPosition}>
        <cylinderGeometry args={[2.2, 2.2, 0.6, 48]} />
        <meshStandardMaterial color="#3a9efd" emissive="#1a4dd9" metalness={0.6} roughness={0.4} />
      </mesh>
      {gpuPositions.map(({ gpu, position, angle }) => {
        const isSelected = selection?.kind === 'gpu' && selection.id === gpu.uuid
        const rotation = [-Math.PI / 2, 0, angle] as [number, number, number]
        const handleClick = (event: ThreeEvent<MouseEvent>) => {
          event.stopPropagation()
          select({ kind: 'gpu', id: gpu.uuid })
        }
        const handleDoubleClick = (event: ThreeEvent<MouseEvent>) => {
          event.stopPropagation()
          select({ kind: 'gpu', id: gpu.uuid })
          event.eventObject.getWorldPosition(temp)
          const target: [number, number, number] = [temp.x, temp.y, temp.z]
          const camera: [number, number, number] = [temp.x + 4, temp.y + 3, temp.z + 4]
          focusOn(camera, target, 36)
          enterGpu(
            { kind: 'gpu', id: gpu.uuid },
            {
              node,
              clusterId: cluster.id,
              clusterLabel: cluster.name,
              rackId: rack.id,
              rackLabel: rack.name,
              gpuLabel: `${gpu.name} (${gpu.model})`
            }
          )
        }

        const from = position.clone()
        from.y = 0.4
        const to = nvSwitchPosition.clone()
        to.y = 0.8

        return (
          <group key={gpu.uuid} position={position}>
            <mesh position={[0, 0.4, 0]} rotation={rotation} onClick={handleClick} onDoubleClick={handleDoubleClick}>
              <boxGeometry args={[2.4, 0.4, 1.2]} />
              <meshStandardMaterial
                color={isSelected ? '#9ad7ff' : '#1d8cf2'}
                emissive={isSelected ? '#6da4ff' : '#0a2c78'}
                roughness={0.38}
              />
            </mesh>
            <NVLinkArc from={from} to={to} capacity={node.nvlinkSwitchAggregateTBs * 1024} />
          </group>
        )
      })}
      <Text position={[0, 4.5, 0]} fontSize={0.8} color="#8fb7ff" anchorX="center" anchorY="middle">
        {nodeLabel}
      </Text>
    </group>
  )
}
