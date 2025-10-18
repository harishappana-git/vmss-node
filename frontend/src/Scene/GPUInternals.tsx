import { useEffect, useMemo } from 'react'
import { Text } from '@react-three/drei'
import { Color, TorusGeometry, Vector3 } from 'three'
import type { ClusterSpec, GPUSpec, NodeSpec, RackSpec } from '../types'
import { useExplorerStore } from '../state/selectionStore'
import { focusOn } from '../lib/camera'

const hbmColor = new Color('#7dd0ff')
const dieColor = new Color('#3f6bff')
const l2Color = new Color('#ffa45c')

function HbmStack({ position, label }: { position: Vector3; label: string }) {
  return (
    <group position={position.toArray() as [number, number, number]}>
      <mesh>
        <cylinderGeometry args={[0.8, 0.8, 1.2, 24]} />
        <meshStandardMaterial color={hbmColor} emissive="#1d4f7a" roughness={0.4} />
      </mesh>
      <Text position={[0, 1, 0]} fontSize={0.28} color="#c8f2ff" anchorX="center">
        {label}
      </Text>
    </group>
  )
}

type GPUInternalsProps = {
  gpu: GPUSpec
  node: NodeSpec
  cluster: ClusterSpec
  rack: RackSpec
}

export function GPUInternals({ gpu, node, cluster, rack }: GPUInternalsProps) {
  const select = useExplorerStore((state) => state.select)
  const header = `${cluster.name} › ${rack.name} › ${node.hostname} › ${gpu.name}`

  useEffect(() => {
    focusOn([0, 5, 12], [0, 0, 0], 32)
  }, [])

  const hbmPositions = useMemo(() => {
    const radius = 4
    const stacks = 8
    return Array.from({ length: stacks }, (_, idx) => {
      const angle = (idx / stacks) * Math.PI * 2
      const x = Math.cos(angle) * radius
      const z = Math.sin(angle) * radius
      return new Vector3(x, 0, z)
    })
  }, [])

  const l2Geometry = useMemo(() => new TorusGeometry(4.8, 0.2, 24, 64), [])

  useEffect(() => () => l2Geometry.dispose(), [l2Geometry])

  return (
    <group>
      <Text position={[0, 6, 0]} fontSize={0.9} color="#b4c9ff" anchorX="center">
        {header}
      </Text>
      <mesh
        position={[0, 0, 0]}
        onClick={(event) => {
          event.stopPropagation()
          select({ kind: 'gpu', id: gpu.uuid })
        }}
      >
        <boxGeometry args={[6, 0.6, 6]} />
        <meshStandardMaterial color="#1a1f33" roughness={0.6} metalness={0.2} />
      </mesh>
      <mesh position={[0, 0.5, 0]}>
        <boxGeometry args={[4.4, 0.6, 4.4]} />
        <meshStandardMaterial color={dieColor} emissive="#153cff" roughness={0.45} metalness={0.5} />
      </mesh>
      <mesh geometry={l2Geometry} rotation={[Math.PI / 2, 0, 0]} position={[0, 0.8, 0]}>
        <meshStandardMaterial color={l2Color} emissive="#ff7a1a" roughness={0.35} />
      </mesh>
      {hbmPositions.map((pos, idx) => (
        <HbmStack key={idx} position={pos} label={`HBM${idx + 1}`} />
      ))}
      <Text position={[0, -1.2, 0]} fontSize={0.5} color="#9fb8ff" anchorX="center" anchorY="middle">
        ≈ {gpu.memoryGB} GB HBM3e · {gpu.hbmBandwidthTBs.toFixed(1)} TB/s · NVLink {gpu.nvlinkTBs.toFixed(1)} TB/s
      </Text>
      <Text position={[0, -2.2, 0]} fontSize={0.4} color="#d6e4ff" anchorX="center" anchorY="middle">
        MIG: {gpu.migSupported ? gpu.migGuide ?? 'Supported' : 'Disabled'}
      </Text>
    </group>
  )
}
