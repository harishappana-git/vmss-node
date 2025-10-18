import { useEffect, useMemo } from 'react'
import { Edges, Text } from '@react-three/drei'
import { Color, QuadraticBezierCurve3, TubeGeometry, Vector3 } from 'three'
import type { ClusterSpec, MemoryDescriptor, NodeSpec, RackSpec } from '../types'
import { useExplorerStore } from '../state/selectionStore'
import type { ThreeEvent } from '@react-three/fiber'
import { useMetricsStore } from '../state/metricsStore'

const nvlinkColor = new Color('#f6c255')
const nvlinkEmissive = new Color('#a36f1f')
const dimmColor = new Color('#2d8f6f')
const dimmSelected = new Color('#64f2c5')
const gpuBaseColor = new Color('#1d8cf2')
const gpuSelectedColor = new Color('#9ad7ff')
const chassisColor = new Color('#162236')
const chassisSelected = new Color('#2754ff')

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

type MemoryModuleProps = {
  descriptor: MemoryDescriptor
  position: [number, number, number]
  rotation?: [number, number, number]
  selected: boolean
  node: NodeSpec
  cluster: ClusterSpec
  rack: RackSpec
}

function MemoryModule({ descriptor, position, rotation, selected, node, cluster, rack }: MemoryModuleProps) {
  const select = useExplorerStore((state) => state.select)
  const openBlueprint = useExplorerStore((state) => state.openMemoryBlueprint)

  const handleClick = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation()
    select({ kind: 'memory', id: descriptor.id }, { memoryInfo: descriptor })
  }

  const handleDoubleClick = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation()
    select({ kind: 'memory', id: descriptor.id }, { memoryInfo: descriptor })
    openBlueprint({
      scope: 'node',
      descriptor,
      node,
      clusterName: cluster.name,
      rackName: rack.name
    })
  }

  return (
    <group position={position} rotation={rotation} onClick={handleClick} onDoubleClick={handleDoubleClick}>
      <mesh>
        <boxGeometry args={[0.45, 0.22, 1.1]} />
        <meshStandardMaterial
          color={selected ? dimmSelected : dimmColor}
          emissive={selected ? '#3bcf96' : '#0f3c2b'}
          roughness={0.35}
          metalness={0.2}
        />
        <Edges scale={1.02} color="#6ff3c4" />
      </mesh>
      <Text position={[0, 0.35, 0]} fontSize={0.22} color="#bfe9da" anchorX="center" anchorY="bottom" billboard>
        {descriptor.label}
      </Text>
    </group>
  )
}

type GPUCardProps = {
  gpu: NodeSpec['gpus'][number]
  position: [number, number, number]
  node: NodeSpec
  cluster: ClusterSpec
  rack: RackSpec
  selected: boolean
  utilText: string
  hbmText: string
}

function GPUCard({ gpu, position, node, cluster, rack, selected, utilText, hbmText }: GPUCardProps) {
  const select = useExplorerStore((state) => state.select)
  const enterGpu = useExplorerStore((state) => state.enterGpu)

  const handleClick = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation()
    select({ kind: 'gpu', id: gpu.uuid })
  }

  const handleDoubleClick = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation()
    select({ kind: 'gpu', id: gpu.uuid })
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

  const top = position[1] + 0.45

  return (
    <group position={position}>
      <mesh position={[0, 0, 0]} onClick={handleClick} onDoubleClick={handleDoubleClick}>
        <boxGeometry args={[1.6, 0.5, 4.4]} />
        <meshStandardMaterial
          color={selected ? gpuSelectedColor : gpuBaseColor}
          emissive={selected ? '#6da4ff' : '#0a2c78'}
          roughness={0.38}
        />
        <Edges scale={1.01} color="#9ad7ff" />
      </mesh>
      <Text position={[0, top, 0]} fontSize={0.34} color="#d6e4ff" anchorX="center" anchorY="bottom" billboard>
        {gpu.name}
      </Text>
      <Text position={[0, top + 0.42, 0]} fontSize={0.26} color="#8fb7ff" anchorX="center" anchorY="bottom" billboard>
        {utilText}
      </Text>
      <Text position={[0, top + 0.72, 0]} fontSize={0.24} color="#8fb7ff" anchorX="center" anchorY="bottom" billboard>
        {hbmText}
      </Text>
    </group>
  )
}

type NodeSceneProps = {
  node: NodeSpec
  cluster: ClusterSpec
  rack: RackSpec
}

export function NodeScene({ node, cluster, rack }: NodeSceneProps) {
  const select = useExplorerStore((state) => state.select)
  const selection = useExplorerStore((state) => state.selection)
  const memoryInfo = useExplorerStore((state) => state.memoryInfo)
  const gpuMetrics = useMetricsStore((state) => state.gpu)
  const nodeMetrics = useMetricsStore((state) => state.node[node.id])
  const nodeLabel = `${cluster.name} / ${rack.name} / ${node.hostname}`

  const nvSwitchPosition = new Vector3(0, 1.2, 0)
  const chassisSelectedState = selection?.kind === 'node' && selection.id === node.id
  const selectedMemoryId = selection?.kind === 'memory' ? selection.id : undefined

  const gpuLayout = useMemo(() => {
    const startX = -4.8
    const spacingX = 3.2
    const rowZ = [-1.9, 1.9]
    return node.gpus.map((gpu, index) => {
      const column = index % 4
      const row = Math.floor(index / 4)
      const x = startX + column * spacingX
      const z = rowZ[row] ?? 0
      return { gpu, position: [x, 1.0, z] as [number, number, number] }
    })
  }, [node.gpus])

  const memoryModules = useMemo(() => {
    const modules = 16
    const perModuleCapacity = node.systemMemoryGB / modules
    const descriptors: {
      descriptor: MemoryDescriptor
      position: [number, number, number]
      rotation: [number, number, number]
    }[] = []
    const zSpacing = 1.0
    const startZ = -3.5
    for (let idx = 0; idx < modules; idx += 1) {
      const side = idx < modules / 2 ? -1 : 1
      const offsetIdx = idx % (modules / 2)
      const z = startZ + offsetIdx * zSpacing
      const x = side * 6.4
      const descriptor: MemoryDescriptor = {
        id: `memory:${node.id}:dimm${idx}`,
        scope: 'node',
        parentId: node.id,
        label: `DIMM ${idx + 1}`,
        type: 'DDR5-5600 DIMM',
        capacity: `${perModuleCapacity.toFixed(0)} GB`,
        bandwidth: '≈45 GB/s peak per DIMM',
        description:
          'One of sixteen DDR5 channels feeding the dual Intel Xeon 8570 CPUs; contributes to the 4 TB system RAM envelope.'
      }
      descriptors.push({
        descriptor,
        position: [x, 1.15, z] as [number, number, number],
        rotation: [0, side > 0 ? Math.PI / 2 : -Math.PI / 2, 0] as [number, number, number]
      })
    }
    return descriptors
  }, [node.id, node.systemMemoryGB])

  const gpuCards = gpuLayout.map(({ gpu, position }) => {
    const metrics = gpuMetrics[gpu.uuid]
    const utilText = metrics ? `Util ${(metrics.util * 100).toFixed(0)}%` : 'Util —'
    const hbmUsed = metrics ? `${metrics.memUsedGB.toFixed(0)}/${gpu.memoryGB}` : `${gpu.memoryGB}`
    const hbmText = `HBM ${hbmUsed} GB`
    const nvlinkFrom = new Vector3(position[0], position[1] + 0.4, position[2])
    const nvlinkTo = nvSwitchPosition.clone()
    nvlinkTo.y += 0.2

    return (
      <group key={gpu.uuid}>
        <GPUCard
          gpu={gpu}
          position={position}
          node={node}
          cluster={cluster}
          rack={rack}
          selected={selection?.kind === 'gpu' && selection.id === gpu.uuid}
          utilText={utilText}
          hbmText={hbmText}
        />
        <NVLinkArc from={nvlinkFrom} to={nvlinkTo} capacity={node.nvlinkSwitchAggregateTBs * 1024} />
      </group>
    )
  })

  const handleNodeClick = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation()
    select({ kind: 'node', id: node.id })
  }

  return (
    <group>
      <mesh position={[0, 0.4, 0]} onClick={handleNodeClick}>
        <boxGeometry args={[16, 0.8, 10]} />
        <meshStandardMaterial
          color={chassisSelectedState ? chassisSelected : chassisColor}
          emissive={chassisSelectedState ? '#2343aa' : '#0b101b'}
          metalness={0.35}
          roughness={0.6}
        />
        <Edges scale={1.001} color="#3f66ff" />
      </mesh>

      <mesh position={[0, 1.15, -4.3]}>
        <boxGeometry args={[14.4, 0.3, 1.2]} />
        <meshStandardMaterial color="#1c2738" roughness={0.55} metalness={0.2} />
      </mesh>

      <mesh position={[0, 1.6, -4.5]}>
        <boxGeometry args={[4.2, 0.8, 0.9]} />
        <meshStandardMaterial color="#3f4d64" emissive="#182035" roughness={0.4} />
      </mesh>
      <Text position={[0, 2.25, -4.5]} fontSize={0.32} color="#9fb8ff" anchorX="center" anchorY="middle" billboard>
        Front Service Bay
      </Text>

      <mesh position={[0, 1.2, 0]}>
        <cylinderGeometry args={[1.8, 1.8, 0.5, 48]} />
        <meshStandardMaterial color="#3a9efd" emissive="#1a4dd9" metalness={0.6} roughness={0.4} />
        <Edges scale={1.01} color="#88c8ff" />
      </mesh>
      <Text position={[0, 2, 0]} fontSize={0.38} color="#b4c9ff" anchorX="center" anchorY="middle" billboard>
        NVSwitch Hub
      </Text>

      <mesh position={[0, 1.1, 3.9]}>
        <boxGeometry args={[4, 0.6, 1.6]} />
        <meshStandardMaterial color="#2f3c52" emissive="#1c273a" roughness={0.45} />
        <Edges scale={1.01} color="#6aa0ff" />
      </mesh>
      <Text position={[0, 1.9, 3.9]} fontSize={0.32} color="#9fb8ff" anchorX="center" anchorY="middle" billboard>
        Dual BlueField-3 &amp; ConnectX-7 IO
      </Text>

      {gpuCards}

      {memoryModules.map(({ descriptor, position, rotation }) => (
        <MemoryModule
          key={descriptor.id}
          descriptor={descriptor}
          position={position}
          rotation={rotation}
          selected={selectedMemoryId === descriptor.id}
          node={node}
          cluster={cluster}
          rack={rack}
        />
      ))}

      <mesh position={[-6.2, 1.05, -4.0]}>
        <boxGeometry args={[2.2, 0.6, 2.4]} />
        <meshStandardMaterial color="#2b3348" emissive="#121a2a" roughness={0.45} />
        <Edges scale={1.01} color="#6a8bff" />
      </mesh>
      <Text position={[-6.2, 1.9, -4.0]} fontSize={0.34} color="#95b6ff" anchorX="center" anchorY="middle" billboard>
        CPU 0
      </Text>

      <mesh position={[6.2, 1.05, -4.0]}>
        <boxGeometry args={[2.2, 0.6, 2.4]} />
        <meshStandardMaterial color="#2b3348" emissive="#121a2a" roughness={0.45} />
        <Edges scale={1.01} color="#6a8bff" />
      </mesh>
      <Text position={[6.2, 1.9, -4.0]} fontSize={0.34} color="#95b6ff" anchorX="center" anchorY="middle" billboard>
        CPU 1
      </Text>

      <Text position={[0, 3.2, 0]} fontSize={0.78} color="#8fb7ff" anchorX="center" anchorY="middle" billboard>
        {nodeLabel}
      </Text>

      <Text position={[0, 2.4, 2.8]} fontSize={0.36} color="#8fe9d2" anchorX="center" anchorY="middle" billboard>
        System RAM Used: {nodeMetrics ? `${nodeMetrics.memoryUsedGB.toFixed(0)} / ${node.systemMemoryGB} GB` : '—'}
      </Text>

      <Text position={[0, 2.8, -2.6]} fontSize={0.32} color="#8fb7ff" anchorX="center" anchorY="middle" billboard>
        NVLink Aggregate: {node.nvlinkSwitchAggregateTBs.toFixed(1)} TB/s
      </Text>

      {memoryInfo && memoryInfo.scope === 'node' && memoryInfo.parentId === node.id && selectedMemoryId && (
        <Text position={[0, 2.6, 0]} fontSize={0.4} color="#bfe9da" anchorX="center" anchorY="middle" billboard>
          {memoryInfo.label}: {memoryInfo.capacity}
        </Text>
      )}
    </group>
  )
}
