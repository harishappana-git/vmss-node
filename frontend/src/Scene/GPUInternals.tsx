import { useEffect, useMemo } from 'react'
import { Text } from '@react-three/drei'
import { Color, TorusGeometry, Vector3 } from 'three'
import type { ThreeEvent } from '@react-three/fiber'
import type { ClusterSpec, GPUSpec, MemoryDescriptor, NodeSpec, RackSpec } from '../types'
import { useExplorerStore } from '../state/selectionStore'
import { focusOn } from '../lib/camera'
import { useMetricsStore } from '../state/metricsStore'

const hbmColor = new Color('#7dd0ff')
const hbmSelectedColor = new Color('#c2f3ff')
const dieColor = new Color('#3f6bff')
const l2Color = new Color('#ffa45c')
const sharedColor = new Color('#ffcf66')
const registersColor = new Color('#7dffb3')
const tmaColor = new Color('#ff7ad1')

type HbmStackProps = {
  position: Vector3
  descriptor: MemoryDescriptor
  selected: boolean
}

function HbmStack({ position, descriptor, selected }: HbmStackProps) {
  const select = useExplorerStore((state) => state.select)

  const handleClick = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation()
    select({ kind: 'memory', id: descriptor.id }, { memoryInfo: descriptor })
  }

  return (
    <group
      position={position.toArray() as [number, number, number]}
      onClick={handleClick}
    >
      <mesh>
        <cylinderGeometry args={[0.8, 0.8, 1.2, 24]} />
        <meshStandardMaterial
          color={selected ? hbmSelectedColor : hbmColor}
          emissive={selected ? '#2f6f9d' : '#1d4f7a'}
          roughness={0.4}
        />
      </mesh>
      <Text position={[0, 1, 0]} fontSize={0.28} color="#c8f2ff" anchorX="center" billboard>
        {descriptor.label}
      </Text>
    </group>
  )
}

type MemoryRingProps = {
  radius: number
  thickness: number
  color: Color
  descriptor: MemoryDescriptor
  selected: boolean
  label: string
  y: number
  emissiveBase: string
  emissiveSelected: string
}

function MemoryRing({ radius, thickness, color, descriptor, selected, label, y, emissiveBase, emissiveSelected }: MemoryRingProps) {
  const select = useExplorerStore((state) => state.select)
  const geometry = useMemo(() => new TorusGeometry(radius, thickness, 32, 64), [radius, thickness])

  useEffect(() => () => geometry.dispose(), [geometry])

  const handleClick = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation()
    select({ kind: 'memory', id: descriptor.id }, { memoryInfo: descriptor })
  }

  return (
    <group>
      <mesh geometry={geometry} rotation={[Math.PI / 2, 0, 0]} position={[0, y, 0]} onClick={handleClick}>
        <meshStandardMaterial
          color={selected ? color.clone().offsetHSL(0, 0, 0.12) : color}
          emissive={selected ? emissiveSelected : emissiveBase}
          roughness={0.35}
        />
      </mesh>
      <Text position={[0, y + 0.6, 0]} fontSize={0.32} color="#ffe0b8" anchorX="center" billboard>
        {label}
      </Text>
    </group>
  )
}

type MemoryPlateProps = {
  size: [number, number, number]
  position: [number, number, number]
  color: Color
  descriptor: MemoryDescriptor
  selected: boolean
  label: string
  emissiveBase: string
  emissiveSelected: string
}

function MemoryPlate({ size, position, color, descriptor, selected, label, emissiveBase, emissiveSelected }: MemoryPlateProps) {
  const select = useExplorerStore((state) => state.select)

  const handleClick = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation()
    select({ kind: 'memory', id: descriptor.id }, { memoryInfo: descriptor })
  }

  return (
    <group position={position}>
      <mesh onClick={handleClick}>
        <boxGeometry args={size} />
        <meshStandardMaterial
          color={selected ? color.clone().offsetHSL(0, 0, 0.12) : color}
          emissive={selected ? emissiveSelected : emissiveBase}
          roughness={0.35}
          metalness={0.15}
        />
      </mesh>
      <Text position={[0, size[1] / 2 + 0.3, 0]} fontSize={0.28} color="#d6ffe8" anchorX="center" billboard>
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
  const selection = useExplorerStore((state) => state.selection)
  const memoryInfo = useExplorerStore((state) => state.memoryInfo)
  const header = `${cluster.name} › ${rack.name} › ${node.hostname} › ${gpu.name}`

  useEffect(() => {
    focusOn([0, 5, 12], [0, 0, 0], 32)
  }, [])

  const gpuMetrics = useMetricsStore((state) => state.gpu[gpu.uuid])
  const selectedMemoryId = selection?.kind === 'memory' ? selection.id : undefined
  const isGpuSelected = selection?.kind === 'gpu' && selection.id === gpu.uuid

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

  const hbmDescriptors = useMemo(() => {
    const stacks = hbmPositions.length
    const perStackCapacity = gpu.memoryGB / stacks
    const perStackBandwidth = (gpu.hbmBandwidthTBs * 1024) / stacks
    return hbmPositions.map((position, idx) => ({
      position,
      descriptor: {
        id: `memory:${gpu.uuid}:hbm${idx}`,
        scope: 'gpu',
        parentId: gpu.uuid,
        label: `HBM Stack ${idx + 1}`,
        type: 'HBM3e',
        capacity: `≈${perStackCapacity.toFixed(1)} GB`,
        bandwidth: `≈${perStackBandwidth.toFixed(0)} GB/s`,
        description:
          'HBM3e stack bonded on-package delivering the ultra-wide memory bandwidth that feeds Blackwell SMs.'
      } satisfies MemoryDescriptor
    }))
  }, [gpu.hbmBandwidthTBs, gpu.memoryGB, gpu.uuid, hbmPositions])

  const l2Descriptor: MemoryDescriptor = useMemo(
    () => ({
      id: `memory:${gpu.uuid}:l2`,
      scope: 'gpu',
      parentId: gpu.uuid,
      label: 'L2 Cache Ring',
      type: 'Unified L2 cache',
      capacity: gpu.l2CacheMB ? `${gpu.l2CacheMB} MB` : '≈126 MB class',
      bandwidth: '≈12 TB/s on-die fabric',
      description: 'High-capacity L2 cache linking SM partitions, NVLink, and HBM with low-latency access.'
    }),
    [gpu.l2CacheMB, gpu.uuid]
  )

  const sharedDescriptor: MemoryDescriptor = useMemo(
    () => ({
      id: `memory:${gpu.uuid}:shared`,
      scope: 'gpu',
      parentId: gpu.uuid,
      label: 'SM Shared / L1 Cache',
      type: 'Configurable shared memory + L1',
      capacity: '≈256 KB per SM (configurable)',
      bandwidth: 'Tens of TB/s intra-SM',
      description: 'On-die scratchpad adjacent to each SM for warp-synchronous data, backing tensor cores and block-level reuse.'
    }),
    [gpu.uuid]
  )

  const registersDescriptor: MemoryDescriptor = useMemo(
    () => ({
      id: `memory:${gpu.uuid}:registers`,
      scope: 'gpu',
      parentId: gpu.uuid,
      label: 'Register File',
      type: '64-bit register file',
      capacity: '≈256 KB per SM (aggregate multi-MB)',
      bandwidth: 'Single-cycle SM access',
      description: 'Per-thread register banks providing immediate operands for warp execution and tensor core pipelines.'
    }),
    [gpu.uuid]
  )

  const tmaDescriptor: MemoryDescriptor = useMemo(
    () => ({
      id: `memory:${gpu.uuid}:tma`,
      scope: 'gpu',
      parentId: gpu.uuid,
      label: 'Tensor Memory Accelerator',
      type: 'TMA staging SRAM',
      capacity: 'Multi-MB staging buffers',
      bandwidth: 'Optimized for asynchronous tensor moves',
      description: 'Dedicated memory path that schedules asynchronous tensor moves between HBM, shared memory, and registers.'
    }),
    [gpu.uuid]
  )

  return (
    <group>
      <Text position={[0, 6, 0]} fontSize={0.9} color="#b4c9ff" anchorX="center" billboard>
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
        <meshStandardMaterial
          color={isGpuSelected ? '#253063' : '#1a1f33'}
          emissive={isGpuSelected ? '#1b2b6d' : '#0b0f1f'}
          roughness={0.6}
          metalness={0.2}
        />
      </mesh>
      <mesh position={[0, 0.5, 0]}>
        <boxGeometry args={[4.4, 0.6, 4.4]} />
        <meshStandardMaterial color={dieColor} emissive="#153cff" roughness={0.45} metalness={0.5} />
      </mesh>
      <MemoryRing
        radius={4.8}
        thickness={0.2}
        color={l2Color}
        descriptor={l2Descriptor}
        selected={selectedMemoryId === l2Descriptor.id}
        label="L2 Cache"
        y={0.8}
        emissiveBase="#ff7a1a"
        emissiveSelected="#fbd6a2"
      />
      <MemoryRing
        radius={3.2}
        thickness={0.18}
        color={sharedColor}
        descriptor={sharedDescriptor}
        selected={selectedMemoryId === sharedDescriptor.id}
        label="Shared / L1"
        y={1.05}
        emissiveBase="#d49c1f"
        emissiveSelected="#ffdf8a"
      />
      <MemoryPlate
        size={[2.6, 0.3, 2.6]}
        position={[0, 1.35, 0]}
        color={registersColor}
        descriptor={registersDescriptor}
        selected={selectedMemoryId === registersDescriptor.id}
        label="Register File"
        emissiveBase="#1d4d3c"
        emissiveSelected="#2f6f5c"
      />
      <MemoryPlate
        size={[1.6, 0.25, 1.6]}
        position={[0, 1.8, 0]}
        color={tmaColor}
        descriptor={tmaDescriptor}
        selected={selectedMemoryId === tmaDescriptor.id}
        label="Tensor Memory Accelerator"
        emissiveBase="#44153b"
        emissiveSelected="#7c2d65"
      />
      {hbmDescriptors.map(({ descriptor, position }) => (
        <HbmStack
          key={descriptor.id}
          position={position}
          descriptor={descriptor}
          selected={selectedMemoryId === descriptor.id}
        />
      ))}
      <Text position={[0, -1.2, 0]} fontSize={0.5} color="#9fb8ff" anchorX="center" anchorY="middle" billboard>
        ≈ {gpu.memoryGB} GB HBM3e · {gpu.hbmBandwidthTBs.toFixed(1)} TB/s · NVLink {gpu.nvlinkTBs.toFixed(1)} TB/s
      </Text>
      <Text position={[0, -2, 0]} fontSize={0.38} color="#d6e4ff" anchorX="center" anchorY="middle" billboard>
        MIG: {gpu.migSupported ? gpu.migGuide ?? 'Supported' : 'Disabled'}
      </Text>
      <Text position={[0, -2.6, 0]} fontSize={0.36} color="#8fe9d2" anchorX="center" anchorY="middle" billboard>
        HBM Used: {gpuMetrics ? `${gpuMetrics.memUsedGB.toFixed(1)} GB (${(gpuMetrics.memUsedGB / gpu.memoryGB * 100).toFixed(0)}%)` : '—'}
      </Text>
      <Text position={[0, -3.2, 0]} fontSize={0.34} color="#8fb7ff" anchorX="center" anchorY="middle" billboard>
        Util {gpuMetrics ? `${(gpuMetrics.util * 100).toFixed(0)}%` : '—'} · Temp {gpuMetrics ? `${gpuMetrics.tempC.toFixed(1)} °C` : '—'} · Power {gpuMetrics ? `${gpuMetrics.powerW.toFixed(0)} W` : '—'}
      </Text>
      {memoryInfo && memoryInfo.scope === 'gpu' && memoryInfo.parentId === gpu.uuid && selectedMemoryId && (
        <Text position={[0, 2.6, 0]} fontSize={0.42} color="#ffe0b8" anchorX="center" anchorY="middle" billboard>
          {memoryInfo.label}: {memoryInfo.capacity}
        </Text>
      )}
    </group>
  )
}
