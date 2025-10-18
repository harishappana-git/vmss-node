import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchKernels } from '../api/client'
import type { Topology, Kernel } from '../types'
import { useSelectionStore } from '../state/selectionStore'

type Props = {
  topology: Topology
}

type KernelLayout = {
  kernel: Kernel
  position: [number, number, number]
  blocks: number
}

export function CudaScene({ topology: _topology }: Props) {
  const { data } = useQuery({ queryKey: ['kernels'], queryFn: fetchKernels })
  const kernels = data?.kernels ?? []
  const select = useSelectionStore((state) => state.select)

  const layouts = useMemo<KernelLayout[]>(() => {
    return kernels.map((kernel, index) => ({
      kernel,
      position: [index * 6 - 4, 0, 0],
      blocks: Math.min(kernel.gridDim, 256)
    }))
  }, [kernels])

  return (
    <group>
      <gridHelper args={[40, 20, '#3a4b7a', '#1a2336']} position={[0, -5, 0]} />
      {layouts.map(({ kernel, position, blocks }) => (
        <group key={kernel.id} position={position}>
          <mesh
            position={[0, 2.2, 0]}
            onClick={(event) => {
              event.stopPropagation()
              select({ kind: 'kernel', id: kernel.id })
            }}
          >
            <boxGeometry args={[4, 0.4, 4]} />
            <meshStandardMaterial color={0x4a7dff} emissive={0x0b245a} />
          </mesh>
          <BlocksVisual blocks={blocks} />
        </group>
      ))}
    </group>
  )
}

type BlocksProps = {
  blocks: number
}

function BlocksVisual({ blocks }: BlocksProps) {
  const cells = Math.ceil(Math.sqrt(blocks))
  const spacing = 0.5
  const offset = (cells - 1) * spacing * 0.5
  const instances = []
  for (let i = 0; i < blocks; i++) {
    const x = (i % cells) * spacing - offset
    const z = Math.floor(i / cells) * spacing - offset
    instances.push(
      <mesh key={i} position={[x, 0, z]}
        scale={[0.4, 0.2, 0.4]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color={0x9faef5} emissive={0x1e2d6d} />
      </mesh>
    )
  }
  return <group position={[0, 0, 0]}>{instances}</group>
}
