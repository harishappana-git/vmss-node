import { useEffect, useMemo } from 'react'
import { create } from 'zustand'
import { MathUtils, Vector3 } from 'three'
import { useFrame, useThree } from '@react-three/fiber'
import type { OrbitControls } from 'three-stdlib'

const HOME_POSITION: [number, number, number] = [32, 26, 32]
const HOME_TARGET: [number, number, number] = [0, 0, 0]
const HOME_FOV = 45

type CameraState = {
  position: [number, number, number]
  target: [number, number, number]
  fov: number
  zoom(step: number): void
  zoomHome(): void
  zoomFit(): void
  moveTo(position: [number, number, number], target: [number, number, number], fov?: number): void
}

export const useCameraStore = create<CameraState>((set) => ({
  position: HOME_POSITION,
  target: HOME_TARGET,
  fov: HOME_FOV,
  zoom: (step) =>
    set((state) => ({
      fov: MathUtils.clamp(state.fov - step * 5, 28, 70)
    })),
  zoomHome: () => set({ position: HOME_POSITION, target: HOME_TARGET, fov: HOME_FOV }),
  zoomFit: () => set({ position: [24, 20, 24], target: [0, 0, 0], fov: 42 }),
  moveTo: (position, target, fov) => set({ position, target, fov: fov ?? HOME_FOV })
}))

type CameraRigProps = {
  controls: React.RefObject<OrbitControls>
}

export function CameraRig({ controls }: CameraRigProps) {
  const { camera } = useThree()
  const state = useCameraStore()
  const targetVec = useMemo(() => new Vector3(), [])
  const positionVec = useMemo(() => new Vector3(), [])

  useEffect(() => {
    camera.position.set(...state.position)
    camera.lookAt(...state.target)
    camera.fov = state.fov
    camera.updateProjectionMatrix()
    if (controls.current) {
      controls.current.target.set(...state.target)
      controls.current.update()
    }
  }, [])

  useFrame((_, delta) => {
    positionVec.set(...state.position)
    targetVec.set(...state.target)

    camera.position.lerp(positionVec, 1 - Math.pow(0.001, delta))
    camera.fov = MathUtils.damp(camera.fov, state.fov, 4, delta)
    camera.updateProjectionMatrix()

    const orbit = controls.current
    if (orbit) {
      orbit.target.lerp(targetVec, 1 - Math.pow(0.001, delta))
      orbit.update()
    }
  })

  return null
}

export function useCameraTools() {
  const zoom = useCameraStore((state) => state.zoom)
  const zoomHome = useCameraStore((state) => state.zoomHome)
  const zoomFit = useCameraStore((state) => state.zoomFit)
  return {
    zoomIn: () => zoom(1),
    zoomOut: () => zoom(-1),
    zoomHome,
    zoomFit
  }
}

export function focusOn(position: [number, number, number], target: [number, number, number], fov?: number) {
  useCameraStore.getState().moveTo(position, target, fov)
}
