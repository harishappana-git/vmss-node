export type GPU = {
  uuid: string
  model: string
  mig: { enabled: boolean; instances: string[] }
  nvlinkPeers?: string[]
}

export type Node = {
  id: string
  hostname: string
  gpus: GPU[]
  ib: { portGuid: string; speedGbps: number; health: string; utilization: number }
}

export type Link = {
  id: string
  type: 'IB' | 'NVLINK'
  from: string
  to: string
  capacityGbps: number
}

export type Rack = { id: string; nodes: Node[] }

export type Cluster = {
  id: string
  racks: Rack[]
  links: Link[]
}

export type Topology = {
  regionId: string
  clusters: Cluster[]
}

export type GPUFrame = {
  t: number
  topic: string
  util: number
  memUsedGB: number
  nvlinkGBs: number
  pcieTxGBs: number
  tempC: number
  powerW: number
}

export type NodeFrame = {
  t: number
  topic?: string
  data?: unknown
  nodeId?: string
  cpuUtil: number
  memoryUsedGB: number
  ibUtilGbps: number
  jobsRunning: number
}

export type LinkFrame = {
  t: number
  linkId: string
  bwGbps: number
  rttUs: number
  errs?: number
}

export type Kernel = {
  id: string
  name: string
  gridDim: number
  blockDim: number
  occupancy: number
  smEff: number
  dramBwGBs: number
}
