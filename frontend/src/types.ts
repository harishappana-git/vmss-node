export type IBLink = {
  id: string
  from: string
  to: string
  type: 'IB'
  capacityGbps: number
}

export type NVLink = {
  id: string
  from: string
  to: string
  type: 'NVLINK'
  capacityGBs: number
}

export type Link = IBLink | NVLink

export type MigProfile = {
  name: string
  memoryGB: number
  smShare: string
}

export type GPUSpec = {
  uuid: string
  name: string
  model: 'B200'
  memoryGB: number
  hbmBandwidthTBs: number
  nvlinkTBs: number
  migSupported: boolean
  migGuide?: string
  l2CacheMB?: number
  powerTargetW?: number
}

export type NodeSpec = {
  id: string
  hostname: string
  vendor: 'NVIDIA'
  model: 'DGX B200'
  rackPosition: number
  cpu: { model: 'Intel Xeon 8570'; sockets: number; coresTotal: number }
  systemMemoryGB: number
  gpus: GPUSpec[]
  nvlinkSwitchId: string
  nvlinkSwitchAggregateTBs: number
  networking: {
    nics: { model: 'ConnectX-7'; speedGbps: number; count: number }[]
    dpus: { model: 'BlueField-3'; ports: number }[]
  }
  storage: { os: string; data?: string }
}

export type RackSpec = {
  id: string
  name: string
  nodes: NodeSpec[]
}

export type ClusterSpec = {
  id: string
  name: string
  racks: RackSpec[]
  links: Link[]
}

export type Topology = {
  name: string
  clusters: ClusterSpec[]
}

export type SelectionKind = 'cluster' | 'rack' | 'node' | 'gpu' | 'link' | 'memory'

export type Selection = {
  kind: SelectionKind
  id: string
}

export type MemoryDescriptor = {
  id: string
  scope: 'node' | 'gpu'
  parentId: string
  label: string
  type: string
  capacity: string
  bandwidth?: string
  description: string
}

export type Breadcrumb = {
  label: string
  kind: SelectionKind | 'rack'
  id: string
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
