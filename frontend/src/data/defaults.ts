import type { ClusterSpec, GPUSpec, Topology } from '../types'

const GPU_MEMORY_GB = 180
const GPU_HBM_BW_TBS = 7.7
const GPU_NVLINK_TBS = 1.8

function createGpu(nodeId: string, index: number): GPUSpec {
  return {
    uuid: `GPU-${nodeId}-${index.toString().padStart(2, '0')}`,
    name: `GPU-${index}`,
    model: 'B200',
    memoryGB: GPU_MEMORY_GB,
    hbmBandwidthTBs: GPU_HBM_BW_TBS,
    nvlinkTBs: GPU_NVLINK_TBS,
    migSupported: true,
    migGuide: 'Guide: split memory ≈1/8, SM ≈1/7 per slice (MIG)',
    l2CacheMB: 126
  }
}

function createNode(clusterId: string, rackId: string, nodeIndex: number, rackSlot: number) {
  const nodeId = `${clusterId}-n${nodeIndex}`
  const gpus = Array.from({ length: 8 }, (_, idx) => createGpu(nodeId, idx))
  return {
    id: nodeId,
    hostname: `${clusterId}-${rackId}-node-${nodeIndex}`,
    vendor: 'NVIDIA' as const,
    model: 'DGX B200' as const,
    rackPosition: rackSlot,
    cpu: { model: 'Intel Xeon 8570' as const, sockets: 2, coresTotal: 112 },
    systemMemoryGB: 4096,
    gpus,
    nvlinkSwitchId: `${nodeId}-nvswitch`,
    nvlinkSwitchAggregateTBs: 14.4,
    networking: {
      nics: [{ model: 'ConnectX-7', speedGbps: 400, count: 4 }],
      dpus: [{ model: 'BlueField-3', ports: 2 }]
    },
    storage: { os: '2x 1.92 TB NVMe (RAID1)', data: 'Expansion NVMe bays available' }
  }
}

function createCluster(id: string): ClusterSpec {
  const racks = ['r0', 'r1'].map((rackId, rackIndex) => ({
    id: rackId,
    name: `Rack ${rackIndex}`,
    nodes: Array.from({ length: 2 }, (_, nodeIdx) => createNode(id, rackId, rackIndex * 2 + nodeIdx, nodeIdx))
  }))

  const nodes = racks.flatMap((rack) => rack.nodes)
  const links = nodes.map((node, index) => {
    const peer = nodes[(index + 1) % nodes.length]
    return {
      id: `ib-${node.id}-${peer.id}`,
      type: 'IB' as const,
      from: `${node.id}:ib0`,
      to: `${peer.id}:ib0`,
      capacityGbps: 400
    }
  })

  return {
    id,
    name: `${id.toUpperCase()} Blackwell Pod`,
    racks,
    links
  }
}

export const demoTopology: Topology = {
  name: 'Lab Blackwell Region',
  clusters: [createCluster('c0')]
}

export type { Topology }
