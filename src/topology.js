const WARP_SIZE = 32;
const MAX_THREADS_PER_BLOCK = 1024;
const WARPS_PER_BLOCK = MAX_THREADS_PER_BLOCK / WARP_SIZE;
const BLOCKS_PER_GRID = 16;

const clampCount = (value, max = 512) => Math.max(1, Math.min(Math.round(value), max));

const buildThreads = (count, parentId) =>
  Array.from({ length: clampCount(count) }, (_, idx) => ({
    id: `${parentId}-thread-${idx}`,
    name: `Thread ${idx + 1}`,
    type: 'thread',
    count,
    meta: {
      lane: idx % WARP_SIZE,
      explanation: 'CUDA thread executing one element of your workload.'
    },
    children: []
  }));

const buildWarps = (threadCount, parentId) => {
  const warps = Math.ceil(threadCount / WARP_SIZE);
  return Array.from({ length: clampCount(warps, 128) }, (_, idx) => {
    const warpId = `${parentId}-warp-${idx}`;
    return {
      id: warpId,
      name: `Warp ${idx + 1}`,
      type: 'warp',
      count: warps,
      meta: {
        size: WARP_SIZE,
        explanation: 'Warp of 32 threads scheduled together on a CUDA core.'
      },
      children: buildThreads(threadCount / warps, warpId)
    };
  });
};

const buildBlocks = (warpCount, threadCount, parentId) => {
  const blocks = Math.ceil((warpCount || 1) / WARPS_PER_BLOCK);
  return Array.from({ length: clampCount(blocks, 64) }, (_, idx) => {
    const blockId = `${parentId}-block-${idx}`;
    const warpsPerBlock = warpCount / blocks;
    const threadsPerBlock = Math.min(MAX_THREADS_PER_BLOCK, Math.round(threadCount / blocks));
    return {
      id: blockId,
      name: `Block ${idx + 1}`,
      type: 'block',
      count: blocks,
      meta: {
        sharedMemory: `${Math.round(48 / blocks)} KB (approx.)`,
        explanation: 'Thread block scheduled on a Streaming Multiprocessor.'
      },
      children: buildWarps(warpsPerBlock * WARP_SIZE, blockId).map((warp, warpIndex) => ({
        ...warp,
        meta: {
          ...warp.meta,
          threads: threadsPerBlock / (warpCount ? warpCount / blocks : 1),
          explanation: 'Warp inside this block sharing resources and synchronization.'
        }
      }))
    };
  });
};

const buildGrids = (blockCount, threadCount, parentId) => {
  const grids = Math.ceil((blockCount || 1) / BLOCKS_PER_GRID);
  return Array.from({ length: clampCount(grids, 32) }, (_, idx) => {
    const gridId = `${parentId}-grid-${idx}`;
    const blocksPerGrid = Math.max(1, Math.round(blockCount / grids));
    return {
      id: gridId,
      name: `Grid ${idx + 1}`,
      type: 'grid',
      count: grids,
      meta: {
        blocksPerGrid,
        explanation: 'Grid launched from a kernel call encompassing many blocks.'
      },
      children: buildBlocks(blocksPerGrid, threadCount / grids, gridId)
    };
  });
};

const buildDevice = (deviceIndex, deviceConfig, parentId) => {
  const deviceId = `${parentId}-device-${deviceIndex}`;
  const { threadCount, warpCount, blockCount } = deviceConfig;
  return {
    id: deviceId,
    name: `GPU ${deviceIndex + 1}`,
    type: 'device',
    count: deviceConfig.deviceCount,
    meta: {
      smCount: deviceConfig.smCount,
      memory: deviceConfig.memory,
      explanation: 'CUDA device executing kernels in parallel.'
    },
    children: buildGrids(blockCount, threadCount, deviceId)
  };
};

const buildNode = (nodeIndex, nodeConfig, parentId) => {
  const nodeId = `${parentId}-node-${nodeIndex}`;
  return {
    id: nodeId,
    name: `Server ${nodeIndex + 1}`,
    type: 'server',
    count: nodeConfig.nodeCount,
    meta: {
      interconnect: nodeConfig.interconnect,
      explanation: 'Host server orchestrating multiple GPUs and participating in distributed training.'
    },
    children: Array.from({ length: nodeConfig.gpusPerNode }, (_, gpuIdx) =>
      buildDevice(gpuIdx, nodeConfig, nodeId)
    )
  };
};

export const buildTopology = (config) => {
  const { datasetSize, batchSize, gpus, nodes } = config;

  const effectiveDataset = Math.max(datasetSize, batchSize);
  const threadCount = clampCount(Math.ceil(effectiveDataset / Math.max(1, batchSize / 4)) * batchSize, 4096);
  const warpCount = Math.ceil(threadCount / WARP_SIZE);
  const blockCount = Math.ceil(warpCount / WARPS_PER_BLOCK);

  const smCount = Math.max(8, Math.min(128, Math.round(threadCount / 128)));
  const deviceMemory = `${Math.max(24, Math.min(120, Math.round(effectiveDataset / 1024)))} GB`;

  const nodeConfig = {
    nodeCount: nodes,
    gpusPerNode: Math.max(1, Math.round(gpus / nodes)),
    deviceCount: gpus,
    smCount,
    memory: deviceMemory,
    interconnect: nodes > 1 ? 'InfiniBand / 100GbE' : 'PCIe Gen4',
    threadCount,
    warpCount,
    blockCount
  };

  return {
    id: 'distributed-system',
    name: 'Distributed Training System',
    type: 'distributed-system',
    meta: {
      datasetSize,
      batchSize,
      explanation: 'Full AI training stack spanning servers, GPUs, and CUDA execution hierarchy.'
    },
    children: Array.from({ length: nodes }, (_, nodeIdx) => buildNode(nodeIdx, nodeConfig, 'distributed-system'))
  };
};

export const deriveMetrics = (config) => {
  const topology = buildTopology(config);
  const nodeCount = topology.children.length;
  const gpuCount = topology.children.reduce((acc, node) => acc + node.children.length, 0);
  const gridCount = topology.children.flatMap((node) => node.children).reduce((acc, device) => acc + device.children.length, 0);
  const blockCount = topology.children
    .flatMap((node) => node.children)
    .flatMap((device) => device.children)
    .reduce((acc, grid) => acc + grid.children.length, 0);
  const warpCount = topology.children
    .flatMap((node) => node.children)
    .flatMap((device) => device.children)
    .flatMap((grid) => grid.children)
    .reduce((acc, block) => acc + block.children.length, 0);
  const threadCount = config.datasetSize * config.batchSize;

  return {
    nodes: nodeCount,
    gpus: gpuCount,
    grids: gridCount,
    blocks: blockCount,
    warps: warpCount,
    threads: threadCount
  };
};
