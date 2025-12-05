const WARP_SIZE = 32;
const MAX_THREADS_PER_BLOCK = 1024;
const WARPS_PER_BLOCK = MAX_THREADS_PER_BLOCK / WARP_SIZE;
const BLOCKS_PER_GRID = 16;

export const LEVEL_LABELS = {
  'distributed-system': 'System',
  server: 'Server',
  device: 'GPU',
  grid: 'Grid',
  block: 'Block',
  warp: 'Warp',
  thread: 'Thread'
};

const clampCount = (value, max = 512) => Math.max(1, Math.min(Math.round(value), max));

const asPercentage = (value) => `${Math.round(value * 100)}%`;

const buildThreads = (count, parentId, kernelDurationMs) =>
  Array.from({ length: clampCount(count) }, (_, idx) => ({
    id: `${parentId}-thread-${idx}`,
    name: `Thread ${idx + 1}`,
    type: 'thread',
    count,
    meta: {
      lane: idx % WARP_SIZE,
      workloadShare: `${(100 / count).toFixed(2)}% of kernel work`,
      latency: `${(kernelDurationMs / count).toFixed(3)} ms (est.)`,
      explanation: 'CUDA thread executing one element of your workload.'
    },
    children: []
  }));

const buildWarps = (threadCount, parentId, kernelDurationMs) => {
  const warps = Math.ceil(threadCount / WARP_SIZE);
  const effectiveWarps = clampCount(warps, 128);
  return Array.from({ length: effectiveWarps }, (_, idx) => {
    const warpId = `${parentId}-warp-${idx}`;
    const threadsInWarp = Math.min(WARP_SIZE, Math.ceil(threadCount / effectiveWarps));
    return {
      id: warpId,
      name: `Warp ${idx + 1}`,
      type: 'warp',
      count: warps,
      meta: {
        threadsPerWarp: threadsInWarp,
        scheduling: 'SIMT (single instruction, multiple threads)',
        executionWindow: `${(kernelDurationMs / effectiveWarps).toFixed(2)} ms slice`,
        explanation: 'Warp of 32 threads scheduled together on a CUDA core.'
      },
      children: buildThreads(threadsInWarp, warpId, kernelDurationMs)
    };
  });
};

const buildBlocks = (warpCount, threadCount, parentId, kernelDurationMs) => {
  const blocks = Math.ceil((warpCount || 1) / WARPS_PER_BLOCK);
  const effectiveBlocks = clampCount(blocks, 64);
  return Array.from({ length: effectiveBlocks }, (_, idx) => {
    const blockId = `${parentId}-block-${idx}`;
    const warpsPerBlock = Math.max(1, Math.round(warpCount / effectiveBlocks));
    const threadsPerBlock = Math.min(
      MAX_THREADS_PER_BLOCK,
      Math.max(WARP_SIZE, Math.round(threadCount / effectiveBlocks))
    );
    const sharedMemoryPerBlock = Math.round(64 / effectiveBlocks);
    return {
      id: blockId,
      name: `Block ${idx + 1}`,
      type: 'block',
      count: blocks,
      meta: {
        sharedMemory: `${sharedMemoryPerBlock} KB (approx.)`,
        warpsPerBlock,
        occupancy: asPercentage(Math.min(1, threadsPerBlock / MAX_THREADS_PER_BLOCK)),
        explanation: 'Thread block scheduled on a Streaming Multiprocessor.'
      },
      children: buildWarps(warpsPerBlock * WARP_SIZE, blockId, kernelDurationMs).map((warp) => ({
        ...warp,
        meta: {
          ...warp.meta,
          threadsPerWarp: Math.min(WARP_SIZE, threadsPerBlock / warpsPerBlock),
          explanation: 'Warp inside this block sharing resources and synchronization.'
        }
      }))
    };
  });
};

const buildGrids = (blockCount, threadCount, parentId, kernelDurationMs) => {
  const grids = Math.ceil((blockCount || 1) / BLOCKS_PER_GRID);
  const effectiveGrids = clampCount(grids, 32);
  return Array.from({ length: effectiveGrids }, (_, idx) => {
    const gridId = `${parentId}-grid-${idx}`;
    const blocksPerGrid = Math.max(1, Math.round(blockCount / effectiveGrids));
    return {
      id: gridId,
      name: `Grid ${idx + 1}`,
      type: 'grid',
      count: grids,
      meta: {
        blocksPerGrid,
        launches: `${Math.max(1, Math.round(threadCount / (blocksPerGrid * WARP_SIZE)))} kernels`,
        explanation: 'Grid launched from a kernel call encompassing many blocks.'
      },
      children: buildBlocks(blocksPerGrid * WARPS_PER_BLOCK, threadCount / effectiveGrids, gridId, kernelDurationMs)
    };
  });
};

const buildDevice = (deviceIndex, deviceConfig, parentId) => {
  const deviceId = `${parentId}-device-${deviceIndex}`;
  const { threadCount, warpCount, blockCount, kernelDurationMs } = deviceConfig;
  const grids = buildGrids(blockCount, threadCount, deviceId, kernelDurationMs);
  const occupancy = Math.min(1, warpCount / (deviceConfig.smCount * WARPS_PER_BLOCK));
  return {
    id: deviceId,
    name: `GPU ${deviceIndex + 1}`,
    type: 'device',
    count: deviceConfig.deviceCount,
    meta: {
      smCount: deviceConfig.smCount,
      memory: deviceConfig.memory,
      bandwidth: `${deviceConfig.bandwidth} GB/s (est.)`,
      occupancy: asPercentage(occupancy),
      explanation: 'CUDA device executing kernels in parallel.'
    },
    children: grids
  };
};

const buildNode = (nodeIndex, nodeConfig, parentId) => {
  const nodeId = `${parentId}-node-${nodeIndex}`;
  const devices = Array.from({ length: nodeConfig.gpusPerNode }, (_, gpuIdx) =>
    buildDevice(gpuIdx, nodeConfig, nodeId)
  );
  return {
    id: nodeId,
    name: `Server ${nodeIndex + 1}`,
    type: 'server',
    count: nodeConfig.nodeCount,
    meta: {
      interconnect: nodeConfig.interconnect,
      network: `${nodeConfig.networkBandwidth} Gbps fabric`,
      explanation: 'Host server orchestrating multiple GPUs and participating in distributed training.'
    },
    children: devices
  };
};

export const buildTopology = (config) => {
  const { datasetSize, batchSize, gpus, nodes } = config;

  const effectiveDataset = Math.max(datasetSize, batchSize);
  const kernelDurationMs = Math.max(4, Math.log2(effectiveDataset + batchSize)) * 0.75;
  const threadCount = clampCount(Math.ceil(effectiveDataset / Math.max(1, batchSize / 4)) * batchSize, 8192);
  const warpCount = Math.ceil(threadCount / WARP_SIZE);
  const blockCount = Math.ceil(warpCount / WARPS_PER_BLOCK);

  const smCount = Math.max(8, Math.min(128, Math.round(threadCount / 128)));
  const deviceMemory = `${Math.max(24, Math.min(120, Math.round(effectiveDataset / 1024)))} GB`;
  const bandwidth = Math.max(600, Math.min(1500, Math.round(threadCount / 10)));

  const nodeConfig = {
    nodeCount: nodes,
    gpusPerNode: Math.max(1, Math.round(gpus / nodes)),
    deviceCount: gpus,
    smCount,
    memory: deviceMemory,
    bandwidth,
    interconnect: nodes > 1 ? 'InfiniBand / 100GbE' : 'PCIe Gen4',
    networkBandwidth: nodes > 1 ? 200 : 64,
    threadCount,
    warpCount,
    blockCount,
    kernelDurationMs
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

const sumByType = (topology, type) => {
  if (!topology) return 0;
  const isMatch = topology.type === type ? 1 : 0;
  const childCount = (topology.children ?? []).reduce((acc, child) => acc + sumByType(child, type), 0);
  return isMatch + childCount;
};

export const deriveMetrics = (config) => {
  const topology = buildTopology(config);
  const nodeCount = topology.children.length;
  const gpuCount = topology.children.reduce((acc, node) => acc + node.children.length, 0);
  const gridCount = sumByType(topology, 'grid');
  const blockCount = sumByType(topology, 'block');
  const warpCount = sumByType(topology, 'warp');
  const threadCount = config.datasetSize * config.batchSize;

  const peakFlopsPerGpu = 312 * 10 ** 12; // 312 TFLOPs for modern GPUs
  const effectiveFlops = peakFlopsPerGpu * gpuCount * 0.55;
  const memoryGb = Math.max(24, Math.min(120, Math.round(config.datasetSize / 1024))) * gpuCount;
  const bandwidthGb = gpuCount * Math.max(600, Math.min(1500, Math.round(threadCount / 10)));
  const epochSeconds = Math.max(1, (config.datasetSize / Math.max(1, gpuCount * config.batchSize)) * 0.9);

  return {
    nodes: nodeCount,
    gpus: gpuCount,
    grids: gridCount,
    blocks: blockCount,
    warps: warpCount,
    threads: threadCount,
    flops: effectiveFlops,
    memoryGb,
    bandwidthGb,
    epochSeconds
  };
};

const flattenNode = (node, parentId = null, rows = []) => {
  const metaEntries = Object.entries(node.meta ?? {})
    .filter(([key]) => key !== 'explanation')
    .map(([key, value]) => `${key}: ${value}`)
    .join(' | ');
  rows.push({
    id: node.id,
    name: node.name,
    type: node.type,
    parent: parentId,
    meta: metaEntries
  });
  (node.children ?? []).forEach((child) => flattenNode(child, node.id, rows));
  return rows;
};

export const findNodeById = (node, id) => {
  if (!node) return null;
  if (node.id === id) return node;
  for (const child of node.children ?? []) {
    const found = findNodeById(child, id);
    if (found) return found;
  }
  return null;
};

export const topologyToCsv = (topology, options = {}) => {
  const { rootId = null, levelType = null } = options;
  const root = rootId ? findNodeById(topology, rootId) ?? topology : topology;
  const rows = flattenNode(root);
  const filteredRows = levelType ? rows.filter((row) => row.type === levelType) : rows;
  const header = 'id,name,type,parent,meta';
  const body = filteredRows
    .map((row) =>
      [row.id, row.name, row.type, row.parent ?? '', row.meta.replace(/"/g, '""')]
        .map((value) => `"${value}"`)
        .join(',')
    )
    .join('\n');
  return `${header}\n${body}`;
};

export const LEVEL_ORDER = [
  'distributed-system',
  'server',
  'device',
  'grid',
  'block',
  'warp',
  'thread'
];
