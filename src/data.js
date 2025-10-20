export const illustrationConfig = {
  title: 'Integrated Training and Serving Infra',
  leftPane: {
    icons: ['image', 'doc', 'audio'],
  },
  platform: {
    modules: [
      {
        name: 'Model Architecture',
        bullets: ['Transformer / long context / MoE'],
        priority: 'high',
      },
      {
        name: 'Parallelism & Sharding',
        bullets: ['3D (data×tensor×pipeline)', 'FSDP / ZeRO'],
        priority: 'high',
      },
      {
        name: 'High-Performance Kernels & Libraries',
        bullets: ['Fused ops · FlashAttention · cuBLAS/cuDNN'],
      },
      {
        name: 'Compilers & Graph Optimizers',
        bullets: ['torch.compile / XLA / MLIR · fusion'],
      },
      {
        name: 'Optimizer Algorithms',
        bullets: ['Adam / LAMB / Adafactor', 'Large-batch tricks'],
      },
      {
        name: 'Numerical Precision',
        bullets: ['BF16 / FP16 / FP8', 'Loss-scaling & calibration'],
      },
      {
        name: 'DL Framework & Runtime',
        bullets: ['PyTorch · DeepSpeed · Megatron · JAX'],
      },
    ],
  },
  clouds: [
    {
      id: 'privateCloud',
      type: 'primary',
    },
    {
      id: 'managedCloud',
      type: 'secondary',
    },
  ],
  connectors: [
    {
      from: 'dataPane',
      to: 'customization',
      color: 'accent',
      style: 'solid',
      startAnchor: 'right',
      endAnchor: 'left',
      curvature: 120,
      width: 3,
    },
    {
      from: 'customization',
      to: 'platform',
      color: 'accent',
      style: 'dashed',
      startAnchor: 'top',
      endAnchor: 'left',
      curvature: 40,
      width: 2.5,
    },
    {
      from: 'platform',
      to: 'privateCloud',
      color: 'accent',
      style: 'dashed-arrow',
      startAnchor: 'right',
      endAnchor: 'left',
      curvature: 90,
      width: 2.5,
    },
    {
      from: 'privateCloud',
      to: 'managedCloud',
      color: 'ink-muted',
      style: 'dashed',
      startAnchor: 'bottom',
      endAnchor: 'top',
      curvature: 60,
      width: 2,
    },
  ],
};
