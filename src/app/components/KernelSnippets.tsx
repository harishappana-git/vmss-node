"use client";

import dynamic from "next/dynamic";
import { useMemo } from "react";
import type { Kernel, TransposeMode } from "../lib/sim";

const Monaco = dynamic(() => import("@monaco-editor/react"), { ssr: false });

const KERNEL_SNIPPETS: Record<Kernel, string> = {
  vector_add: `__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}`,
  saxpy: `__global__ void saxpy(float a, const float* x, const float* y, float* z, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    z[idx] = a * x[idx] + y[idx];
  }
}`,
  reduce_sum: `__global__ void reduce_sum(const float* input, float* output, int n) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  float sum = 0.0f;
  if (i < n) sum += input[i];
  if (i + blockDim.x < n) sum += input[i + blockDim.x];
  sdata[tid] = sum;
  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}`,
  transpose: `__global__ void transpose(float* out, const float* in, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < width) {
    out[x * width + y] = in[y * width + x];
  }
}

__global__ void transpose_tiled(float* out, const float* in, int width) {
  __shared__ float tile[32][33];
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  if (x < width && y < width) {
    tile[threadIdx.y][threadIdx.x] = in[y * width + x];
  }
  __syncthreads();
  int transposedX = blockIdx.y * 32 + threadIdx.x;
  int transposedY = blockIdx.x * 32 + threadIdx.y;
  if (transposedX < width && transposedY < width) {
    out[transposedY * width + transposedX] = tile[threadIdx.x][threadIdx.y];
  }
}`,
  matmul_tiled: `#define TILE 16
__global__ void matmul_tiled(const float* A, const float* B, float* C,
                             int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.0f;
  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
    if (row < M && t * TILE + threadIdx.x < K) {
      As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
    } else {
      As[threadIdx.y][threadIdx.x] = 0.0f;
    }
    if (col < N && t * TILE + threadIdx.y < K) {
      Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < M && col < N) {
    C[row * N + col] = acc;
  }
}`,
};

interface KernelSnippetsProps {
  kernel: Kernel;
  transposeMode: TransposeMode;
}

export function KernelSnippets({ kernel, transposeMode }: KernelSnippetsProps) {
  const code = useMemo(() => {
    if (kernel === "transpose") {
      if (transposeMode === "naive") {
        return KERNEL_SNIPPETS.transpose.split("\n\n")[0];
      }
      return KERNEL_SNIPPETS.transpose;
    }
    return KERNEL_SNIPPETS[kernel];
  }, [kernel, transposeMode]);

  return (
    <div className="h-full min-h-[320px] overflow-hidden rounded-lg border border-neutral-800 bg-neutral-900/70">
      <div className="border-b border-neutral-800 px-4 py-2 text-sm font-semibold uppercase tracking-wide text-neutral-400">
        Kernel reference
      </div>
      <Monaco
        height="280px"
        defaultLanguage="cpp"
        theme="vs-dark"
        value={code}
        options={{
          minimap: { enabled: false },
          readOnly: true,
          fontFamily: "JetBrains Mono, Fira Code, monospace",
          fontSize: 13,
          scrollBeyondLastLine: false,
        }}
      />
    </div>
  );
}
