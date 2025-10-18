import type { Kernel, Topology } from '../types'

const BASE = ''

async function request<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) {
    throw new Error(`Request failed: ${res.status}`)
  }
  return res.json() as Promise<T>
}

export const fetchTopology = () => request<Topology>('/v1/topology')

export const fetchKernels = () => request<{ kernels: Kernel[] }>('/v1/kernels')

export type SearchResult = { type: string; id: string }

export async function search(query: string) {
  const data = await request<{ results: SearchResult[] }>(`/v1/search?q=${encodeURIComponent(query)}`)
  return data.results
}
