import type { Kernel, Topology } from '../types'
import { demoTopology } from '../data/defaults'

const BASE_URL = import.meta.env.VITE_BACKEND_HTTP ?? ''

async function request<T>(path: string): Promise<T> {
  if (!BASE_URL) {
    throw new Error('Backend URL not configured')
  }
  const res = await fetch(`${BASE_URL}${path}`)
  if (!res.ok) {
    throw new Error(`Request failed: ${res.status}`)
  }
  return res.json() as Promise<T>
}

export async function fetchTopology(): Promise<Topology> {
  if (!BASE_URL) {
    return demoTopology
  }
  try {
    return await request<Topology>('/v1/topology')
  } catch (error) {
    console.warn('Falling back to demo topology:', error)
    return demoTopology
  }
}

export async function fetchKernels(): Promise<{ kernels: Kernel[] }> {
  if (!BASE_URL) {
    return { kernels: [] }
  }
  return request<{ kernels: Kernel[] }>('/v1/kernels')
}

export type SearchResult = { type: string; id: string }

export async function search(query: string) {
  if (!BASE_URL) return []
  const data = await request<{ results: SearchResult[] }>(`/v1/search?q=${encodeURIComponent(query)}`)
  return data.results
}
