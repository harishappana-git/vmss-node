import { Fragment, useMemo } from 'react'
import { useSelectionStore } from '../state/selectionStore'
import { useMetricsStore } from '../state/metricsStore'

const formatPercent = (value?: number) => (value ?? 0).toLocaleString(undefined, { style: 'percent', maximumFractionDigits: 1 })
const formatNumber = (value?: number, unit = '') => `${(value ?? 0).toFixed(1)} ${unit}`.trim()

export function MetricsOverlay() {
  const selection = useSelectionStore((state) => state.selected)
  const metrics = useMetricsStore((state) => {
    if (!selection) return null
    if (selection.kind === 'gpu') return state.gpu[selection.id] ?? null
    if (selection.kind === 'node') return state.node[selection.id] ?? null
    if (selection.kind === 'link') return state.link[selection.id] ?? null
    return null
  })

  const rows = useMemo(() => {
    if (!selection || !metrics) return []
    switch (selection.kind) {
      case 'gpu':
        return [
          ['Utilization', formatPercent((metrics as any).util)],
          ['Memory Used', formatNumber((metrics as any).memUsedGB, 'GB')],
          ['NVLink', formatNumber((metrics as any).nvlinkGBs, 'GB/s')],
          ['PCIe TX', formatNumber((metrics as any).pcieTxGBs, 'GB/s')],
          ['Temperature', formatNumber((metrics as any).tempC, '°C')],
          ['Power', formatNumber((metrics as any).powerW, 'W')]
        ]
      case 'node':
        return [
          ['CPU Util', formatPercent((metrics as any).cpuUtil)],
          ['Memory', formatNumber((metrics as any).memoryUsedGB, 'GB')],
          ['IB Util', formatNumber((metrics as any).ibUtilGbps, 'Gb/s')],
          ['Jobs', String((metrics as any).jobsRunning ?? 0)]
        ]
      case 'link':
        return [
          ['Bandwidth', formatNumber((metrics as any).bwGbps, 'Gb/s')],
          ['RTT', formatNumber((metrics as any).rttUs, 'µs')]
        ]
      default:
        return []
    }
  }, [metrics, selection])

  if (!selection) return null
  const hasMetrics = rows.length > 0

  return (
    <aside className="metrics-overlay">
      <h2>{selection.kind.toUpperCase()}</h2>
      <p>ID: {selection.id}</p>
      {hasMetrics ? (
        <dl>
          {rows.map(([label, value]) => (
            <Fragment key={label}>
              <dt>{label}</dt>
              <dd>{value}</dd>
            </Fragment>
          ))}
        </dl>
      ) : (
        <p>No live metrics available yet.</p>
      )}
    </aside>
  )
}
