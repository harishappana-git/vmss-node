import { useEffect, useMemo } from 'react'
import { useMetricsStore } from '../state/metricsStore'
import { demoTopology } from '../data/defaults'

const WS_URL = import.meta.env.VITE_BACKEND_WS as string | undefined

export function useMetricsStream() {
  const update = useMetricsStore((state) => state.ingest)
  const demoEntities = useMemo(() => {
    const nodes = demoTopology.clusters.flatMap((cluster) =>
      cluster.racks.flatMap((rack) => rack.nodes)
    )
    const gpus = nodes.flatMap((node) => node.gpus)
    const links = demoTopology.clusters.flatMap((cluster) => cluster.links)
    return { nodes, gpus, links }
  }, [])
  useEffect(() => {
    if (!WS_URL) return
    const ws = new WebSocket(WS_URL)

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data)
        update(payload)
      } catch (err) {
        console.error('failed to parse frame', err)
      }
    }

    ws.onerror = (event) => {
      console.warn('WebSocket error', event)
    }

    return () => {
      ws.close()
    }
  }, [update, WS_URL])

  useEffect(() => {
    if (WS_URL) return undefined
    let tick = 0
    const interval = setInterval(() => {
      tick += 1
      const now = Date.now()
      demoEntities.gpus.forEach((gpu, index) => {
        const util = 0.55 + 0.35 * Math.abs(Math.sin((tick + index) * 0.15))
        update({
          topic: `gpu.${gpu.uuid}`,
          t: now,
          data: {
            topic: `gpu.${gpu.uuid}`,
            util,
            memUsedGB: gpu.memoryGB * 0.65,
            nvlinkGBs: 950 + 150 * Math.sin((tick + index) * 0.12),
            pcieTxGBs: 10 + 2 * Math.cos((tick + index) * 0.07),
            tempC: 64 + 4 * Math.sin((tick + index) * 0.05),
            powerW: 300 + 12 * Math.sin((tick + index) * 0.09)
          }
        })
      })

      demoEntities.nodes.forEach((node, index) => {
        update({
          topic: `node.${node.id}`,
          t: now,
          data: {
            cpuUtil: 0.45 + 0.2 * Math.abs(Math.sin((tick + index) * 0.08)),
            memoryUsedGB: node.systemMemoryGB * 0.42,
            ibUtilGbps: 140 + 60 * Math.abs(Math.sin((tick + index) * 0.1)),
            jobsRunning: 3 + ((tick + index) % 3)
          }
        })
      })

      demoEntities.links.forEach((link, index) => {
        update({
          topic: `link.${link.id}`,
          t: now,
          data: {
            linkId: link.id,
            bwGbps: link.type === 'IB' ? link.capacityGbps * (0.3 + 0.3 * Math.abs(Math.sin((tick + index) * 0.09))) : 0,
            rttUs: 6 + 1.2 * Math.sin((tick + index) * 0.11),
            errs: 0
          }
        })
      })
    }, 1000)

    return () => {
      clearInterval(interval)
    }
  }, [demoEntities, update])
}
