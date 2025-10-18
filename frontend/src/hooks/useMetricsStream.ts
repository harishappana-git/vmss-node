import { useEffect } from 'react'
import { useMetricsStore } from '../state/metricsStore'

export function useMetricsStream() {
  const update = useMetricsStore((state) => state.ingest)
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${protocol}://${window.location.host}/stream`)

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data)
        update(payload)
      } catch (err) {
        console.error('failed to parse frame', err)
      }
    }

    return () => {
      ws.close()
    }
  }, [update])
}
