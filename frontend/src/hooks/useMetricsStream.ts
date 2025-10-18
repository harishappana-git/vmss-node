import { useEffect } from 'react'
import { useMetricsStore } from '../state/metricsStore'

const WS_URL = import.meta.env.VITE_BACKEND_WS as string | undefined

export function useMetricsStream() {
  const update = useMetricsStore((state) => state.ingest)
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
  }, [update])
}
