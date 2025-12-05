const isBrowser = typeof window !== 'undefined'

function resolveVerbosePreference(): boolean {
  if (!isBrowser) return false
  try {
    const stored = window.localStorage?.getItem('explorer:verboseLogging')
    if (stored === '0') {
      return false
    }
    if (stored === '1') {
      return true
    }
  } catch (error) {
    // Ignore storage access issues and fall back to default.
  }
  return true
}

let verboseEnabled = resolveVerbosePreference()

export function setVerboseLogging(enabled: boolean) {
  if (!isBrowser) return
  verboseEnabled = enabled
  try {
    window.localStorage?.setItem('explorer:verboseLogging', enabled ? '1' : '0')
  } catch (error) {
    // Ignore storage write errors.
  }
}

export function verboseLog(message: string, details?: Record<string, unknown>) {
  if (!isBrowser || !verboseEnabled) {
    return
  }
  if (details) {
    console.info(`[explorer] ${message}`, details)
  } else {
    console.info(`[explorer] ${message}`)
  }
}
