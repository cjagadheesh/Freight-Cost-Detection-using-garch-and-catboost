import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api'

const cache = new Map()

/** GET with a small in-memory cache so switching views does not refetch. */
export function useGet(path) {
  const [state, setState] = useState(() => ({ data: cache.get(path) ?? null, error: null, loading: !cache.has(path) }))
  useEffect(() => {
    if (!path) return undefined
    let alive = true
    if (cache.has(path)) {
      setState({ data: cache.get(path), error: null, loading: false })
      return undefined
    }
    setState((s) => ({ ...s, loading: true, error: null }))
    api
      .get(path)
      .then((data) => {
        cache.set(path, data)
        if (alive) setState({ data, error: null, loading: false })
      })
      .catch((error) => alive && setState({ data: null, error, loading: false }))
    return () => {
      alive = false
    }
  }, [path])
  return state
}

/** POST runner: returns [run, {data, error, loading}]. Stale responses are ignored. */
export function usePost(path) {
  const [state, setState] = useState({ data: null, error: null, loading: false })
  const seq = useRef(0)
  const run = useCallback(
    async (body) => {
      const id = ++seq.current
      setState((s) => ({ ...s, loading: true, error: null }))
      try {
        const data = await api.post(path, body)
        if (id === seq.current) setState({ data, error: null, loading: false })
        return data
      } catch (error) {
        if (id === seq.current) setState((s) => ({ ...s, error, loading: false }))
        return null
      }
    },
    [path],
  )
  return [run, state]
}
