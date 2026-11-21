const BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

async function request(path, options = {}) {
  let res
  try {
    res = await fetch(`${BASE}${path}`, {
      ...options,
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    })
  } catch {
    throw new ApiError('Cannot reach the API. Is the backend running?', 0)
  }
  const text = await res.text()
  let body = null
  try {
    body = text ? JSON.parse(text) : null
  } catch {
    body = { detail: text }
  }
  if (!res.ok) {
    const detail = body?.detail
    const msg = Array.isArray(detail)
      ? detail.map((d) => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`).join('; ')
      : detail || `HTTP ${res.status}`
    throw new ApiError(msg, res.status)
  }
  return body
}

export const api = {
  get: (path) => request(path),
  post: (path, body) => request(path, { method: 'POST', body: JSON.stringify(body ?? {}) }),
}
