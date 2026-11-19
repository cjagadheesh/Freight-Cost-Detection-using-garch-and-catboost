const formatters = new Map()
const nf = (d) => {
  if (!formatters.has(d)) {
    formatters.set(d, new Intl.NumberFormat('en-US', { maximumFractionDigits: d, minimumFractionDigits: d }))
  }
  return formatters.get(d)
}

export const fmt = (v, d = 0) => (v == null || Number.isNaN(v) ? '–' : nf(d).format(v))
export const usd = (v, d = 0) => (v == null ? '–' : `${v < 0 ? '−' : ''}$${fmt(Math.abs(v), d)}`)
export const pct = (v, d = 1) => (v == null ? '–' : `${v > 0 ? '+' : ''}${fmt(v, d)}%`)
export const compactUsd = (v) => {
  if (v == null) return '–'
  const a = Math.abs(v)
  const sign = v < 0 ? '−' : ''
  if (a >= 1e6) return `${sign}$${fmt(a / 1e6, 2)}M`
  if (a >= 1e3) return `${sign}$${fmt(a / 1e3, 1)}k`
  return `${sign}$${fmt(a, 0)}`
}
export const shortDate = (s) => {
  if (!s) return ''
  const d = new Date(`${s}T00:00:00Z`)
  return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', timeZone: 'UTC' })
}
export const monthYear = (s) => {
  const d = new Date(`${s}T00:00:00Z`)
  return d.toLocaleDateString('en-GB', { month: 'short', year: '2-digit', timeZone: 'UTC' })
}
export const titleCase = (s) => s.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
export const ordinal = (n) => {
  const v = Math.round(n)
  const s = ['th', 'st', 'nd', 'rd']
  const m = v % 100
  return `${v}${s[(m - 20) % 10] || s[m] || s[0]}`
}
