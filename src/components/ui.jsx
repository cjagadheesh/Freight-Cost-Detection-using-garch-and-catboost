import { AlertTriangle, Info, Loader2 } from 'lucide-react'

export function Card({ title, subtitle, actions, children, className = '' }) {
  return (
    <section className={`rounded-xl border border-line bg-panel shadow-[0_1px_2px_rgba(15,23,42,0.04)] ${className}`}>
      {(title || actions) && (
        <header className="flex flex-wrap items-start justify-between gap-3 border-b border-line px-5 py-3.5">
          <div>
            <h2 className="text-[15px] font-semibold text-ink">{title}</h2>
            {subtitle && <p className="mt-0.5 text-[13px] text-muted">{subtitle}</p>}
          </div>
          {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  )
}

export function Stat({ label, value, sub, tone }) {
  const toneCls = tone === 'up' ? 'text-up' : tone === 'down' ? 'text-down' : 'text-muted'
  return (
    <div className="rounded-xl border border-line bg-panel px-4 py-3.5">
      <div className="text-xs font-medium tracking-wide text-muted uppercase">{label}</div>
      <div className="num mt-1 text-2xl font-semibold text-ink">{value}</div>
      {sub && <div className={`num mt-0.5 text-[13px] ${toneCls}`}>{sub}</div>}
    </div>
  )
}

const BADGE = {
  high: 'bg-red-50 text-up ring-red-200',
  elevated: 'bg-amber-50 text-warn ring-amber-200',
  normal: 'bg-slate-50 text-slate-600 ring-slate-200',
  low: 'bg-emerald-50 text-down ring-emerald-200',
  info: 'bg-brand-soft text-brand ring-blue-200',
}

export function Badge({ tone = 'normal', children }) {
  return (
    <span className={`inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${BADGE[tone]}`}>
      {children}
    </span>
  )
}

export function Loading({ label = 'Loading…' }) {
  return (
    <div className="flex items-center gap-2 py-10 text-sm text-muted">
      <Loader2 className="h-4 w-4 animate-spin" /> {label}
    </div>
  )
}

export function ErrorBox({ error }) {
  if (!error) return null
  return (
    <div className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-up">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <span>{error.message || String(error)}</span>
    </div>
  )
}

export function Note({ children, tone = 'info' }) {
  const cls = tone === 'warn' ? 'border-amber-200 bg-amber-50 text-amber-900' : 'border-blue-100 bg-brand-soft text-slate-700'
  return (
    <div className={`flex items-start gap-2 rounded-lg border px-4 py-3 text-[13px] leading-relaxed ${cls}`}>
      {tone === 'warn' ? <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" /> : <Info className="mt-0.5 h-4 w-4 shrink-0 text-brand" />}
      <div>{children}</div>
    </div>
  )
}

export function Table({ columns, rows, dense = false, rowKey }) {
  const pad = dense ? 'px-3 py-1.5' : 'px-3 py-2'
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-[13px]">
        <thead>
          <tr className="border-b border-line text-xs text-muted uppercase">
            {columns.map((c) => (
              <th key={c.key} className={`${pad} font-medium whitespace-nowrap ${c.align === 'right' ? 'text-right' : ''}`}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={rowKey ? rowKey(r) : i} className={`border-b border-line/70 last:border-0 ${r._highlight ? 'bg-brand-soft/60' : ''}`}>
              {columns.map((c) => (
                <td key={c.key} className={`${pad} whitespace-nowrap ${c.align === 'right' ? 'num text-right' : ''}`}>
                  {c.render ? c.render(r) : r[c.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function Field({ label, children, hint }) {
  return (
    <label className="flex flex-col gap-1 text-[13px]">
      <span className="font-medium text-slate-700">{label}</span>
      {children}
      {hint && <span className="text-xs text-muted">{hint}</span>}
    </label>
  )
}

const inputCls =
  'rounded-lg border border-line bg-white px-2.5 py-1.5 text-[13px] text-ink outline-none focus:border-brand focus:ring-2 focus:ring-brand/15'

export function Select({ value, onChange, options, className = '' }) {
  return (
    <select className={`${inputCls} ${className}`} value={value} onChange={(e) => onChange(e.target.value)}>
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  )
}

export function NumberInput({ value, onChange, step = 1, min, max, className = '' }) {
  return (
    <input
      type="number"
      className={`num ${inputCls} ${className}`}
      value={value ?? ''}
      step={step}
      min={min}
      max={max}
      onChange={(e) => onChange(e.target.value === '' ? null : Number(e.target.value))}
    />
  )
}

export function TextInput({ value, onChange, className = '' }) {
  return <input className={`${inputCls} ${className}`} value={value} onChange={(e) => onChange(e.target.value)} />
}

export function Button({ children, onClick, loading, variant = 'primary', type = 'button', disabled }) {
  const cls =
    variant === 'primary'
      ? 'bg-brand text-white hover:bg-[#094c8f] disabled:bg-slate-300'
      : 'border border-line bg-white text-ink hover:bg-slate-50 disabled:text-slate-400'
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={loading || disabled}
      className={`inline-flex items-center gap-2 rounded-lg px-3.5 py-1.5 text-[13px] font-medium transition ${cls}`}
    >
      {loading && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
      {children}
    </button>
  )
}

export function Segmented({ value, onChange, options }) {
  return (
    <div className="inline-flex rounded-lg border border-line bg-slate-50 p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          onClick={() => onChange(o.value)}
          className={`rounded-md px-2.5 py-1 text-xs font-medium transition ${
            value === o.value ? 'bg-white text-ink shadow-sm' : 'text-muted hover:text-ink'
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}
