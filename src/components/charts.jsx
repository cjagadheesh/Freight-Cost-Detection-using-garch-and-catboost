import {
  Area, Bar, BarChart, CartesianGrid, Cell, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts'
import { fmt, monthYear, shortDate } from '../lib/format'

export const C = {
  brand: '#0b5cad',
  band1: '#9ec5ec',
  band2: '#d4e6f7',
  hist: '#334155',
  actual: '#0f172a',
  alt: '#c2410c',
  grid: '#e2e8f0',
  muted: '#94a3b8',
  up: '#b42318',
  down: '#067647',
}

const axis = { fontSize: 11, fill: '#64748b' }

function TipBox({ active, payload, label, valueFmt, labelFmt }) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-lg border border-line bg-white px-3 py-2 text-xs shadow-lg">
      <div className="mb-1 font-medium text-ink">{labelFmt ? labelFmt(label) : label}</div>
      {payload
        .filter((p) => p.value != null && p.name && !p.name.startsWith('_'))
        .map((p) => (
          <div key={p.dataKey} className="num flex justify-between gap-4 text-slate-600">
            <span style={{ color: p.color }}>{p.name}</span>
            <span>
              {Array.isArray(p.value) ? `${valueFmt(p.value[0])} – ${valueFmt(p.value[1])}` : valueFmt(p.value)}
            </span>
          </div>
        ))}
    </div>
  )
}

/**
 * History line + forecast fan. `data` rows: {date, hist?, mid?, b1?: [lo,hi], b2?: [lo,hi]}.
 */
export function FanChart({ data, height = 300, digits = 0, labels = {}, splitDate, yDomain }) {
  const vf = (v) => fmt(v, digits)
  const long = data.length > 160
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="date" tick={axis} tickFormatter={long ? monthYear : shortDate} minTickGap={36} />
        <YAxis tick={axis} width={58} tickFormatter={vf} domain={yDomain || ['auto', 'auto']} />
        <Tooltip content={<TipBox valueFmt={vf} labelFmt={shortDate} />} />
        <Area dataKey="b2" name={labels.b2 || '95% band'} stroke="none" fill={C.band2} isAnimationActive={false} />
        <Area dataKey="b1" name={labels.b1 || '80% band'} stroke="none" fill={C.band1} isAnimationActive={false} />
        <Line dataKey="hist" name={labels.hist || 'Observed'} stroke={C.hist} dot={false} strokeWidth={1.6} isAnimationActive={false} />
        {data.some((d) => d.alt != null) && (
          <Line dataKey="alt" name={labels.alt || 'Nowcast'} stroke={C.alt} dot={false} strokeWidth={1.6} strokeDasharray="4 3" isAnimationActive={false} />
        )}
        <Line dataKey="mid" name={labels.mid || 'Forecast'} stroke={C.brand} dot={false} strokeWidth={2} isAnimationActive={false} />
        {splitDate && <ReferenceLine x={splitDate} stroke={C.muted} strokeDasharray="3 3" />}
      </ComposedChart>
    </ResponsiveContainer>
  )
}

export function HBar({ data, valueKey, labelKey, height, valueFmt = (v) => fmt(v, 3) }) {
  return (
    <ResponsiveContainer width="100%" height={height || Math.max(160, data.length * 26 + 30)}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 20, bottom: 4, left: 8 }}>
        <CartesianGrid stroke={C.grid} horizontal={false} />
        <XAxis type="number" tick={axis} tickFormatter={valueFmt} />
        <YAxis type="category" dataKey={labelKey} tick={axis} width={150} />
        <Tooltip content={<TipBox valueFmt={valueFmt} />} />
        <ReferenceLine x={0} stroke={C.muted} />
        <Bar dataKey={valueKey} name="Contribution" isAnimationActive={false}>
          {data.map((d, i) => (
            <Cell key={i} fill={d[valueKey] >= 0 ? C.up : C.down} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

export function LineSet({ data, lines, height = 260, digits = 0, xKey = 'date', xFmt = shortDate, band, yLabel, extra }) {
  const vf = (v) => fmt(v, digits)
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey={xKey} tick={axis} tickFormatter={xFmt} minTickGap={30} />
        <YAxis tick={axis} width={62} tickFormatter={vf} domain={['auto', 'auto']} label={yLabel} />
        <Tooltip content={<TipBox valueFmt={vf} labelFmt={xFmt} />} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {band && <Area dataKey={band.key} name={band.name} stroke="none" fill={C.band2} isAnimationActive={false} />}
        {lines.map((l) => (
          <Line key={l.key} dataKey={l.key} name={l.name} stroke={l.color} dot={false} strokeWidth={l.width || 1.8}
            strokeDasharray={l.dash} yAxisId={l.axis || 0} isAnimationActive={false} />
        ))}
        {extra}
      </ComposedChart>
    </ResponsiveContainer>
  )
}
