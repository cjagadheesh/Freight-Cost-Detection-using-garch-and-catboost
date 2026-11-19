import { useMemo, useState } from 'react'
import { useGet } from '../lib/hooks'
import { fmt, monthYear } from '../lib/format'
import { Badge, Card, ErrorBox, Loading, Select, Table } from '../components/ui'
import { C, LineSet } from '../components/charts'

function SeriesCard({ catalog }) {
  const [a, setA] = useState('bdry')
  const [b, setB] = useState('gnk')
  const q = useGet(`/api/v1/market/series?keys=${a},${b}&start=2018-01-01`)
  const data = useMemo(() => {
    if (!q.data) return []
    // Rebase both series to 100 at their first common date.
    const sa = new Map(q.data[a].points)
    const sb = new Map(q.data[b].points)
    const dates = [...sa.keys()].filter((d) => sb.has(d))
    if (!dates.length) return []
    const a0 = sa.get(dates[0])
    const b0 = sb.get(dates[0])
    return dates.filter((_, i) => i % 3 === 0).map((d) => ({ date: d, a: (sa.get(d) / a0) * 100, b: (sb.get(d) / b0) * 100 }))
  }, [q.data, a, b])
  const opts = catalog.map((c) => ({ value: c.key, label: c.name }))
  return (
    <Card title="Compare series" subtitle="Rebased to 100 at the first common date"
      actions={<><Select value={a} onChange={setA} options={opts} /><Select value={b} onChange={setB} options={opts} /></>}>
      {q.loading ? <Loading /> : q.error ? <ErrorBox error={q.error} /> : (
        <LineSet data={data} xFmt={monthYear} digits={0}
          lines={[{ key: 'a', name: q.data[a].name, color: C.brand }, { key: 'b', name: q.data[b].name, color: C.alt }]} />
      )}
    </Card>
  )
}

export default function DataPage() {
  const quality = useGet('/api/v1/market/quality')
  const catalog = useGet('/api/v1/market/catalog')
  const ref = useGet('/api/v1/reference')
  if (quality.loading || catalog.loading || ref.loading) return <Loading />
  if (quality.error) return <ErrorBox error={quality.error} />
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Data</h1>
        <p className="mt-1 text-sm text-muted">
          Every series the models use, where it comes from, and how fresh and complete it is. Nothing here is simulated.
        </p>
      </div>
      <Card title="Market data quality" subtitle={`Panel as of ${quality.data.as_of}`}>
        <Table dense rowKey={(r) => r.key} columns={[
          { key: 'name', label: 'Series', render: (r) => <span><b>{r.name}</b> <span className="text-muted">{r.symbol}</span></span> },
          { key: 'source', label: 'Source' },
          { key: 'role', label: 'Role', render: (r) => <Badge tone={r.role === 'target' ? 'info' : 'normal'}>{r.role}</Badge> },
          { key: 'range', label: 'Range', render: (r) => `${r.first_date} → ${r.last_date}` },
          { key: 'observations', label: 'Obs', align: 'right' },
          { key: 'coverage_pct', label: 'Coverage', align: 'right', render: (r) => `${fmt(r.coverage_pct, 1)}%` },
          { key: 'stale', label: 'Stale (bd)', align: 'right', render: (r) => <span className={r.business_days_stale > 5 ? 'text-warn font-medium' : ''}>{r.business_days_stale}</span> },
          { key: 'last_value', label: 'Last', align: 'right', render: (r) => fmt(r.last_value, 2) },
          { key: 'max', label: 'Max daily move', align: 'right', render: (r) => `${fmt(r.max_abs_daily_move_pct, 1)}%` },
        ]} rows={quality.data.series} />
        <p className="mt-3 text-xs text-muted">
          Coverage is observations over business days in range (exchange holidays lower it). Drivers are aligned to the
          BDRY calendar with a causal forward-fill of at most 5 days and lagged one day in features.
        </p>
      </Card>

      {catalog.data && <SeriesCard catalog={catalog.data.filter((c) => c.key !== 'bdi')} />}

      {ref.data && (
        <div className="grid gap-6 xl:grid-cols-2">
          <Card title="Vessel classes" subtitle="Typical modern specifications (planning assumptions)">
            <Table dense rowKey={(r) => r.name} columns={[
              { key: 'name', label: 'Class' },
              { key: 'dwt', label: 'DWT', align: 'right', render: (r) => fmt(r.dwt) },
              { key: 'laden_draft_m', label: 'Draft m', align: 'right', render: (r) => fmt(r.laden_draft_m, 1) },
              { key: 'loa_m', label: 'LOA m', align: 'right' },
              { key: 'design_speed_kn', label: 'Speed kn', align: 'right' },
              { key: 'fuel_at_design_tpd', label: 'Fuel t/d', align: 'right' },
            ]} rows={ref.data.vessel_classes} />
          </Card>
          <Card title="Discharge ports" subtitle="Indian east coast (planning assumptions)">
            <Table dense rowKey={(r) => r.key} columns={[
              { key: 'name', label: 'Port' },
              { key: 'max_draft_m', label: 'Draft m', align: 'right' },
              { key: 'max_loa_m', label: 'LOA m', align: 'right' },
              { key: 'discharge_rate_tpd', label: 't/day', align: 'right', render: (r) => fmt(r.discharge_rate_tpd) },
              { key: 'avg_wait_days', label: 'Wait d', align: 'right' },
              { key: 'berths', label: 'Berths', align: 'right' },
              { key: 'lightering_via', label: 'Lightering', render: (r) => r.lightering_via || '–' },
            ]} rows={ref.data.discharge_ports} />
          </Card>
        </div>
      )}
    </div>
  )
}
