import { useEffect, useState } from 'react'
import { usePost } from '../lib/hooks'
import { compactUsd, fmt, pct, usd } from '../lib/format'
import { Badge, Button, Card, ErrorBox, Field, Loading, NumberInput, Table } from '../components/ui'
import RouteForm, { DEFAULT_ROUTE } from '../components/RouteForm'

function LandedCard() {
  const [route, setRoute] = useState(DEFAULT_ROUTE)
  const [fob, setFob] = useState(null)
  const [cv, setCv] = useState(null)
  const [day, setDay] = useState(1)
  const [run, res] = usePost('/api/v1/landed-cost')
  const go = () => run({ ...route, fob_usd_t: fob || null, cv_kcal_kg: cv || null, fixing_day: day || 1 })
  useEffect(() => {
    go()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  const d = res.data
  const rows = d
    ? [
        ['fob', 'FOB'], ['sea_freight', 'Sea freight'], ['insurance', 'Insurance'], ['port_and_lightering', 'Ports & lightering'],
        ['landed_usd_t', 'Landed $/t'], ['landed_inr_t', 'Landed ₹/t'], ['usd_per_gj', '$/GJ'], ['freight_share_pct', 'Freight share'],
        ['tce_usd_day', 'TCE used ($/day)'],
      ].map(([k, label]) => ({
        label, k,
        p10: d.bands.p10[k], p50: d.bands.p50[k], p90: d.bands.p90[k],
        _highlight: k === 'landed_usd_t',
      }))
    : []
  const f = (k, v) => (k === 'landed_inr_t' ? `₹${fmt(v)}` : k === 'usd_per_gj' ? usd(v, 3) : k === 'freight_share_pct' ? `${fmt(v, 1)}%` : k === 'tce_usd_day' ? usd(v) : usd(v, 2))
  return (
    <Card title="Landed cost" subtitle="FOB + freight + insurance + port and lightering, with freight at the forecast TCE distribution">
      <RouteForm value={route} onChange={setRoute}>
        <Field label="FOB $/t" hint="blank = illustrative default"><NumberInput value={fob} step={1} onChange={setFob} className="w-24" /></Field>
        <Field label="CV kcal/kg"><NumberInput value={cv} step={100} onChange={setCv} className="w-24" /></Field>
        <Field label="Fix on day"><NumberInput value={day} min={1} max={63} onChange={setDay} className="w-20" /></Field>
        <Button onClick={go} loading={res.loading}>Calculate</Button>
      </RouteForm>
      <div className="mt-4"><ErrorBox error={res.error} /></div>
      {!d ? (
        res.loading && <Loading />
      ) : (
        <>
          <div className="mt-2">
            <Table dense columns={[
              { key: 'label', label: '' },
              { key: 'p10', label: 'Soft market (P10)', align: 'right', render: (r) => f(r.k, r.p10) },
              { key: 'p50', label: 'Median', align: 'right', render: (r) => <b>{f(r.k, r.p50)}</b> },
              { key: 'p90', label: 'Firm market (P90)', align: 'right', render: (r) => f(r.k, r.p90) },
            ]} rows={rows} />
          </div>
          <p className="mt-3 text-xs text-muted">
            {fmt(d.cv_kcal_kg)} kcal/kg = {fmt(d.gj_per_t, 2)} GJ/t · USD/INR {fmt(d.usdinr, 2)} (live) · bunkers {usd(d.bunker_usd_t)}/t
            (Brent-linked) · FOB: {d.fob_source}.
          </p>
        </>
      )}
    </Card>
  )
}

function StressCard() {
  const [route, setRoute] = useState({ ...DEFAULT_ROUTE, load_port: 'newcastle', vessel_class: 'Panamax' })
  const [win, setWin] = useState(42)
  const [run, res] = usePost('/api/v1/stress')
  const go = () => run({ ...route, window_days: win })
  useEffect(() => {
    go()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  const d = res.data
  return (
    <Card title="Historical stress replay" subtitle="The largest freight rallies and slumps since 2018, replayed on today's voyage">
      <RouteForm value={route} onChange={setRoute}>
        <Field label="Window (bdays)"><NumberInput value={win} min={5} max={126} onChange={setWin} className="w-20" /></Field>
        <Button onClick={go} loading={res.loading}>Replay</Button>
      </RouteForm>
      <div className="mt-4"><ErrorBox error={res.error} /></div>
      {d && (
        <>
          <p className="mt-2 text-[13px] text-muted">
            Base case today: {usd(d.base.usd_per_t, 2)}/t ({compactUsd(d.base.total_usd)} voyage). Each scenario applies the
            actual BDRY move over the window (scaled by the ECM's long-run BDI elasticity {fmt(d.elasticity_beta, 2)}) and
            Brent's move over the same dates.
          </p>
          <div className="mt-3">
            <Table dense columns={[
              { key: 'kind', label: 'Episode', render: (r) => <Badge tone={r.kind === 'rally' ? 'high' : 'low'}>{r.kind}</Badge> },
              { key: 'w', label: 'Window', render: (r) => `${r.start} → ${r.end}` },
              { key: 'bdry', label: 'BDRY', align: 'right', render: (r) => pct(r.bdry_change_pct) },
              { key: 'tce', label: 'Spot TCE', align: 'right', render: (r) => pct(r.spot_tce_change_pct) },
              { key: 'brent', label: 'Brent', align: 'right', render: (r) => (r.brent_change_usd == null ? '–' : `${r.brent_change_usd > 0 ? '+' : ''}${usd(r.brent_change_usd, 1)}/bbl`) },
              { key: 'pt', label: 'Freight $/t', align: 'right', render: (r) => usd(r.usd_per_t, 2) },
              { key: 'chg', label: 'Δ $/t', align: 'right', render: (r) => <span className={r.change_usd_per_t > 0 ? 'text-up' : 'text-down'}>{r.change_usd_per_t > 0 ? '+' : '−'}{fmt(Math.abs(r.change_usd_per_t), 2)}</span> },
              { key: 'imp', label: 'Cargo impact', align: 'right', render: (r) => compactUsd(r.programme_impact_usd) },
            ]} rows={d.scenarios} />
          </div>
        </>
      )}
    </Card>
  )
}

export default function Costs() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Landed cost and stress</h1>
        <p className="mt-1 text-sm text-muted">
          What a tonne (and a gigajoule) of imported coal costs at the plant gate, and how badly freight shocks from the
          last eight years would hit it.
        </p>
      </div>
      <LandedCard />
      <StressCard />
    </div>
  )
}
