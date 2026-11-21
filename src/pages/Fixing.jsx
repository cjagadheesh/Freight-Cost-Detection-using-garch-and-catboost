import { useEffect, useMemo, useState } from 'react'
import { Line } from 'recharts'
import { usePost } from '../lib/hooks'
import { compactUsd, fmt, usd } from '../lib/format'
import { Badge, Button, Card, ErrorBox, Field, Loading, Note, NumberInput, Stat } from '../components/ui'
import { C, LineSet } from '../components/charts'
import RouteForm, { DEFAULT_ROUTE } from '../components/RouteForm'

function TimingCard() {
  const [route, setRoute] = useState(DEFAULT_ROUTE)
  const [windowDays, setWindow] = useState(30)
  const [risk, setRisk] = useState(0.5)
  const [run, res] = usePost('/api/v1/optimize/timing')
  const go = () => run({ ...route, window_days: windowDays, risk_aversion: risk })
  useEffect(() => {
    go()
    // initial run only
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  const d = res.data
  const chart = useMemo(
    () => d?.days.map((r) => ({ date: r.date, day: r.day, exp: r.expected_cost_usd, cvar: r.cvar_usd, band: [r.p5_usd, r.p95_usd] })) ?? [],
    [d],
  )
  return (
    <Card title="When to fix a spot charter" subtitle="Hire cost of the voyage if fixed on each day, across 5,000 simulated rate paths">
      <RouteForm value={route} onChange={setRoute}>
        <Field label="Window (days)"><NumberInput value={windowDays} min={2} max={63} onChange={setWindow} className="w-20" /></Field>
        <Field label="Risk aversion (0–1)"><NumberInput value={risk} min={0} max={1} step={0.1} onChange={setRisk} className="w-20" /></Field>
        <Button onClick={go} loading={res.loading}>Analyse</Button>
      </RouteForm>
      <div className="mt-4"><ErrorBox error={res.error} /></div>
      {!d ? (
        res.loading && <Loading />
      ) : (
        <>
          <div className="mt-2 grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="Recommendation" value={d.recommended_day === 1 ? 'Fix now' : `Wait to day ${d.recommended_day}`} sub={d.recommended_date} />
            <Stat label="Hire days" value={fmt(d.hire_days, 1)} sub={`${d.voyage.vessel_class}, ${fmt(d.voyage.distance_nm)} nm`} />
            <Stat label="Expected saving" value={compactUsd(d.expected_saving_vs_day1_usd)} sub={`MC std. error ${usd(d.mc_standard_error_usd)}`} />
            <Stat label="Tail risk change" value={compactUsd(d.cvar_change_vs_day1_usd)} sub={`CVaR ${fmt(d.cvar_level * 100)}% vs day 1`} tone={d.cvar_change_vs_day1_usd > 0 ? 'up' : 'down'} />
          </div>
          <div className="mt-4">
            <LineSet data={chart} xKey="date" band={{ key: 'band', name: '5–95% range' }}
              lines={[{ key: 'exp', name: 'Expected hire cost', color: C.brand }, { key: 'cvar', name: 'CVaR (worst 5%)', color: C.up, dash: '4 3' }]} />
          </div>
          <Note>{d.interpretation} Waiting is only recommended when it beats fixing now by more than two Monte Carlo standard errors.</Note>
        </>
      )}
    </Card>
  )
}

function SpeedCard() {
  const [route, setRoute] = useState(DEFAULT_ROUTE)
  const [berth, setBerth] = useState(null)
  const [carbon, setCarbon] = useState(0)
  const [run, res] = usePost('/api/v1/optimize/speed')
  const go = () => run({ ...route, berth_available_day: berth || null, carbon_usd_t: carbon || 0 })
  useEffect(() => {
    go()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  const d = res.data
  const chart = useMemo(() => d?.curve.map((c) => ({ ...c, kusd: c.total_usd / 1000 })) ?? [], [d])
  return (
    <Card title="Speed and virtual arrival" subtitle="Cubic fuel law vs hire; anchorage waiting burns time and port fuel">
      <RouteForm value={route} onChange={setRoute}>
        <Field label="Berth free on day" hint="blank = passage + avg wait"><NumberInput value={berth} min={1} max={60} step={0.5} onChange={setBerth} className="w-24" /></Field>
        <Field label="Carbon $/t CO₂"><NumberInput value={carbon} min={0} step={5} onChange={setCarbon} className="w-20" /></Field>
        <Button onClick={go} loading={res.loading}>Optimise</Button>
      </RouteForm>
      <div className="mt-4"><ErrorBox error={res.error} /></div>
      {d && (
        <>
          <div className="mt-2 grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="Optimal speed" value={`${fmt(d.optimal.speed_kn, 1)} kn`} sub={`design ${fmt(d.at_design_speed.speed_kn, 1)} kn`} />
            <Stat label="Saving vs design" value={compactUsd(d.saving_vs_design_usd)} sub={`${fmt(d.co2_saving_vs_design_t)} t CO₂ avoided`} tone="down" />
            <Stat label="Virtual-arrival speed" value={`${fmt(d.virtual_arrival_speed_kn, 1)} kn`} sub={`berth on day ${fmt(d.berth_available_day, 1)}`} />
            <Stat label="CII (route profile)" value={<span className="flex items-center gap-2">{d.cii.optimal.rating} <span className="text-sm text-muted">vs {d.cii.design.rating} at design</span></span>}
              sub={`attained ${fmt(d.cii.optimal.attained, 2)} / required ${fmt(d.cii.optimal.required, 2)} (${d.cii.year})`} />
          </div>
          <div className="mt-4">
            <LineSet data={chart} xKey="speed_kn" xFmt={(v) => `${v} kn`} digits={0}
              lines={[{ key: 'kusd', name: 'Passage cost ($k)', color: C.brand }]}
              extra={<Line dataKey="co2_t" name="CO₂ (t)" stroke={C.alt} dot={false} strokeDasharray="4 3" isAnimationActive={false} />} />
          </div>
          <div className="mt-2 flex flex-wrap gap-1.5 text-xs text-muted">
            CII rating by speed:
            {d.curve.filter((_, i) => i % 5 === 0).map((c) => (
              <Badge key={c.speed_kn} tone={c.cii_rating <= 'B' ? 'low' : c.cii_rating === 'C' ? 'normal' : 'high'}>
                {c.speed_kn} kn: {c.cii_rating}
              </Badge>
            ))}
          </div>
          {d.virtual_arrival_note && <p className="mt-3 text-[13px] text-slate-600">{d.virtual_arrival_note}</p>}
          <p className="mt-3 text-xs text-muted">
            CII uses IMO bulk-carrier parameters (a = 4745, c = 0.622, Z = 11% for 2026, rating boundaries 0.86/0.94/1.06/1.18)
            for a vessel shuttling this route laden and in ballast all year. Unconstrained economic speed is{' '}
            {fmt(d.economic_speed_unconstrained_kn, 1)} kn.
          </p>
        </>
      )}
    </Card>
  )
}

export default function Fixing() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Fixing and speed decisions</h1>
        <p className="mt-1 text-sm text-muted">
          Turn the rate distribution into two everyday chartering calls: when to fix, and how fast to steam.
        </p>
      </div>
      <TimingCard />
      <SpeedCard />
    </div>
  )
}
