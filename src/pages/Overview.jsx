import { useMemo } from 'react'
import { useGet } from '../lib/hooks'
import { fmt, ordinal, pct, usd } from '../lib/format'
import { Badge, Card, ErrorBox, Loading, Note, Stat } from '../components/ui'
import { FanChart } from '../components/charts'

function bdiFanData(series, fc, historyDays = 130) {
  const hist = (series?.bdi?.points || []).slice(-historyDays).map(([date, v]) => ({ date, hist: v }))
  const nc = fc.bdi.nowcast.path.map((p) => ({ date: p.date, alt: p.bdi, b1: [p.p10, p.p90] }))
  const fwd = fc.bdi.path.map((p) => ({ date: p.date, mid: p.p50, b1: [p.p10, p.p90], b2: [p.p5, p.p95] }))
  if (hist.length && nc.length) nc.unshift({ date: hist[hist.length - 1].date, alt: hist[hist.length - 1].hist })
  if (nc.length && fwd.length) {
    const last = nc[nc.length - 1]
    fwd.unshift({ date: last.date, mid: last.alt, b1: last.b1, b2: last.b1 })
  }
  // merge rows with the same date
  const map = new Map()
  for (const r of [...hist, ...nc, ...fwd]) map.set(r.date, { ...(map.get(r.date) || {}), ...r })
  return [...map.values()].sort((a, b) => a.date.localeCompare(b.date))
}

export default function Overview() {
  const fc = useGet('/api/v1/forecast')
  const snap = useGet('/api/v1/market/snapshot')
  const risk = useGet('/api/v1/risk')
  const series = useGet('/api/v1/market/series?keys=bdi&start=2025-06-01')

  const chart = useMemo(() => (fc.data?.bdi && series.data ? bdiFanData(series.data, fc.data) : null), [fc.data, series.data])

  if (fc.loading || snap.loading) return <Loading />
  if (fc.error || snap.error) return <ErrorBox error={fc.error || snap.error} />

  const f = fc.data
  const nc = f.bdi?.nowcast
  const bdry = snap.data.series.find((s) => s.key === 'bdry')
  const tce = f.bdi?.tce_today || {}
  const h21 = f.bdi?.path?.[20]

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Freight market today</h1>
        <p className="mt-1 text-sm text-muted">
          Spot dry-bulk rates for an east-coast coal and ore import programme, the 3-month outlook, and current risk
          signals, all from live market data as of {f.as_of}.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <Stat
          label="BDI (nowcast)"
          value={nc ? fmt(nc.level) : '–'}
          sub={nc ? `80% ${fmt(nc.p10)}–${fmt(nc.p90)}` : 'no BDI history'}
        />
        <Stat label="BDRY (FFA ETF)" value={usd(bdry.last, 2)} sub={`${pct(bdry.chg_1d_pct)} 1d · ${pct(bdry.chg_21d_pct)} 1m`}
          tone={bdry.chg_21d_pct > 0 ? 'up' : 'down'} />
        <Stat label="Capesize TCE" value={usd(tce.Capesize)} sub="$/day, from BDI" />
        <Stat label="Panamax TCE" value={usd(tce.Panamax)} sub="$/day, from BDI" />
        <Stat label="VLSFO (proxy)" value={usd(snap.data.bunker_vlsfo_usd_t)} sub={`Brent ${usd(snap.data.brent, 2)}/bbl`} />
        <Stat label="USD/INR" value={fmt(snap.data.usdinr, 2)} sub="rupee per dollar" />
      </div>

      {nc && nc.gap_bdays > 5 && (
        <Note tone="warn">
          The last Baltic Dry Index print in the Koyfin export is from <b>{nc.last_print_date}</b> ({fmt(nc.last_print)}).
          The {nc.gap_bdays} business days since then are <b>nowcast</b> from actual BDRY freight-futures closes with the
          error-correction model; the band shows how much that inference is worth. Add a newer Koyfin export to tighten it.
        </Note>
      )}

      <div className="grid gap-6 xl:grid-cols-3">
        <Card
          className="xl:col-span-2"
          title="Baltic Dry Index: observed, nowcast and 63-day outlook"
          subtitle="Monte Carlo of 5,000 GARCH-FHS freight-futures paths through the BDI error-correction model"
        >
          {chart ? (
            <FanChart data={chart} labels={{ hist: 'BDI print', alt: 'Nowcast', mid: 'Median outlook', b1: '80% band', b2: '90% band' }}
              splitDate={nc?.as_of} />
          ) : (
            <Loading />
          )}
          {h21 && (
            <p className="mt-3 text-[13px] text-muted">
              One month ahead ({h21.date}): median <b className="text-ink">{fmt(h21.p50)}</b>, 80% range {fmt(h21.p10)}–
              {fmt(h21.p90)}. Beyond a week the central path is deliberately close to a random walk; see Model validation
              for why.
            </p>
          )}
        </Card>

        <Card title="Risk signals" subtitle="Percentiles against each series' own history since 2018">
          {risk.loading ? (
            <Loading />
          ) : risk.error ? (
            <ErrorBox error={risk.error} />
          ) : (
            <ul className="divide-y divide-line">
              {risk.data.signals.map((s) => (
                <li key={s.key} className="py-2.5 first:pt-0 last:pb-0">
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-[13px] font-medium">{s.name}</span>
                    <Badge tone={s.level}>{s.level}</Badge>
                  </div>
                  <div className="num mt-0.5 text-[13px] text-slate-600">
                    {fmt(s.value, s.unit === 'trading days' ? 0 : 2)} {s.unit}
                    {s.percentile != null && <span className="text-muted"> · {ordinal(s.percentile)} percentile</span>}
                  </div>
                  <div className="mt-0.5 text-xs text-muted">{s.detail}</div>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>

      <Card title="Time-charter rates by vessel class" subtitle="BDI × 10 = Baltic-weighted TCE (2018 formula) × class ratio">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-7">
          {Object.entries(tce).map(([k, v]) => (
            <div key={k} className="rounded-lg border border-line px-3 py-2.5">
              <div className="text-xs text-muted">{k}</div>
              <div className="num text-lg font-semibold">{usd(v)}</div>
              <div className="num text-xs text-muted">
                {usd(f.bdi.tce_p10_today[k])}–{usd(f.bdi.tce_p90_today[k])}
              </div>
            </div>
          ))}
        </div>
        <p className="mt-3 text-xs text-muted">
          Class ratios are long-run averages relative to the weighted index (assumption, configurable). The range is the
          80% nowcast band.
        </p>
      </Card>
    </div>
  )
}
