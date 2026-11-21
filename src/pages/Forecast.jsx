import { useMemo, useState } from 'react'
import { useGet } from '../lib/hooks'
import { fmt, pct, usd } from '../lib/format'
import { Card, ErrorBox, Loading, Note, Segmented, Select, Table } from '../components/ui'
import { FanChart, HBar } from '../components/charts'

const CLASSES = ['Newcastlemax', 'Capesize', 'Kamsarmax', 'Panamax', 'Ultramax', 'Supramax', 'Handysize']

function BdryCard({ fc }) {
  const series = useGet('/api/v1/market/series?keys=bdry&start=2025-09-01')
  const data = useMemo(() => {
    if (!series.data) return null
    const hist = series.data.bdry.points.map(([date, v]) => ({ date, hist: v }))
    const last = hist[hist.length - 1]
    const fwd = fc.bdry.path.map((p) => ({ date: p.date, mid: p.point, b1: [p.lo80, p.hi80], b2: [p.lo95, p.hi95] }))
    return [...hist.slice(0, -1), { ...last, mid: last.hist, b1: [last.hist, last.hist], b2: [last.hist, last.hist] }, ...fwd]
  }, [series.data, fc])

  const rows = fc.bdry.horizons.map((h) => ({
    ...h,
    chg: (h.point / fc.bdry.last - 1) * 100,
  }))
  return (
    <Card
      title="BDRY freight-futures ETF: 63-day forecast"
      subtitle="Stacked ensemble (random walk, Ridge, CatBoost) with GARCH-t volatility and adaptive conformal bands"
    >
      {data ? <FanChart data={data} digits={2} labels={{ hist: 'BDRY close', mid: 'Ensemble' }} splitDate={fc.as_of} /> : <Loading />}
      <div className="mt-4">
        <Table
          dense
          columns={[
            { key: 'h', label: 'Horizon', render: (r) => `${r.h}d` },
            { key: 'point', label: 'Forecast', align: 'right', render: (r) => usd(r.point, 2) },
            { key: 'chg', label: 'Change', align: 'right', render: (r) => pct(r.chg, 2) },
            { key: 'b80', label: '80% interval', align: 'right', render: (r) => `${usd(r.lo80, 2)} – ${usd(r.hi80, 2)}` },
            { key: 'b95', label: '95% interval', align: 'right', render: (r) => `${usd(r.lo95, 2)} – ${usd(r.hi95, 2)}` },
            { key: 'w', label: 'Weights RW / Ridge / CB', align: 'right',
              render: (r) => `${fmt(r.weights.random_walk, 2)} / ${fmt(r.weights.ridge, 2)} / ${fmt(r.weights.catboost, 2)}` },
            { key: 'm', label: 'Ridge · CatBoost alone', align: 'right',
              render: (r) => `${usd(r.members.ridge, 2)} · ${usd(r.members.catboost, 2)}` },
          ]}
          rows={rows}
        />
      </div>
      <p className="mt-3 text-xs text-muted">
        Stacking weights are fitted on prior out-of-sample forecasts and used only if the stack beats the random walk
        there at p &lt; 0.05. Neither learner passes, so the published forecast is the random walk. The individual learner
        forecasts are shown for transparency, not as advice.
      </p>
    </Card>
  )
}

function TceCard() {
  const [cls, setCls] = useState('Panamax')
  const q = useGet(`/api/v1/forecast/tce?vessel_class=${cls}`)
  const data = useMemo(
    () =>
      q.data?.path.map((p) => ({ date: p.date, mid: p.p50, b1: [p.p25, p.p75], b2: [p.p5, p.p95] })) ?? null,
    [q.data],
  )
  return (
    <Card
      title="Time-charter equivalent outlook"
      subtitle="$/day by vessel class from the BDI Monte Carlo paths"
      actions={<Select value={cls} onChange={setCls} options={CLASSES.map((c) => ({ value: c, label: c }))} />}
    >
      {q.error ? (
        <ErrorBox error={q.error} />
      ) : !data ? (
        <Loading />
      ) : (
        <>
          <FanChart data={data} height={260} labels={{ mid: 'Median', b1: '50% band', b2: '90% band' }} />
          <div className="mt-3 grid grid-cols-3 gap-3 text-center text-[13px]">
            {[1, 21, 63].map((h) => {
              const p = q.data.path[h - 1]
              return (
                <div key={h} className="rounded-lg border border-line py-2">
                  <div className="text-xs text-muted">{h === 1 ? 'Next day' : `${h} business days`}</div>
                  <div className="num font-semibold">{usd(p.p50)}</div>
                  <div className="num text-xs text-muted">
                    {usd(p.p5)}–{usd(p.p95)}
                  </div>
                </div>
              )
            })}
          </div>
        </>
      )}
    </Card>
  )
}

function ExplainCard() {
  const [h, setH] = useState(21)
  const q = useGet(`/api/v1/forecast/explain?h=${h}`)
  const data = useMemo(
    () => q.data?.contributions.map((c) => ({ ...c, label: `${c.feature} = ${c.value == null ? 'n/a' : fmt(c.value, 3)}` })) ?? [],
    [q.data],
  )
  return (
    <Card
      title="What the CatBoost learner sees"
      subtitle="Exact Tree SHAP contributions to its log-return forecast for the latest row"
      actions={
        <Segmented value={h} onChange={setH} options={[1, 5, 10, 21, 42, 63].map((v) => ({ value: v, label: `${v}d` }))} />
      }
    >
      {q.loading ? (
        <Loading />
      ) : q.error ? (
        <ErrorBox error={q.error} />
      ) : (
        <>
          <Note>
            CatBoost forecasts {pct(q.data.prediction_log_return * 100, 2)} (log) over {h} days, but its weight in the
            published ensemble is <b>{fmt(q.data.catboost_weight_in_ensemble * 100, 1)}%</b>. This chart explains the
            learner's view; it is not what drives the headline forecast.
          </Note>
          <div className="mt-3">
            <HBar data={data} valueKey="shap_log_return" labelKey="label" valueFmt={(v) => fmt(v * 100, 2)} />
          </div>
          <p className="text-xs text-muted">Bars in percentage points of log return. Red pushes the forecast up, green down.</p>
        </>
      )}
    </Card>
  )
}

export default function Forecast() {
  const fc = useGet('/api/v1/forecast')
  if (fc.loading) return <Loading />
  if (fc.error) return <ErrorBox error={fc.error} />
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Forecasts</h1>
        <p className="mt-1 text-sm text-muted">
          Two layers: the traded freight-futures ETF (BDRY) and the physical Baltic Dry Index, translated to $/day by class.
        </p>
      </div>
      <BdryCard fc={fc.data} />
      <div className="grid gap-6 xl:grid-cols-2">
        <TceCard />
        <ExplainCard />
      </div>
    </div>
  )
}
