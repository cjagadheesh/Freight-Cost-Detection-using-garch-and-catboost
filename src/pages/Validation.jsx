import { useMemo, useState } from 'react'
import { useGet } from '../lib/hooks'
import { fmt } from '../lib/format'
import { Badge, Card, ErrorBox, Loading, Note, Segmented, Table } from '../components/ui'
import { C, FanChart } from '../components/charts'

const H = [1, 5, 10, 21, 42, 63]

function skillTone(v) {
  if (v == null) return 'normal'
  return v > 2 ? 'low' : v < -2 ? 'high' : 'normal'
}

export default function Validation() {
  const [h, setH] = useState(21)
  const bt = useGet(`/api/v1/backtest?h=${h}`)
  const models = useGet('/api/v1/models')

  const chart = useMemo(
    () => bt.data?.series.map((r) => ({ date: r.target_date, hist: r.actual, mid: r.forecast, b1: [r.lo80, r.hi80] })) ?? null,
    [bt.data],
  )

  if (bt.loading && !bt.data) return <Loading />
  if (bt.error) return <ErrorBox error={bt.error} />
  const d = bt.data
  const ens = d.metrics.filter((m) => m.model === 'ensemble')
  const byH = d.metrics.filter((m) => m.h === h)

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Model validation</h1>
        <p className="mt-1 text-sm text-muted">
          Every number here is out-of-sample: expanding-window walk-forward, refit each quarter, training labels purged so
          none overlap the test period, stacking weights and intervals fitted only on earlier forecasts.
        </p>
      </div>

      <Note>
        <b>Headline result.</b> On the BDRY freight-futures ETF, neither Ridge nor CatBoost beats a random walk at any horizon
        from 1 to 63 days (Diebold–Mariano p &gt; 0.85), so the ensemble puts its weight on the random walk. That is what
        efficient futures pricing predicts. The value is in the <b>uncertainty</b>: the 80% interval covers 80% of
        outcomes on average at every horizon, though at long horizons the hit rate drifts over time (see the halves). On the physical <b>BDI</b>, which lags its futures, the error-correction model cuts
        one-day-ahead squared error by about a quarter.
      </Note>

      <Card title="Ensemble accuracy and calibration by horizon" subtitle={`${ens[0]?.n ?? ''} out-of-sample forecasts per horizon, origins ${d.series[0]?.date ?? ''} to ${d.series[d.series.length - 1]?.date ?? ''}`}>
        <Table
          columns={[
            { key: 'h', label: 'Horizon', render: (r) => `${r.h}d` },
            { key: 'rmse_pct', label: 'RMSE', align: 'right', render: (r) => `${fmt(r.rmse_pct, 2)}%` },
            { key: 'mae_pct', label: 'MAE', align: 'right', render: (r) => `${fmt(r.mae_pct, 2)}%` },
            { key: 'skill', label: 'Skill vs RW', align: 'right', render: (r) => <Badge tone={skillTone(r.skill_vs_rw_pct)}>{fmt(r.skill_vs_rw_pct, 2)}%</Badge> },
            { key: 'dm', label: 'DM p-value', align: 'right', render: (r) => fmt(r.dm_pvalue, 3) },
            { key: 'c50', label: '50% cover', align: 'right', render: (r) => `${fmt(r.coverage50_pct, 1)}%` },
            { key: 'c80', label: '80% cover', align: 'right', render: (r) => `${fmt(r.coverage80_pct, 1)}%` },
            { key: 'c80h', label: '80% 1st / 2nd half', align: 'right',
              render: (r) => `${fmt(r.coverage80_first_half_pct, 1)} / ${fmt(r.coverage80_second_half_pct, 1)}` },
            { key: 'c95', label: '95% cover', align: 'right', render: (r) => `${fmt(r.coverage95_pct, 1)}%` },
            { key: 'w80', label: '80% width', align: 'right', render: (r) => `${fmt(r.width80_pct, 1)}%` },
          ]}
          rows={ens}
        />
        <p className="mt-3 text-xs text-muted">
          Skill = 1 − MSE / MSE(random walk). Coverage uses adaptive conformal inference: the miscoverage target is nudged
          after each realised outcome, with feedback delayed by h days so nothing is seen early. On average it hits the
          target, but look at the halves: at 42–63 days the bands under-covered through the 2021–22 boom and bust and
          over-covered afterwards. Average coverage hides that drift, and long-horizon bands should be read with it in mind.
        </p>
      </Card>

      <Card
        title="Forecast vs outcome"
        subtitle="Each point is a forecast made h business days before the plotted date"
        actions={<Segmented value={h} onChange={setH} options={H.map((v) => ({ value: v, label: `${v}d` }))} />}
      >
        {chart && (
          <FanChart data={chart} digits={2} height={320}
            labels={{ hist: 'Realised BDRY', mid: 'Ensemble forecast', b1: '80% interval' }} />
        )}
        <div className="mt-4">
          <Table
            dense
            columns={[
              { key: 'model', label: `Model (h = ${h})` },
              { key: 'rmse_pct', label: 'RMSE', align: 'right', render: (r) => `${fmt(r.rmse_pct, 2)}%` },
              { key: 'skill', label: 'Skill vs RW', align: 'right', render: (r) => `${fmt(r.skill_vs_rw_pct, 2)}%` },
              { key: 'dir', label: 'Direction hit', align: 'right', render: (r) => (r.directional_accuracy_pct == null ? '–' : `${fmt(r.directional_accuracy_pct, 1)}%`) },
              { key: 'dm', label: 'DM p', align: 'right', render: (r) => (r.dm_pvalue == null ? '–' : fmt(r.dm_pvalue, 3)) },
            ]}
            rows={byH}
          />
        </div>
      </Card>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card title="BDI error-correction model (walk-forward)" subtitle="Expanding window from 120 prints; refit every 10 days">
          <Table
            dense
            columns={[
              { key: 'h', label: 'Horizon', render: (r) => `${r.h}d` },
              { key: 'n', label: 'n', align: 'right' },
              { key: 'rmse', label: 'RMSE ECM / RW', align: 'right', render: (r) => `${fmt(r.rmse_ecm_pct, 2)}% / ${fmt(r.rmse_rw_pct, 2)}%` },
              { key: 'skill', label: 'Skill', align: 'right', render: (r) => <Badge tone={skillTone(r.skill_vs_rw_pct)}>{fmt(r.skill_vs_rw_pct, 1)}%</Badge> },
              { key: 'dir', label: 'Direction', align: 'right', render: (r) => `${fmt(r.directional_accuracy_pct, 1)}%` },
              { key: 'p', label: 'DM p', align: 'right', render: (r) => fmt(r.dm_pvalue, 3) },
            ]}
            rows={d.ecm_metrics}
          />
          <p className="mt-3 text-xs text-muted">
            Significant at one day (p ≈ 0.03). Beyond a week the evidence runs out: one year of BDI holds about six
            independent 21-day windows (block-bootstrap 90% range for skill: +12% to +40% at 1 day, −8% to +15% at 5 days).
            The production forecast uses the ECM signal fully up to 5 days and tapers it to a random walk by 21 days. That
            rule was chosen after seeing these results, so it is informed by this sample rather than independent of it.
          </p>
        </Card>

        <Card title="Fitted models (full sample)">
          {models.loading ? (
            <Loading />
          ) : models.error ? (
            <ErrorBox error={models.error} />
          ) : (
            <div className="grid gap-4 text-[13px] sm:grid-cols-2">
              <div>
                <div className="mb-1.5 font-medium">GARCH(1,1), Student-t on BDRY</div>
                <dl className="num grid grid-cols-2 gap-y-1 text-slate-600">
                  <dt>α / β</dt><dd className="text-right">{fmt(models.data.garch.alpha, 3)} / {fmt(models.data.garch.beta, 3)}</dd>
                  <dt>Persistence</dt><dd className="text-right">{fmt(models.data.garch.persistence, 4)}</dd>
                  <dt>Half-life</dt><dd className="text-right">{fmt(models.data.garch.half_life_days, 1)} days</dd>
                  <dt>Tail ν</dt><dd className="text-right">{fmt(models.data.garch.nu, 2)}</dd>
                  <dt>Vol now / long-run</dt>
                  <dd className="text-right">{fmt(models.data.garch.current_daily_vol_pct, 2)}% / {fmt(models.data.garch.long_run_daily_vol_pct, 2)}%</dd>
                </dl>
              </div>
              {models.data.ecm && (
                <div>
                  <div className="mb-1.5 font-medium">BDI error-correction model</div>
                  <dl className="num grid grid-cols-2 gap-y-1 text-slate-600">
                    <dt>Long-run elasticity β</dt><dd className="text-right">{fmt(models.data.ecm.cointegration.beta, 3)}</dd>
                    <dt>Adjustment γ</dt><dd className="text-right">{fmt(models.data.ecm.adjustment_speed_gamma, 4)}</dd>
                    <dt>Basis half-life</dt><dd className="text-right">{fmt(models.data.ecm.basis_half_life_days, 1)} days</dd>
                    <dt>BDRY same-day pass-through</dt><dd className="text-right">{fmt(models.data.ecm.coefficients.df_0, 3)}</dd>
                    <dt>Residual sd</dt><dd className="text-right">{fmt(models.data.ecm.residual_daily_sd_pct, 2)}%/day</dd>
                    <dt>Sample</dt><dd className="text-right">{models.data.ecm.n_obs} prints</dd>
                  </dl>
                </div>
              )}
            </div>
          )}
        </Card>
      </div>

      <Card title="Fold-by-fold (21-day horizon)" subtitle="Quarterly refits; RMSE of the ensemble vs the random walk in each test window">
        <Table
          dense
          columns={[
            { key: 'fold', label: 'Fold' },
            { key: 'test_start', label: 'Test window', render: (r) => `${r.test_start} → ${r.test_end}` },
            { key: 'train_rows', label: 'Train rows', align: 'right' },
            { key: 'rw', label: 'RMSE RW', align: 'right', render: (r) => `${fmt(r.rmse_rw_pct, 2)}%` },
            { key: 'en', label: 'RMSE ensemble', align: 'right',
              render: (r) => <span style={{ color: r.rmse_ensemble_pct > r.rmse_rw_pct + 0.01 ? C.up : undefined }}>{fmt(r.rmse_ensemble_pct, 2)}%</span> },
          ]}
          rows={d.folds}
        />
      </Card>
    </div>
  )
}
