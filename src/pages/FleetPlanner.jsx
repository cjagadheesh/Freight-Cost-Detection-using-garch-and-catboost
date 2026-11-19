import { useEffect, useMemo, useState } from 'react'
import { Plus, Trash2 } from 'lucide-react'
import { useGet, usePost } from '../lib/hooks'
import { compactUsd, fmt, usd } from '../lib/format'
import { Badge, Button, Card, ErrorBox, Field, Loading, Note, NumberInput, Segmented, Select, Stat, Table, TextInput } from '../components/ui'
import PortMap from '../components/PortMap'

const DEFAULT_CARGOES = [
  { id: 'C1', load_port: 'hay_point', discharge_port: 'dhamra', quantity_t: 160000, laycan_start: 3, laycan_end: 12 },
  { id: 'C2', load_port: 'newcastle', discharge_port: 'paradip', quantity_t: 75000, laycan_start: 1, laycan_end: 8 },
  { id: 'C3', load_port: 'samarinda', discharge_port: 'haldia', quantity_t: 55000, laycan_start: 2, laycan_end: 10 },
  { id: 'C4', load_port: 'richards_bay', discharge_port: 'gangavaram', quantity_t: 150000, laycan_start: 8, laycan_end: 18 },
  { id: 'C5', load_port: 'taboneo', discharge_port: 'gopalpur', quantity_t: 52000, laycan_start: 5, laycan_end: 14 },
  { id: 'C6', load_port: 'samarinda', discharge_port: 'vizag', quantity_t: 70000, laycan_start: 10, laycan_end: 20 },
]

function Timeline({ assignments, ports }) {
  const max = Math.max(30, ...assignments.map((a) => Math.ceil(a.berth_to_day + 1)))
  const byPort = ports.filter((p) => assignments.some((a) => a.voyage.discharge_port === p.key))
  return (
    <div className="space-y-2">
      <div className="num ml-28 flex justify-between text-[10px] text-muted">
        {Array.from({ length: 7 }, (_, i) => Math.round((i * max) / 6)).map((d) => (
          <span key={d}>day {d}</span>
        ))}
      </div>
      {byPort.map((p) => (
        <div key={p.key} className="flex items-center gap-2">
          <div className="w-26 shrink-0 text-right text-xs">
            <div className="font-medium">{p.name}</div>
            <div className="text-muted">{p.berths} berth{p.berths > 1 ? 's' : ''}</div>
          </div>
          <div className="relative h-8 flex-1 rounded bg-slate-50 ring-1 ring-line">
            {assignments
              .filter((a) => a.voyage.discharge_port === p.key)
              .map((a) => (
                <div
                  key={a.cargo_id}
                  title={`${a.cargo_id} ${a.vessel_name}: berth day ${a.berth_from_day}–${a.berth_to_day}`}
                  className={`absolute top-1 bottom-1 flex items-center overflow-hidden rounded px-1 text-[10px] font-medium text-white ${a.kind === 'spot' ? 'bg-[#c2410c]' : 'bg-brand'}`}
                  style={{ left: `${(a.berth_from_day / max) * 100}%`, width: `${Math.max(((a.berth_to_day - a.berth_from_day) / max) * 100, 1.5)}%` }}
                >
                  {a.cargo_id}
                </div>
              ))}
          </div>
        </div>
      ))}
      <div className="flex gap-4 pl-28 text-xs text-muted">
        <span className="flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-sm bg-brand" /> Fleet vessel</span>
        <span className="flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-sm bg-[#c2410c]" /> Spot charter</span>
      </div>
    </div>
  )
}

export default function FleetPlanner() {
  const ref = useGet('/api/v1/reference')
  const [run, res] = usePost('/api/v1/optimize/allocation')
  const [cargoes, setCargoes] = useState(DEFAULT_CARGOES)
  const [fleet, setFleet] = useState(null)
  const [quantile, setQuantile] = useState('mean')
  const [slotDays, setSlotDays] = useState(2)
  const [carbon, setCarbon] = useState(0)

  useEffect(() => {
    if (ref.data && fleet === null) setFleet(ref.data.sample_fleet)
  }, [ref.data, fleet])

  const solve = () =>
    run({
      cargoes,
      fleet,
      slot_days: slotDays,
      rate_quantile: quantile === 'mean' ? null : Number(quantile),
      carbon_usd_t: carbon || 0,
    })

  useEffect(() => {
    if (fleet && !res.data && !res.loading) solve()
    // run once when the fleet is first available
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fleet])

  const opts = useMemo(() => {
    if (!ref.data) return null
    return {
      lp: ref.data.load_ports.map((p) => ({ value: p.key, label: p.name })),
      dp: ref.data.discharge_ports.map((p) => ({ value: p.key, label: p.name })),
      vc: ref.data.vessel_classes.map((v) => ({ value: v.name, label: v.name })),
    }
  }, [ref.data])

  if (ref.loading || !fleet) return <Loading />
  if (ref.error) return <ErrorBox error={ref.error} />

  const setCargo = (i, k, v) => setCargoes((cs) => cs.map((c, j) => (j === i ? { ...c, [k]: v } : c)))
  const setVessel = (i, k, v) => setFleet((fs) => fs.map((f, j) => (j === i ? { ...f, [k]: v } : f)))
  const r = res.data
  const saving = r && r.baseline_all_spot_today_usd ? r.baseline_all_spot_today_usd - r.objective_usd : null

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Fleet planner</h1>
        <p className="mt-1 text-sm text-muted">
          Mixed-integer programme (HiGHS) that assigns each cargo to a fleet vessel or a spot charter and a loading day
          within its laycan. Spot and index-linked hire is priced off the forecast TCE for that day. Constraints cover
          vessel intake, load- and discharge-port draft and length limits (with lightering where the port supports it),
          and day-by-day berth occupancy at each discharge port.
        </p>
      </div>

      <div className="grid gap-6 xl:grid-cols-5">
        <Card className="xl:col-span-3" title="Import programme" subtitle="Laycan in days from today"
          actions={
            <Button variant="secondary" onClick={() => setCargoes((cs) => [...cs, { id: `C${cs.length + 1}`, load_port: 'samarinda', discharge_port: 'paradip', quantity_t: 70000, laycan_start: 5, laycan_end: 15 }])}>
              <Plus className="h-3.5 w-3.5" /> Cargo
            </Button>
          }>
          <div className="overflow-x-auto">
            <table className="w-full text-[13px]">
              <thead className="text-left text-xs text-muted uppercase">
                <tr>
                  <th className="pb-2">ID</th><th className="pb-2">Load</th><th className="pb-2">Discharge</th>
                  <th className="pb-2">Tonnes</th><th className="pb-2">Laycan</th><th />
                </tr>
              </thead>
              <tbody>
                {cargoes.map((c, i) => (
                  <tr key={i} className="align-middle">
                    <td className="py-1 pr-2"><TextInput value={c.id} onChange={(v) => setCargo(i, 'id', v)} className="w-14" /></td>
                    <td className="py-1 pr-2"><Select value={c.load_port} onChange={(v) => setCargo(i, 'load_port', v)} options={opts.lp} /></td>
                    <td className="py-1 pr-2"><Select value={c.discharge_port} onChange={(v) => setCargo(i, 'discharge_port', v)} options={opts.dp} /></td>
                    <td className="py-1 pr-2"><NumberInput value={c.quantity_t} step={1000} onChange={(v) => setCargo(i, 'quantity_t', v)} className="w-24" /></td>
                    <td className="py-1 pr-2 whitespace-nowrap">
                      <NumberInput value={c.laycan_start} onChange={(v) => setCargo(i, 'laycan_start', v)} className="w-14" />
                      <span className="px-1 text-muted">–</span>
                      <NumberInput value={c.laycan_end} onChange={(v) => setCargo(i, 'laycan_end', v)} className="w-14" />
                    </td>
                    <td className="py-1">
                      <button className="text-muted hover:text-up" onClick={() => setCargoes((cs) => cs.filter((_, j) => j !== i))} aria-label="Remove cargo">
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card className="xl:col-span-2" title="Controlled fleet" subtitle="Blank hire = index-linked at forecast TCE">
          <div className="overflow-x-auto">
            <table className="w-full text-[13px]">
              <thead className="text-left text-xs text-muted uppercase">
                <tr><th className="pb-2">Vessel</th><th className="pb-2">Class</th><th className="pb-2">Open</th><th className="pb-2">Hire $/d</th></tr>
              </thead>
              <tbody>
                {fleet.map((v, i) => (
                  <tr key={v.id}>
                    <td className="py-1 pr-2 text-xs">{v.name}</td>
                    <td className="py-1 pr-2"><Select value={v.vessel_class} onChange={(x) => setVessel(i, 'vessel_class', x)} options={opts.vc} /></td>
                    <td className="py-1 pr-2"><NumberInput value={v.open_day} onChange={(x) => setVessel(i, 'open_day', x)} className="w-14" /></td>
                    <td className="py-1"><NumberInput value={v.hire_usd_day} step={250} onChange={(x) => setVessel(i, 'hire_usd_day', x)} className="w-24" /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-muted">Sample fleet for illustration; edit to match your own tonnage.</p>
        </Card>
      </div>

      <Card>
        <div className="flex flex-wrap items-end gap-5">
          <Field label="Price the spot market at" hint="A higher quantile plans against a firmer market">
            <Segmented value={quantile} onChange={setQuantile}
              options={[{ value: 'mean', label: 'Expected' }, { value: '0.75', label: 'P75' }, { value: '0.9', label: 'P90' }]} />
          </Field>
          <Field label="Slot grid (days)"><NumberInput value={slotDays} min={1} max={7} onChange={setSlotDays} className="w-20" /></Field>
          <Field label="Carbon price ($/t CO₂)"><NumberInput value={carbon} min={0} step={5} onChange={setCarbon} className="w-24" /></Field>
          <Button onClick={solve} loading={res.loading} disabled={!cargoes.length}>Optimise</Button>
          {!cargoes.length && <span className="text-xs text-up">Add at least one cargo.</span>}
        </div>
      </Card>

      <ErrorBox error={res.error} />

      {r && (
        <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
            <Stat label="Marginal programme cost" value={compactUsd(r.objective_usd)} sub={`${usd(r.usd_per_t, 2)}/t · + ${compactUsd(r.committed_hire_used_usd)} committed hire`} />
            <Stat label="All-spot today" value={compactUsd(r.baseline_all_spot_today_usd)} sub="fleet left idle; cheapest spot ship at day-1 rate" />
            <Stat label="Saving" value={compactUsd(saving)} sub={saving != null ? `${fmt((saving / r.baseline_all_spot_today_usd) * 100, 1)}% of baseline, mostly from using ships already on hire` : ''} tone={saving > 0 ? 'down' : 'up'} />
            <Stat label="Solver" value={r.status} sub={`${r.candidates} options · ${r.constraints} constraints · ${fmt(r.solve_seconds * 1000)} ms`} />
            <Stat label="Unserved" value={r.unserved.length} sub={r.unserved.length ? r.unserved.join(', ') : 'all cargoes covered'} tone={r.unserved.length ? 'up' : undefined} />
          </div>

          <Card title="Optimal assignment" subtitle={`Rates: ${r.rate_basis}; bunkers ${usd(r.inputs.bunker_usd_t)}/t`}>
            <Table
              dense
              rowKey={(a) => a.cargo_id}
              columns={[
                { key: 'cargo_id', label: 'Cargo' },
                { key: 'v', label: 'Vessel', render: (a) => <span>{a.vessel_name} <Badge tone={a.kind === 'spot' ? 'elevated' : 'info'}>{a.kind}</Badge></span> },
                { key: 'vessel_class', label: 'Class' },
                { key: 'route', label: 'Route', render: (a) => `${a.voyage.load_port} → ${a.voyage.discharge_port}` },
                { key: 'start_day', label: 'Load day', align: 'right' },
                { key: 'tce', label: 'Hire $/d', align: 'right', render: (a) => usd(a.tce_usd_day) },
                { key: 'days', label: 'Days', align: 'right', render: (a) => fmt(a.voyage.total_days, 1) },
                { key: 'light', label: 'Lightered', align: 'right', render: (a) => (a.voyage.lightered_t ? `${fmt(a.voyage.lightered_t / 1000, 1)}kt` : '–') },
                { key: 'cost', label: 'Voyage cost', align: 'right', render: (a) => compactUsd(a.voyage.total_usd) },
                { key: 'pt', label: '$/t', align: 'right', render: (a) => usd(a.voyage.usd_per_t, 2) },
                { key: 'co2', label: 'CO₂ t', align: 'right', render: (a) => fmt(a.voyage.co2_t) },
              ]}
              rows={r.assignments}
            />
          </Card>

          <div className="grid gap-6 xl:grid-cols-2">
            <Card title="Berth occupancy" subtitle="Arrival after loading, passage and the port's average wait">
              <Timeline assignments={r.assignments} ports={ref.data.discharge_ports} />
            </Card>
            <Card title="Routes" subtitle="Dashed = spot charter">
              <PortMap reference={ref.data} assignments={r.assignments} />
            </Card>
          </div>
          <Note>
            Period hire on controlled ships is already committed, so it is treated as sunk: the optimiser minimises the
            marginal cost (bunkers, ports, lightering, and hire on spot or index-linked ships), and the committed hire used is
            shown separately. Each fleet vessel lifts at most one cargo in the window. The baseline fixes every cargo on the
            cheapest feasible spot ship at today's rate. Port limits and distances are planning assumptions (see the Data tab).
          </Note>
        </>
      )}
    </div>
  )
}
