import { useGet } from '../lib/hooks'
import { Field, NumberInput, Select } from './ui'

export const DEFAULT_ROUTE = { load_port: 'samarinda', discharge_port: 'paradip', vessel_class: 'Panamax', cargo_t: 70000 }

export default function RouteForm({ value, onChange, children }) {
  const ref = useGet('/api/v1/reference')
  if (!ref.data) return null
  const set = (k) => (v) => onChange({ ...value, [k]: v })
  return (
    <div className="flex flex-wrap items-end gap-4">
      <Field label="Load port">
        <Select value={value.load_port} onChange={set('load_port')} options={ref.data.load_ports.map((p) => ({ value: p.key, label: `${p.name} (${p.country})` }))} />
      </Field>
      <Field label="Discharge port">
        <Select value={value.discharge_port} onChange={set('discharge_port')} options={ref.data.discharge_ports.map((p) => ({ value: p.key, label: p.name }))} />
      </Field>
      <Field label="Vessel class">
        <Select value={value.vessel_class} onChange={set('vessel_class')} options={ref.data.vessel_classes.map((v) => ({ value: v.name, label: `${v.name} (${Math.round(v.dwt / 1000)}k dwt)` }))} />
      </Field>
      <Field label="Cargo (t)">
        <NumberInput value={value.cargo_t} step={1000} min={1000} onChange={set('cargo_t')} className="w-28" />
      </Field>
      {children}
    </div>
  )
}
