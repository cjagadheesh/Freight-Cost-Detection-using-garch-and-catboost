import { useEffect, useRef } from 'react'
import L from 'leaflet'

const COLORS = ['#0b5cad', '#c2410c', '#067647', '#7c3aed', '#b42318', '#0e7490', '#a16207', '#be185d']

/** Load and discharge ports with straight-line route markers for the chosen assignments. */
export default function PortMap({ reference, assignments = [], height = 360 }) {
  const el = useRef(null)
  const map = useRef(null)
  const layer = useRef(null)

  useEffect(() => {
    if (!el.current || map.current) return undefined
    map.current = L.map(el.current, { zoomControl: true, scrollWheelZoom: false, worldCopyJump: true }).setView([5, 105], 3)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
      maxZoom: 10,
    }).addTo(map.current)
    layer.current = L.layerGroup().addTo(map.current)
    return () => {
      map.current.remove()
      map.current = null
    }
  }, [])

  useEffect(() => {
    if (!map.current || !reference) return
    const g = layer.current
    g.clearLayers()
    const lp = Object.fromEntries(reference.load_ports.map((p) => [p.key, p]))
    const dp = Object.fromEntries(reference.discharge_ports.map((p) => [p.key, p]))
    const used = new Set(assignments.flatMap((a) => [a.voyage.load_port, a.voyage.discharge_port]))

    assignments.forEach((a, i) => {
      const o = lp[a.voyage.load_port]
      const d = dp[a.voyage.discharge_port]
      if (!o || !d) return
      // Draw west of the antimeridian for the US East Coast origin.
      const olon = o.lon < -30 ? o.lon + 360 : o.lon
      L.polyline([[o.lat, olon], [d.lat, d.lon]], { color: COLORS[i % COLORS.length], weight: 2, opacity: 0.8, dashArray: a.kind === 'spot' ? '5 5' : null })
        .bindTooltip(`${a.cargo_id}: ${a.vessel_name} (${a.vessel_class}), ${Math.round(a.voyage.cargo_t / 1000)}kt`)
        .addTo(g)
    })
    for (const p of reference.load_ports) {
      const lon = p.lon < -30 ? p.lon + 360 : p.lon
      L.circleMarker([p.lat, lon], { radius: used.has(p.key) ? 6 : 4, color: '#334155', fillColor: '#f8fafc', fillOpacity: 1, weight: 2 })
        .bindTooltip(`<b>${p.name}</b><br/>${p.commodity}<br/>Max draft ${p.max_draft_m} m`)
        .addTo(g)
    }
    for (const p of reference.discharge_ports) {
      L.circleMarker([p.lat, p.lon], { radius: used.has(p.key) ? 7 : 5, color: '#0b5cad', fillColor: '#0b5cad', fillOpacity: 0.85, weight: 1 })
        .bindTooltip(`<b>${p.name}</b><br/>Draft ${p.max_draft_m} m · LOA ${p.max_loa_m} m · ${p.berths} berth(s)${p.lightering_via ? `<br/>Lightering via ${p.lightering_via}` : ''}`)
        .addTo(g)
    }
    // Fit to the ports in play (or all of them) once the container has its final size.
    const pts = []
    for (const p of [...reference.load_ports, ...reference.discharge_ports]) {
      if (used.size && !used.has(p.key)) continue
      pts.push([p.lat, p.lon < -30 ? p.lon + 360 : p.lon])
    }
    map.current.invalidateSize()
    if (pts.length) map.current.fitBounds(pts, { padding: [24, 24], maxZoom: 5 })
  }, [reference, assignments])

  return <div ref={el} style={{ height }} className="w-full overflow-hidden rounded-lg border border-line" />
}
