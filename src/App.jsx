import { useEffect, useState } from 'react'
import { Activity, Anchor, BarChart3, Database, Gauge, LineChart, Ship, Timer } from 'lucide-react'
import { useGet } from './lib/hooks'
import { ErrorBox, Loading } from './components/ui'
import Overview from './pages/Overview'
import Forecast from './pages/Forecast'
import Validation from './pages/Validation'
import FleetPlanner from './pages/FleetPlanner'
import Fixing from './pages/Fixing'
import Costs from './pages/Costs'
import DataPage from './pages/Data'

const PAGES = [
  { id: 'overview', label: 'Overview', icon: Gauge, el: Overview },
  { id: 'forecast', label: 'Forecasts', icon: LineChart, el: Forecast },
  { id: 'validation', label: 'Model validation', icon: BarChart3, el: Validation },
  { id: 'fleet', label: 'Fleet planner', icon: Ship, el: FleetPlanner },
  { id: 'fixing', label: 'Fixing & speed', icon: Timer, el: Fixing },
  { id: 'costs', label: 'Landed cost & stress', icon: Activity, el: Costs },
  { id: 'data', label: 'Data', icon: Database, el: DataPage },
]

function useHashPage() {
  const read = () => {
    const h = window.location.hash.replace('#/', '')
    return PAGES.some((p) => p.id === h) ? h : 'overview'
  }
  const [page, setPage] = useState(read)
  useEffect(() => {
    const on = () => setPage(read())
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return page
}

export default function App() {
  const page = useHashPage()
  const status = useGet('/api/v1/status')
  const Page = PAGES.find((p) => p.id === page).el
  const s = status.data

  return (
    <div className="flex min-h-full flex-col">
      <header className="sticky top-0 z-[1000] border-b border-line bg-white/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3 md:px-6">
          <a href="#/overview" className="flex items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand text-white">
              <Anchor className="h-4.5 w-4.5" />
            </span>
            <span className="leading-tight">
              <span className="block text-[15px] font-semibold">OceanPulse</span>
              <span className="block text-[11px] text-muted">Dry-bulk freight intelligence · Indian east coast</span>
            </span>
          </a>
          <nav className="order-3 -mx-1 flex w-full gap-1 overflow-x-auto md:order-2 md:w-auto md:flex-1">
            {PAGES.map(({ id, label, icon: Icon }) => (
              <a
                key={id}
                href={`#/${id}`}
                className={`flex shrink-0 items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[13px] font-medium transition ${
                  page === id ? 'bg-brand-soft text-brand' : 'text-slate-600 hover:bg-slate-50 hover:text-ink'
                }`}
              >
                <Icon className="h-4 w-4" />
                {label}
              </a>
            ))}
          </nav>
          <div className="order-2 ml-auto text-right text-[11px] leading-tight text-muted md:order-3">
            {s?.ready ? (
              <>
                <div>
                  Market data <span className="num font-medium text-ink">{s.data_as_of}</span>
                </div>
                <div>
                  BDI print <span className="num font-medium text-ink">{s.bdi_last_print ?? 'n/a'}</span>
                </div>
              </>
            ) : status.loading ? (
              'Connecting…'
            ) : (
              <span className="text-up">API unavailable</span>
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-6 md:px-6">
        {status.loading && !s ? (
          <Loading label="Connecting to the forecasting API…" />
        ) : status.error ? (
          <ErrorBox error={status.error} />
        ) : s && !s.ready ? (
          <ErrorBox error={{ message: `Model state not loaded: ${s.last_error || 'starting up'}` }} />
        ) : (
          <Page />
        )}
      </main>

      <footer className="border-t border-line bg-white">
        <div className="mx-auto max-w-[1400px] px-4 py-3 text-[11px] text-muted md:px-6">
          Market data: Yahoo Finance (BDRY, Brent, iron ore, coal proxy, copper, FX, VIX, S&amp;P 500, listed owners) and a
          user-supplied Koyfin BDI export. Every model metric shown is out-of-sample (walk-forward). Not investment advice.
        </div>
      </footer>
    </div>
  )
}
