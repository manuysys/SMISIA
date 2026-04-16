import { Silobag } from '@/lib/models'

const riskColors = {
  safe: 'bg-success text-white',
  low: 'bg-warning text-white',
  medium: 'bg-warning text-white',
  high: 'bg-error text-white',
}

const riskLabels = {
  safe: 'SEGURO',
  low: 'BAJO',
  medium: 'PRECAUCIÓN',
  high: 'PELIGRO',
}

interface SilobagCardProps {
  silobag: Silobag
}

export default function SilobagCard({ silobag }: SilobagCardProps) {
  const r = silobag.lastReading

  return (
    <div className="bg-surface dark:bg-dark-surface border-l-4 border-l-primary rounded-lg p-6 hover:shadow-lg transition-shadow cursor-pointer">
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="text-lg font-bold text-foreground dark:text-dark">{silobag.name}</h3>
          <p className="text-sm text-muted dark:text-dark-muted">📍 {silobag.location} · {silobag.grainType}</p>
        </div>
        <span className={`px-3 py-1 rounded-full text-xs font-bold ${riskColors[silobag.riskLevel]}`}>
          {riskLabels[silobag.riskLevel]}
        </span>
      </div>

      {r && (
        <div className="grid grid-cols-3 gap-3 mb-4">
          <div className="bg-background dark:bg-dark rounded-lg p-3 text-center">
            <p className="text-xs text-muted dark:text-dark-muted">Temp</p>
            <p className="text-lg font-bold text-foreground dark:text-dark">🌡️ {r.temperature.toFixed(1)}°C</p>
          </div>
          <div className="bg-background dark:bg-dark rounded-lg p-3 text-center">
            <p className="text-xs text-muted dark:text-dark-muted">Humedad</p>
            <p className="text-lg font-bold text-foreground dark:text-dark">💧 {r.humidity.toFixed(1)}%</p>
          </div>
          <div className="bg-background dark:bg-dark rounded-lg p-3 text-center">
            <p className="text-xs text-muted dark:text-dark-muted">CO₂</p>
            <p className="text-lg font-bold text-foreground dark:text-dark">🌬️ {r.co2.toFixed(0)}</p>
          </div>
        </div>
      )}

      <p className="text-xs text-muted dark:text-dark-muted">
        Última lectura: {r ? new Date(r.timestamp).toLocaleTimeString('es-AR') : 'N/A'}
      </p>
    </div>
  )
}
