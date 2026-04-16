'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { useParams } from 'next/navigation'
import { generateInitialSilobags } from '@/lib/mock-data'
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

export default function SilobagDetailPage() {
  const params = useParams()
  const id = params.id as string
  const [silobag, setSilobag] = useState<Silobag | null>(null)
  const [timeRange, setTimeRange] = useState<'24h' | '7d' | '30d'>('24h')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const mockSilobags = generateInitialSilobags()
    const found = mockSilobags.find((s) => s.id === id)
    setSilobag(found || null)
    setLoading(false)
  }, [id])

  if (loading) {
    return (
      <div className="p-8 flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin text-4xl mb-4">⏳</div>
          <p className="text-muted dark:text-dark-muted">Cargando silobolsa...</p>
        </div>
      </div>
    )
  }

  if (!silobag) {
    return (
      <div className="p-8">
        <div className="text-center">
          <p className="text-2xl mb-4">❌</p>
          <p className="text-foreground dark:text-dark mb-4">Silobolsa no encontrada</p>
          <Link href="/" className="text-primary hover:underline">
            Volver al dashboard
          </Link>
        </div>
      </div>
    )
  }

  const r = silobag.lastReading

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <Link href="/" className="text-primary hover:underline text-sm mb-2 inline-block">
            ← Volver
          </Link>
          <h1 className="text-3xl font-bold text-foreground dark:text-dark">{silobag.name}</h1>
          <p className="text-muted dark:text-dark-muted">📍 {silobag.location} · {silobag.grainType}</p>
        </div>
        <span className={`px-4 py-2 rounded-lg font-bold ${riskColors[silobag.riskLevel]}`}>
          {riskLabels[silobag.riskLevel]}
        </span>
      </div>

      {/* Current Readings */}
      {r && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-surface dark:bg-dark-surface border border-border dark:border-slate-700 rounded-lg p-6">
            <p className="text-sm text-muted dark:text-dark-muted font-medium">Temperatura</p>
            <p className="text-4xl font-bold text-foreground dark:text-dark mt-2">🌡️ {r.temperature.toFixed(1)}°C</p>
            <p className="text-xs text-muted dark:text-dark-muted mt-2">Máximo: 28°C</p>
          </div>
          <div className="bg-surface dark:bg-dark-surface border border-border dark:border-slate-700 rounded-lg p-6">
            <p className="text-sm text-muted dark:text-dark-muted font-medium">Humedad</p>
            <p className="text-4xl font-bold text-foreground dark:text-dark mt-2">💧 {r.humidity.toFixed(1)}%</p>
            <p className="text-xs text-muted dark:text-dark-muted mt-2">Máximo: 18%</p>
          </div>
          <div className="bg-surface dark:bg-dark-surface border border-border dark:border-slate-700 rounded-lg p-6">
            <p className="text-sm text-muted dark:text-dark-muted font-medium">CO₂</p>
            <p className="text-4xl font-bold text-foreground dark:text-dark mt-2">🌬️ {r.co2.toFixed(0)} ppm</p>
            <p className="text-xs text-muted dark:text-dark-muted mt-2">Máximo: 1000 ppm</p>
          </div>
        </div>
      )}

      {/* AI Prediction */}
      {silobag.prediction && (
        <div className="bg-surface dark:bg-dark-surface border-l-4 border-l-primary rounded-lg p-6">
          <h2 className="text-xl font-bold text-foreground dark:text-dark mb-2">🤖 Predicción IA</h2>
          <p className="text-sm text-muted dark:text-dark-muted mb-4">
            Confianza: <span className="font-bold text-foreground dark:text-dark">{(silobag.prediction.confidence * 100).toFixed(0)}%</span>
          </p>
          <p className="text-foreground dark:text-dark mb-4">{silobag.prediction.explanation}</p>
          <div className="bg-background dark:bg-dark rounded-lg p-4">
            <p className="text-sm font-bold text-foreground dark:text-dark mb-2">Recomendaciones:</p>
            <ul className="text-sm text-muted dark:text-dark-muted space-y-1">
              {silobag.prediction.recommendations.map((rec, i) => (
                <li key={i}>• {rec}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Charts */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold text-foreground dark:text-dark">Gráficos</h2>
          <div className="flex gap-2">
            {(['24h', '7d', '30d'] as const).map((range) => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  timeRange === range
                    ? 'bg-primary text-white'
                    : 'bg-background dark:bg-dark text-foreground dark:text-dark border border-border dark:border-slate-700'
                }`}
              >
                {range}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Temperature Chart Placeholder */}
          <div className="bg-surface dark:bg-dark-surface border border-border dark:border-slate-700 rounded-lg p-6">
            <h3 className="font-bold text-foreground dark:text-dark mb-4">Temperatura ({timeRange})</h3>
            <div className="h-48 bg-background dark:bg-dark rounded-lg flex items-center justify-center text-muted dark:text-dark-muted">
              📊 Gráfico de línea (Recharts)
            </div>
          </div>

          {/* Humidity Chart Placeholder */}
          <div className="bg-surface dark:bg-dark-surface border border-border dark:border-slate-700 rounded-lg p-6">
            <h3 className="font-bold text-foreground dark:text-dark mb-4">Humedad ({timeRange})</h3>
            <div className="h-48 bg-background dark:bg-dark rounded-lg flex items-center justify-center text-muted dark:text-dark-muted">
              📊 Gráfico de línea (Recharts)
            </div>
          </div>

          {/* CO2 Chart Placeholder */}
          <div className="bg-surface dark:bg-dark-surface border border-border dark:border-slate-700 rounded-lg p-6 md:col-span-2">
            <h3 className="font-bold text-foreground dark:text-dark mb-4">CO₂ ({timeRange})</h3>
            <div className="h-48 bg-background dark:bg-dark rounded-lg flex items-center justify-center text-muted dark:text-dark-muted">
              📊 Gráfico de línea (Recharts)
            </div>
          </div>
        </div>
      </div>

      {/* History */}
      <div>
        <h2 className="text-2xl font-bold text-foreground dark:text-dark mb-4">Historial</h2>
        <div className="bg-surface dark:bg-dark-surface border border-border dark:border-slate-700 rounded-lg overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-background dark:bg-dark border-b border-border dark:border-slate-700">
              <tr>
                <th className="px-6 py-3 text-left font-bold text-foreground dark:text-dark">Fecha</th>
                <th className="px-6 py-3 text-left font-bold text-foreground dark:text-dark">Temp</th>
                <th className="px-6 py-3 text-left font-bold text-foreground dark:text-dark">Humedad</th>
                <th className="px-6 py-3 text-left font-bold text-foreground dark:text-dark">CO₂</th>
              </tr>
            </thead>
            <tbody>
              {silobag.history.slice(0, 10).map((reading, i) => (
                <tr key={i} className="border-b border-border dark:border-slate-700 hover:bg-background dark:hover:bg-dark transition-colors">
                  <td className="px-6 py-3 text-muted dark:text-dark-muted">
                    {new Date(reading.timestamp).toLocaleString('es-AR')}
                  </td>
                  <td className="px-6 py-3 text-foreground dark:text-dark">{reading.temperature.toFixed(1)}°C</td>
                  <td className="px-6 py-3 text-foreground dark:text-dark">{reading.humidity.toFixed(1)}%</td>
                  <td className="px-6 py-3 text-foreground dark:text-dark">{reading.co2.toFixed(0)} ppm</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
