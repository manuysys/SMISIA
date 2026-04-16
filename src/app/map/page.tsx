'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { generateInitialSilobags } from '@/lib/mock-data'
import { Silobag } from '@/lib/models'

const MARKER_COLORS = {
  safe: '#16a34a',
  low: '#d97706',
  medium: '#f59e0b',
  high: '#dc2626',
}

export default function MapPage() {
  const [silobags, setSilobags] = useState<Silobag[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const mockSilobags = generateInitialSilobags()
    setSilobags(mockSilobags)
    setLoading(false)
  }, [])

  if (loading) {
    return (
      <div className="p-8 flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin text-4xl mb-4">⏳</div>
          <p className="text-muted dark:text-dark-muted">Cargando mapa...</p>
        </div>
      </div>
    )
  }

  const selectedSilobag = selectedId ? silobags.find((s) => s.id === selectedId) : null

  // Calculate bounds
  const lats = silobags.map((s) => s.latitude)
  const lngs = silobags.map((s) => s.longitude)
  const minLat = Math.min(...lats) - 0.003
  const maxLat = Math.max(...lats) + 0.003
  const minLng = Math.min(...lngs) - 0.003
  const maxLng = Math.max(...lngs) + 0.003

  const mapW = 800
  const mapH = 500

  const toX = (lng: number) => ((lng - minLng) / (maxLng - minLng)) * (mapW - 60) + 30
  const toY = (lat: number) => ((maxLat - lat) / (maxLat - minLat)) * (mapH - 80) + 40

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-foreground dark:text-dark mb-2">Mapa de Silobolsas</h1>
        <p className="text-muted dark:text-dark-muted">Vista geográfica de ubicaciones</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Map */}
        <div className="lg:col-span-3">
          <div className="bg-surface dark:bg-dark-surface border border-border dark:border-slate-700 rounded-lg p-4 relative" style={{ width: mapW, height: mapH }}>
            {/* Legend */}
            <div className="absolute top-4 left-4 bg-surface dark:bg-dark-surface border border-border dark:border-slate-700 rounded-lg p-3 space-y-2 text-sm z-10">
              {(['safe', 'low', 'medium', 'high'] as const).map((level) => {
                const count = silobags.filter((s) => s.riskLevel === level).length
                const labels = { safe: 'Seguro', low: 'Bajo', medium: 'Precaución', high: 'Peligro' }
                return (
                  <div key={level} className="flex items-center gap-2">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: MARKER_COLORS[level] }}
                    />
                    <span className="text-muted dark:text-dark-muted">{labels[level]} ({count})</span>
                  </div>
                )
              })}
            </div>

            {/* Markers */}
            {silobags.map((sb) => {
              const x = toX(sb.longitude)
              const y = toY(sb.latitude)
              const isSelected = sb.id === selectedId

              return (
                <button
                  key={sb.id}
                  onClick={() => setSelectedId(sb.id)}
                  className="absolute transform -translate-x-1/2 -translate-y-1/2 focus:outline-none transition-transform"
                  style={{
                    left: x,
                    top: y,
                    transform: `translate(-50%, -50%) scale(${isSelected ? 1.3 : 1})`,
                  }}
                >
                  <div
                    className="w-8 h-8 rounded-full border-2 border-white shadow-lg flex items-center justify-center text-white font-bold text-sm"
                    style={{ backgroundColor: MARKER_COLORS[sb.riskLevel] }}
                  >
                    {sb.riskLevel === 'high' ? '⚠' : sb.riskLevel === 'medium' ? '!' : '✓'}
                  </div>
                </button>
              )
            })}
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          <h3 className="text-lg font-bold text-foreground dark:text-dark">Silobolsas</h3>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {silobags.map((sb) => (
              <button
                key={sb.id}
                onClick={() => setSelectedId(sb.id)}
                className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${
                  selectedId === sb.id
                    ? 'bg-primary text-white'
                    : 'bg-surface dark:bg-dark-surface border border-border dark:border-slate-700 text-foreground dark:text-dark hover:border-primary'
                }`}
              >
                <p className="font-medium">{sb.name}</p>
                <p className="text-xs opacity-75">{sb.location}</p>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Selected Silobag Details */}
      {selectedSilobag && (
        <div className="bg-surface dark:bg-dark-surface border border-border dark:border-slate-700 rounded-lg p-6">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h2 className="text-2xl font-bold text-foreground dark:text-dark">{selectedSilobag.name}</h2>
              <p className="text-muted dark:text-dark-muted">📍 {selectedSilobag.location} · {selectedSilobag.grainType}</p>
            </div>
            <span className="px-4 py-2 rounded-lg font-bold text-white" style={{ backgroundColor: MARKER_COLORS[selectedSilobag.riskLevel] }}>
              {selectedSilobag.riskLevel.toUpperCase()}
            </span>
          </div>

          {selectedSilobag.lastReading && (
            <div className="grid grid-cols-3 gap-4 mb-4">
              <div className="bg-background dark:bg-dark rounded-lg p-4 text-center">
                <p className="text-xs text-muted dark:text-dark-muted">Temperatura</p>
                <p className="text-2xl font-bold text-foreground dark:text-dark mt-2">
                  🌡️ {selectedSilobag.lastReading.temperature.toFixed(1)}°C
                </p>
              </div>
              <div className="bg-background dark:bg-dark rounded-lg p-4 text-center">
                <p className="text-xs text-muted dark:text-dark-muted">Humedad</p>
                <p className="text-2xl font-bold text-foreground dark:text-dark mt-2">
                  💧 {selectedSilobag.lastReading.humidity.toFixed(1)}%
                </p>
              </div>
              <div className="bg-background dark:bg-dark rounded-lg p-4 text-center">
                <p className="text-xs text-muted dark:text-dark-muted">CO₂</p>
                <p className="text-2xl font-bold text-foreground dark:text-dark mt-2">
                  🌬️ {selectedSilobag.lastReading.co2.toFixed(0)} ppm
                </p>
              </div>
            </div>
          )}

          <Link
            href={`/silobag/${selectedSilobag.id}`}
            className="inline-block px-6 py-2 bg-primary text-white rounded-lg hover:opacity-90 transition-opacity font-medium"
          >
            Ver Detalle →
          </Link>
        </div>
      )}
    </div>
  )
}
