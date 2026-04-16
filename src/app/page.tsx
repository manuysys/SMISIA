'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { generateInitialSilobags, generateInitialAlerts } from '@/lib/mock-data'
import { Silobag, Alert } from '@/lib/models'
import KPICard from '@/components/KPICard'
import SilobagCard from '@/components/SilobagCard'
import AlertCard from '@/components/AlertCard'

export default function Dashboard() {
  const [silobags, setSilobags] = useState<Silobag[]>([])
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const mockSilobags = generateInitialSilobags()
    setSilobags(mockSilobags)
    setAlerts(generateInitialAlerts(mockSilobags))
    setLoading(false)
  }, [])

  if (loading) {
    return (
      <div className="p-8 flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin text-4xl mb-4">⏳</div>
          <p className="text-muted dark:text-dark-muted">Cargando datos...</p>
        </div>
      </div>
    )
  }

  const highRiskCount = silobags.filter((s) => s.riskLevel === 'high').length
  const avgTemp = (silobags.reduce((sum, s) => sum + (s.lastReading?.temperature || 0), 0) / silobags.length).toFixed(1)

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-foreground dark:text-dark mb-2">Dashboard</h1>
        <p className="text-muted dark:text-dark-muted">Monitoreo en tiempo real de silobolsas</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <KPICard title="Silobolsas" value={String(silobags.length)} icon="📦" />
        <KPICard title="Alertas" value={String(alerts.length)} icon="⚠️" color="error" />
        <KPICard title="Temp. Prom." value={`${avgTemp}°C`} icon="🌡️" />
        <KPICard title="Riesgo Alto" value={String(highRiskCount)} icon="🔴" color="error" />
      </div>

      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold text-foreground dark:text-dark">Silobolsas Activas</h2>
          <button className="px-4 py-2 bg-primary text-white rounded-lg hover:opacity-90 transition-opacity">
            🔄 Actualizar
          </button>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {silobags.map((silobag) => (
            <Link key={silobag.id} href={`/silobag/${silobag.id}`}>
              <SilobagCard silobag={silobag} />
            </Link>
          ))}
        </div>
      </div>

      <div>
        <h2 className="text-2xl font-bold text-foreground dark:text-dark mb-4">Alertas Recientes</h2>
        <div className="space-y-3">
          {alerts.slice(0, 5).map((alert) => {
            const silobag = silobags.find((s) => s.id === alert.silobagId)
            return (
              <AlertCard
                key={alert.id}
                alert={alert}
                silobagName={silobag?.name || 'Unknown'}
              />
            )
          })}
        </div>
      </div>
    </div>
  )
}
