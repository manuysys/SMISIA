'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { generateInitialSilobags, generateInitialAlerts } from '@/lib/mock-data'
import { Silobag, Alert } from '@/lib/models'
import AlertCard from '@/components/AlertCard'

export default function AlertsPage() {
  const [silobags, setSilobags] = useState<Silobag[]>([])
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [filter, setFilter] = useState<'all' | 'high' | 'medium' | 'low'>('all')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const mockSilobags = generateInitialSilobags()
    setSilobags(mockSilobags)
    setAlerts(generateInitialAlerts(mockSilobags))
    setLoading(false)
  }, [])

  const filteredAlerts = filter === 'all' 
    ? alerts 
    : alerts.filter((a) => a.riskLevel === filter)

  if (loading) {
    return (
      <div className="p-8 flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin text-4xl mb-4">⏳</div>
          <p className="text-muted dark:text-dark-muted">Cargando alertas...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-foreground dark:text-dark mb-2">Alertas</h1>
        <p className="text-muted dark:text-dark-muted">Gestión de alertas del sistema</p>
      </div>

      {/* Filter Tabs */}
      <div className="flex gap-2 border-b border-border dark:border-slate-700">
        {(['all', 'high', 'medium', 'low'] as const).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-4 py-3 font-medium transition-colors border-b-2 ${
              filter === f
                ? 'border-primary text-primary'
                : 'border-transparent text-muted dark:text-dark-muted hover:text-foreground dark:hover:text-dark'
            }`}
          >
            {f === 'all' && `Todas (${alerts.length})`}
            {f === 'high' && `Alto (${alerts.filter((a) => a.riskLevel === 'high').length})`}
            {f === 'medium' && `Medio (${alerts.filter((a) => a.riskLevel === 'medium').length})`}
            {f === 'low' && `Bajo (${alerts.filter((a) => a.riskLevel === 'low').length})`}
          </button>
        ))}
      </div>

      {/* Alerts List */}
      <div className="space-y-3">
        {filteredAlerts.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-2xl mb-2">✅</p>
            <p className="text-muted dark:text-dark-muted">No hay alertas en esta categoría</p>
          </div>
        ) : (
          filteredAlerts.map((alert) => {
            const silobag = silobags.find((s) => s.id === alert.silobagId)
            return (
              <Link
                key={alert.id}
                href={`/silobag/${alert.silobagId}`}
                className="block hover:opacity-75 transition-opacity"
              >
                <AlertCard
                  alert={alert}
                  silobagName={silobag?.name || 'Unknown'}
                />
              </Link>
            )
          })
        )}
      </div>
    </div>
  )
}
