'use client'

import { useState } from 'react'
import { DEFAULT_SETTINGS } from '@/lib/models'

export default function SettingsPage() {
  const [maxTemp, setMaxTemp] = useState(String(DEFAULT_SETTINGS.maxTemperature))
  const [maxHumidity, setMaxHumidity] = useState(String(DEFAULT_SETTINGS.maxHumidity))
  const [maxCO2, setMaxCO2] = useState(String(DEFAULT_SETTINGS.maxCO2))
  const [interval, setInterval] = useState(DEFAULT_SETTINGS.samplingIntervalHours)
  const [notifications, setNotifications] = useState(DEFAULT_SETTINGS.pushNotificationsEnabled)

  const handleSave = () => {
    alert('Configuración guardada correctamente')
  }

  return (
    <div className="p-8 space-y-8 max-w-2xl">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-foreground dark:text-dark mb-2">Ajustes</h1>
        <p className="text-muted dark:text-dark-muted">Configuración del sistema SMISIA</p>
      </div>

      {/* Thresholds */}
      <div className="bg-surface dark:bg-dark-surface border border-border dark:border-slate-700 rounded-lg p-6">
        <h2 className="text-xl font-bold text-foreground dark:text-dark mb-4">⚠️ Umbrales de Alerta</h2>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-foreground dark:text-dark mb-2">
              Temperatura máxima (°C)
            </label>
            <input
              type="number"
              value={maxTemp}
              onChange={(e) => setMaxTemp(e.target.value)}
              className="w-full px-4 py-2 border border-border dark:border-slate-700 rounded-lg bg-background dark:bg-dark text-foreground dark:text-dark"
            />
            <p className="text-xs text-muted dark:text-dark-muted mt-1">Umbral para alerta de temperatura</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-foreground dark:text-dark mb-2">
              Humedad máxima (%)
            </label>
            <input
              type="number"
              value={maxHumidity}
              onChange={(e) => setMaxHumidity(e.target.value)}
              className="w-full px-4 py-2 border border-border dark:border-slate-700 rounded-lg bg-background dark:bg-dark text-foreground dark:text-dark"
            />
            <p className="text-xs text-muted dark:text-dark-muted mt-1">Umbral para alerta de humedad</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-foreground dark:text-dark mb-2">
              CO₂ máximo (ppm)
            </label>
            <input
              type="number"
              value={maxCO2}
              onChange={(e) => setMaxCO2(e.target.value)}
              className="w-full px-4 py-2 border border-border dark:border-slate-700 rounded-lg bg-background dark:bg-dark text-foreground dark:text-dark"
            />
            <p className="text-xs text-muted dark:text-dark-muted mt-1">Umbral para alerta de gases</p>
          </div>

          <button
            onClick={handleSave}
            className="w-full px-4 py-2 bg-primary text-white rounded-lg hover:opacity-90 transition-opacity font-medium"
          >
            Guardar Umbrales
          </button>
        </div>
      </div>

      {/* Sampling Interval */}
      <div className="bg-surface dark:bg-dark-surface border border-border dark:border-slate-700 rounded-lg p-6">
        <h2 className="text-xl font-bold text-foreground dark:text-dark mb-4">🕐 Intervalo de Muestreo</h2>
        
        <div className="grid grid-cols-3 gap-2">
          {[1, 2, 4].map((h) => (
            <button
              key={h}
              onClick={() => setInterval(h)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                interval === h
                  ? 'bg-primary text-white'
                  : 'bg-background dark:bg-dark text-foreground dark:text-dark border border-border dark:border-slate-700 hover:border-primary'
              }`}
            >
              {h}h
            </button>
          ))}
        </div>
      </div>

      {/* Notifications */}
      <div className="bg-surface dark:bg-dark-surface border border-border dark:border-slate-700 rounded-lg p-6">
        <h2 className="text-xl font-bold text-foreground dark:text-dark mb-4">🔔 Notificaciones</h2>
        
        <label className="flex items-center gap-3 cursor-pointer">
          <input
            type="checkbox"
            checked={notifications}
            onChange={(e) => setNotifications(e.target.checked)}
            className="w-5 h-5 rounded"
          />
          <span className="text-foreground dark:text-dark font-medium">Habilitar notificaciones push</span>
        </label>
      </div>

      {/* Simulation */}
      <div className="bg-surface dark:bg-dark-surface border border-border dark:border-slate-700 rounded-lg p-6">
        <h2 className="text-xl font-bold text-foreground dark:text-dark mb-4">📡 Simulación IoT</h2>
        
        <div className="space-y-3">
          <button className="w-full px-4 py-2 bg-primary text-white rounded-lg hover:opacity-90 transition-opacity font-medium">
            📡 Generar Datos de Prueba
          </button>
          <button className="w-full px-4 py-2 bg-error text-white rounded-lg hover:opacity-90 transition-opacity font-medium">
            🔄 Resetear Todos los Datos
          </button>
        </div>
      </div>

      {/* About */}
      <div className="bg-surface dark:bg-dark-surface border border-border dark:border-slate-700 rounded-lg p-6">
        <h2 className="text-xl font-bold text-foreground dark:text-dark mb-4">ℹ️ Acerca de SMISIA</h2>
        
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-muted dark:text-dark-muted">Sistema</span>
            <span className="font-medium text-foreground dark:text-dark">SMISIA v1.0</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted dark:text-dark-muted">Protocolo IoT</span>
            <span className="font-medium text-foreground dark:text-dark">LoRa / HTTP</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted dark:text-dark-muted">IA</span>
            <span className="font-medium text-foreground dark:text-dark">Motor de reglas + ML</span>
          </div>
        </div>
      </div>
    </div>
  )
}
