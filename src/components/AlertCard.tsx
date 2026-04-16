import { Alert } from '@/lib/models'

const riskColors = {
  low: 'bg-warning/10 border-warning text-warning',
  medium: 'bg-warning/10 border-warning text-warning',
  high: 'bg-error/10 border-error text-error',
}

const riskLabels = {
  low: 'Bajo',
  medium: 'Medio',
  high: 'Alto',
}

interface AlertCardProps {
  alert: Alert
  silobagName: string
}

export default function AlertCard({ alert, silobagName }: AlertCardProps) {
  return (
    <div className={`border-l-4 rounded-lg p-4 bg-surface dark:bg-dark-surface ${riskColors[alert.riskLevel]}`}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <h4 className="font-bold text-foreground dark:text-dark">{silobagName}</h4>
          <p className="text-sm text-muted dark:text-dark-muted mt-1">
            {alert.message}
          </p>
          <p className="text-xs text-muted dark:text-dark-muted mt-2">
            {new Date(alert.timestamp).toLocaleString('es-AR')}
          </p>
        </div>
        <span className={`px-3 py-1 rounded-full text-xs font-bold whitespace-nowrap ml-4`}>
          {riskLabels[alert.riskLevel]}
        </span>
      </div>
    </div>
  )
}
