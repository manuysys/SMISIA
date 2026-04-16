interface KPICardProps {
  title: string
  value: string
  icon: string
  color?: 'primary' | 'success' | 'warning' | 'error'
}

export default function KPICard({
  title,
  value,
  icon,
  color = 'primary',
}: KPICardProps) {
  const colorClasses = {
    primary: 'bg-primary',
    success: 'bg-success',
    warning: 'bg-warning',
    error: 'bg-error',
  }

  return (
    <div className="bg-surface dark:bg-dark-surface border border-border dark:border-slate-700 rounded-lg p-6 hover:shadow-lg transition-shadow">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-muted dark:text-dark-muted font-medium">{title}</p>
          <p className="text-3xl font-bold text-foreground dark:text-dark mt-2">{value}</p>
        </div>
        <div className={`${colorClasses[color]} text-white p-3 rounded-lg text-2xl`}>
          {icon}
        </div>
      </div>
    </div>
  )
}
