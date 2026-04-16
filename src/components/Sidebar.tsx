'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'

const navItems = [
  { href: '/', label: 'Dashboard', icon: '📊' },
  { href: '/alerts', label: 'Alertas', icon: '⚠️' },
  { href: '/map', label: 'Mapa', icon: '🗺️' },
  { href: '/settings', label: 'Ajustes', icon: '⚙️' },
]

export default function Sidebar() {
  const pathname = usePathname()

  return (
    <aside className="w-64 bg-surface dark:bg-dark-surface border-r border-border dark:border-slate-700 flex flex-col">
      <nav className="flex-1 p-4 space-y-2">
        {navItems.map((item) => {
          const isActive = pathname === item.href
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                isActive
                  ? 'bg-primary text-white'
                  : 'text-foreground dark:text-dark hover:bg-background dark:hover:bg-dark'
              }`}
            >
              <span className="text-xl">{item.icon}</span>
              <span className="font-medium">{item.label}</span>
            </Link>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-border dark:border-slate-700">
        <p className="text-xs text-muted dark:text-dark-muted">SMISIA v1.0</p>
        <p className="text-xs text-muted dark:text-dark-muted mt-1">© 2026 Agrícola</p>
      </div>
    </aside>
  )
}
