'use client'

import { useState } from 'react'
import Link from 'next/link'

export default function Header() {
  const [darkMode, setDarkMode] = useState(false)

  const toggleDarkMode = () => {
    setDarkMode(!darkMode)
    document.documentElement.classList.toggle('dark')
  }

  return (
    <header className="border-b border-border bg-surface dark:bg-dark-surface dark:border-slate-700 px-6 py-4 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div className="text-2xl font-bold text-primary">🌾</div>
        <h1 className="text-2xl font-bold text-foreground dark:text-dark">SMISIA</h1>
        <p className="text-sm text-muted dark:text-dark-muted ml-2">Monitoreo de Silobolsas</p>
      </div>

      <div className="flex items-center gap-4">
        {/* Notifications Bell */}
        <button className="relative p-2 hover:bg-background dark:hover:bg-dark rounded-lg transition-colors">
          <span className="text-xl">🔔</span>
          <span className="absolute top-0 right-0 w-2 h-2 bg-error rounded-full"></span>
        </button>

        {/* Dark Mode Toggle */}
        <button
          onClick={toggleDarkMode}
          className="p-2 hover:bg-background dark:hover:bg-dark rounded-lg transition-colors"
        >
          {darkMode ? '☀️' : '🌙'}
        </button>

        {/* User Profile */}
        <div className="flex items-center gap-2 pl-4 border-l border-border dark:border-slate-700">
          <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center text-white text-sm font-bold">
            👤
          </div>
          <span className="text-sm font-medium text-foreground dark:text-dark">Admin</span>
        </div>
      </div>
    </header>
  )
}
