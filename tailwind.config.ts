import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: '#0d9488',
        success: '#16a34a',
        warning: '#d97706',
        error: '#dc2626',
        background: '#f9fafb',
        surface: '#ffffff',
        foreground: '#111827',
        muted: '#6b7280',
        border: '#e5e7eb',
      },
      backgroundColor: {
        dark: '#0f172a',
        'dark-surface': '#1e293b',
      },
      textColor: {
        dark: '#f1f5f9',
        'dark-muted': '#94a3b8',
      },
    },
  },
  plugins: [],
  darkMode: 'class',
}
export default config
