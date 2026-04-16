# SMISIA Web — Project TODO

## Setup & Foundation
- [ ] Copy data models from mobile (lib/models.ts)
- [ ] Copy mock data engine from mobile (lib/mock-data.ts)
- [ ] Set up Tailwind CSS color palette matching mobile
- [ ] Create layout components: Header, Sidebar, MainContent
- [ ] Set up routing structure (dashboard, silobag/[id], alerts, map, settings)

## Screens
- [ ] Dashboard: KPI cards + silobag grid + alert summary
- [ ] Silobag Detail: readings + AI prediction + charts + history
- [ ] Alerts: filtered list with priority tabs
- [ ] Map: Leaflet map with markers + bottom sidebar
- [ ] Settings: thresholds + intervals + notifications + simulation

## Features
- [ ] Pull-to-refresh on Dashboard (mobile)
- [ ] Time-series line charts (24h/7d/30d toggle) using Recharts
- [ ] AI risk prediction card with confidence and explanation
- [ ] Auto-generated recommendations based on sensor readings
- [ ] Alert priority classification (Bajo/Medio/Alto)
- [ ] Settings persistence with localStorage
- [ ] "Generar datos de prueba" button in Settings
- [ ] Notification badge on header
- [ ] Dark mode toggle

## Branding
- [ ] Update app title and favicon
- [ ] Add SMISIA logo to header
- [ ] Apply color palette to all components

## Testing & Polish
- [ ] Unit tests for data models and AI engine
- [ ] Component tests for key screens
- [ ] Responsive design testing (desktop/tablet/mobile)
- [ ] Dark mode testing
- [ ] Performance optimization (lazy loading, code splitting)
- [ ] Accessibility audit (WCAG 2.1 AA)

## Deployment
- [ ] Create web branch and push to GitHub
- [ ] Set up CI/CD pipeline
- [ ] Deploy to staging environment
