# SMISIA — Project TODO

## Setup & Foundation
- [x] Generate and set app logo/icon
- [x] Update theme colors (agricultural/industrial palette)
- [x] Configure navigation: 4 bottom tabs (Dashboard, Alertas, Mapa, Ajustes)
- [x] Add icon mappings for all tabs in icon-symbol.tsx
- [x] Create data models (Silobag, SensorReading, Alert)
- [x] Build mock IoT data engine with auto-generation
- [x] Build AI alert engine (rule-based: temp+humidity+CO2 thresholds)
- [x] Create global state/context for silobags and alerts

## Screens
- [x] Dashboard screen: KPI row + silobag FlatList cards
- [x] Silobag Detail screen: readings, AI prediction, recommendations, charts, history
- [x] Alerts screen: filtered list with priority badges
- [x] Map screen: markers with status colors + bottom sheet
- [x] Settings screen: thresholds, intervals, notifications, simulation controls

## Features
- [x] Pull-to-refresh on Dashboard (simulate new sensor data)
- [x] Time-series line charts (24h/7d/30d toggle)
- [x] AI risk prediction card with confidence and explanation
- [x] Auto-generated recommendations based on sensor readings
- [x] Alert priority classification (Bajo/Medio/Alto)
- [x] Settings persistence with AsyncStorage
- [x] "Generar datos de prueba" button in Settings
- [x] Notification bell badge on header

## Branding
- [x] Update app.config.ts with app name and logo URL
- [x] Update tailwind.config.js with SMISIA color palette
