# SMISIA — Mobile App Interface Design

## Brand Identity

**App Name:** SMISIA  
**Full Name:** Sistema de Monitoreo Inteligente de Silobolsas con IA  
**Tagline:** Monitoreo inteligente para tu cosecha  
**Target Users:** Agricultural producers (non-technical), agronomists, field managers

### Color Palette

| Token | Light | Dark | Usage |
|-------|-------|------|-------|
| `primary` | `#1B6CA8` | `#3B9EE0` | Brand accent, CTAs, active states |
| `background` | `#F4F7FB` | `#0F1923` | Screen backgrounds |
| `surface` | `#FFFFFF` | `#1A2535` | Cards, panels |
| `foreground` | `#0D1B2A` | `#E8F0F9` | Primary text |
| `muted` | `#6B7E93` | `#8A9DB5` | Secondary text, labels |
| `border` | `#D6E3F0` | `#2A3D54` | Dividers, card borders |
| `success` | `#16A34A` | `#4ADE80` | Safe / green status |
| `warning` | `#D97706` | `#FBBF24` | Medium risk / yellow |
| `error` | `#DC2626` | `#F87171` | High risk / red |
| `info` | `#0EA5E9` | `#38BDF8` | Informational badges |

---

## Screen List

1. **Dashboard** — Main overview of all silobags
2. **Silobag Detail** — Deep-dive per silobag: charts, AI prediction, recommendations
3. **Alerts** — Prioritized alert list with filters
4. **Map** — Geographic view of silobag locations and status
5. **Settings** — Thresholds, sampling interval, notifications

---

## Primary Content & Functionality

### 1. Dashboard (`/`)
- Header: App logo + "SMISIA" title + notification bell badge
- Summary KPI row: Total silobags | Active alerts | Average temperature
- Silobag list (FlatList): each card shows:
  - Name + location
  - Status badge (🟢 Seguro / 🟡 Precaución / 🔴 Peligro)
  - Last reading: temp, humidity, CO2
  - Last update timestamp
- Pull-to-refresh to simulate new sensor data
- Tap card → navigate to Silobag Detail

### 2. Silobag Detail (`/silobag/[id]`)
- Header: back arrow + silobag name + status badge
- Current readings row: Temp | Humidity | CO2 (large KPI cards)
- AI Risk Prediction card: risk level + confidence % + explanation text
- Recommendations list: auto-generated based on readings
- Time-series chart: line chart with 24h/7d/30d toggle (temperature, humidity, CO2)
- History table: last 10 readings

### 3. Alerts (`/alerts`)
- Filter tabs: Todos | Alto | Medio | Bajo
- Alert list (FlatList): each item shows:
  - Priority badge (colored)
  - Silobag name
  - Alert message (e.g., "Posible deterioro por alta humedad")
  - Timestamp
- Empty state when no alerts

### 4. Map (`/map`)
- Interactive map (expo-maps or react-native-maps)
- Markers per silobag: color-coded by status
- Tap marker → bottom sheet with silobag summary
- Legend overlay

### 5. Settings (`/settings`)
- Section: Umbrales de alerta
  - Temperatura máxima (°C)
  - Humedad máxima (%)
  - CO2 máximo (ppm)
- Section: Muestreo
  - Intervalo de datos (1h / 2h / 4h)
- Section: Notificaciones
  - Toggle alertas push
  - Toggle alertas de alto riesgo
- Section: Simulación
  - Botón "Generar datos de prueba"
  - Botón "Resetear datos"

---

## Key User Flows

### Flow 1: Check silobag status
Home tab → Silobag card → Detail screen → View charts and AI prediction

### Flow 2: Respond to alert
Alerts tab → Tap alert → Navigate to silobag detail → Read recommendation

### Flow 3: Locate silobag on map
Map tab → Tap marker → View summary → Tap "Ver detalle" → Detail screen

### Flow 4: Adjust alert thresholds
Settings tab → Umbrales section → Edit value → Save

---

## Navigation Structure

Bottom Tab Bar (5 tabs):
- 🏠 Dashboard (house.fill)
- 🔔 Alertas (bell.fill)
- 🗺️ Mapa (map.fill)
- ⚙️ Ajustes (gearshape.fill)

Stack navigator inside Dashboard tab for Silobag Detail screen.

---

## Visual Style

- **Layout:** Cards with rounded corners (12px), subtle shadows, generous padding
- **Typography:** System font, bold headers, regular body, small muted labels
- **Icons:** SF Symbols (iOS) / Material Icons (Android)
- **Charts:** Line charts with colored fills, grid lines, axis labels
- **Status colors:** Always use semantic tokens (success/warning/error), never hardcoded hex
- **Spacing:** 16px base unit, 8px half-unit for tight spaces
- **Inspiration:** Industrial IoT dashboards + modern agricultural apps
