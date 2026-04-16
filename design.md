# SMISIA Web Dashboard — Design Document

## Overview

SMISIA web is a **responsive dashboard** for agricultural IoT monitoring of silobags. Built with **Next.js + React + Tailwind CSS**, it provides real-time insights into grain storage conditions with AI-powered risk predictions.

**Target Users:** Farm managers, agronomists, storage facility operators  
**Primary Devices:** Desktop (1920×1080), Tablet (768×1024), Mobile (375×667)

---

## Screen List

### 1. **Dashboard (Home)**
- **Purpose:** Real-time overview of all silobags and system health
- **Content:**
  - Header: Logo, user profile, notifications bell
  - KPI Cards: Total silobags, active alerts, avg temperature, critical readings
  - Silobag Grid: Cards showing each silobag with status badge, current readings, last update
  - Alert Summary: Recent alerts with priority colors
  - Quick Stats: Risk distribution pie chart, temperature trend sparkline
- **Functionality:**
  - Click silobag card → navigate to detail page
  - Click alert → navigate to alerts page
  - Pull-to-refresh (mobile) / Refresh button (desktop)
  - Filter by risk level (All/Safe/Low/Medium/High)

### 2. **Silobag Detail Page**
- **Purpose:** In-depth analysis of a single silobag
- **Content:**
  - Header: Silobag name, location, grain type, risk badge
  - Current Readings: Large cards for temp, humidity, CO2 with trend arrows
  - AI Prediction Card: Risk level, confidence %, explanation, recommendations
  - Charts Section:
    - Line chart: Temperature (24h/7d/30d toggle)
    - Line chart: Humidity (24h/7d/30d toggle)
    - Line chart: CO2 (24h/7d/30d toggle)
  - Alert History: Table of recent alerts with timestamps and actions
  - Metadata: Last reading time, sensor ID, battery level
- **Functionality:**
  - Time range toggle (24h/7d/30d)
  - Export chart as image
  - Acknowledge/dismiss alerts
  - Back button to dashboard

### 3. **Alerts Page**
- **Purpose:** Centralized alert management
- **Content:**
  - Filter tabs: All / High / Medium / Low / Acknowledged
  - Alert List: Cards with silobag name, alert type, time, action buttons
  - Alert Details: Click to expand and see full context
  - Bulk Actions: Select multiple, mark as read, acknowledge all
- **Functionality:**
  - Filter by priority
  - Sort by time/severity
  - Acknowledge individual alerts
  - Navigate to silobag detail from alert

### 4. **Map Page**
- **Purpose:** Geographic view of silobag locations
- **Content:**
  - Interactive map (Leaflet/Mapbox)
  - Markers colored by risk level (Green/Yellow/Orange/Red)
  - Marker popup: Silobag name, current readings, risk level
  - Legend: Risk color codes with counts
  - Sidebar: Silobag list with status
- **Functionality:**
  - Click marker → silobag detail
  - Click list item → center map on location
  - Zoom/pan controls
  - Filter by risk level

### 5. **Settings Page**
- **Purpose:** System configuration
- **Content:**
  - Alert Thresholds: Max temp, max humidity, max CO2 (editable inputs)
  - Sampling Interval: Dropdown (1h/2h/4h)
  - Notifications: Toggle push notifications, high-risk-only filter
  - User Preferences: Dark mode toggle, language, timezone
  - IoT Simulation: Generate test data button, reset data button
  - About: System version, API status, database info
- **Functionality:**
  - Save threshold changes
  - Generate mock sensor data
  - Reset all data
  - Toggle dark/light mode

---

## Primary Content & Functionality

### Data Models
- **Silobag:** id, name, location, grain_type, latitude, longitude, is_active, risk_level, last_reading, prediction
- **SensorReading:** id, silobag_id, timestamp, temperature, humidity, co2
- **Alert:** id, silobag_id, risk_level, triggered_rules, created_at, acknowledged_at
- **AppSettings:** max_temperature, max_humidity, max_co2, sampling_interval_hours, push_notifications_enabled

### Key User Flows

**Flow 1: Monitor Silobag Status**
1. User opens Dashboard
2. Sees KPI cards and silobag grid
3. Identifies silobag with HIGH risk (red badge)
4. Clicks card → navigates to Detail page
5. Sees current readings, AI prediction, and recommendations
6. Views 7-day temperature chart to understand trend
7. Acknowledges alert

**Flow 2: Respond to Alert**
1. User receives push notification: "Silobag B2 — HIGH RISK"
2. Clicks notification → navigates to Alerts page
3. Filters by "High" priority
4. Sees alert details: "Temperature critical (32°C > 28°C)"
5. Clicks "View Silobag" → Detail page
6. Reads AI recommendation: "Increase ventilation"
7. Marks alert as acknowledged

**Flow 3: Configure System**
1. User goes to Settings
2. Adjusts max temperature threshold from 28°C to 26°C
3. Clicks "Save Thresholds"
4. System re-analyzes all silobags with new rules
5. New alerts generated if readings exceed new threshold

---

## Color Palette (Agricultural/Industrial Theme)

| Role | Color | Hex | Usage |
|------|-------|-----|-------|
| **Primary** | Teal | `#0d9488` | Buttons, links, accents |
| **Success** | Green | `#16a34a` | Safe status, positive states |
| **Warning** | Amber | `#d97706` | Low/medium risk |
| **Error** | Red | `#dc2626` | High risk, critical alerts |
| **Background** | Off-white | `#f9fafb` | Page backgrounds |
| **Surface** | White | `#ffffff` | Cards, panels |
| **Foreground** | Dark Gray | `#111827` | Primary text |
| **Muted** | Gray | `#6b7280` | Secondary text |
| **Border** | Light Gray | `#e5e7eb` | Dividers, borders |

### Dark Mode
- Background: `#0f172a`
- Surface: `#1e293b`
- Foreground: `#f1f5f9`
- Muted: `#94a3b8`

---

## Layout & Responsive Design

### Desktop (1920×1080)
- Sidebar navigation (left, 240px)
- Main content area (full width)
- 3-column grid for silobag cards
- Full-size charts (line charts span 100%)

### Tablet (768×1024)
- Hamburger menu (sidebar collapses)
- 2-column grid for silobag cards
- Stacked charts

### Mobile (375×667)
- Full-width hamburger menu
- 1-column grid for silobag cards
- Stacked sections
- Touch-friendly buttons (48px min)

---

## Key Interactions

1. **Real-time Updates:** WebSocket connection for live sensor data (optional, mock for MVP)
2. **Animations:** Smooth transitions between pages, subtle hover effects on cards
3. **Accessibility:** WCAG 2.1 AA compliance, keyboard navigation, screen reader support
4. **Performance:** Server-side rendering for fast initial load, client-side hydration for interactivity

---

## Technology Stack

- **Framework:** Next.js 16 (App Router)
- **UI Library:** React 19
- **Styling:** Tailwind CSS 4
- **Charts:** Recharts (line charts)
- **Maps:** Leaflet + React-Leaflet (or Mapbox GL)
- **State Management:** React Context + hooks
- **API Client:** Fetch API / Axios
- **Testing:** Vitest + React Testing Library
- **Deployment:** Vercel / Self-hosted

---

## Success Metrics

- Dashboard loads in <2s (desktop), <3s (mobile)
- All user flows complete end-to-end without errors
- Charts render smoothly with 60+ data points
- Responsive design works on all target devices
- Dark mode toggle works without page reload
