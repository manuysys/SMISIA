// SMISIA Shared UI Components
// Reusable components for consistent visual language across the app

import React from "react";
import { View, Text, StyleSheet } from "react-native";
import { useColors } from "@/hooks/use-colors";
import { RiskLevel } from "@/lib/models";

// ─── Risk Badge ───────────────────────────────────────────────────────────────

const RISK_LABELS: Record<RiskLevel, string> = {
  safe:   "Seguro",
  low:    "Bajo",
  medium: "Precaución",
  high:   "Peligro",
};

const RISK_COLORS = {
  safe:   { bg: "#DCFCE7", text: "#16A34A", dark_bg: "#14532D", dark_text: "#4ADE80" },
  low:    { bg: "#FEF9C3", text: "#CA8A04", dark_bg: "#713F12", dark_text: "#FDE047" },
  medium: { bg: "#FEF3C7", text: "#D97706", dark_bg: "#78350F", dark_text: "#FBBF24" },
  high:   { bg: "#FEE2E2", text: "#DC2626", dark_bg: "#7F1D1D", dark_text: "#F87171" },
};

export function RiskBadge({ level, size = "md" }: { level: RiskLevel; size?: "sm" | "md" | "lg" }) {
  const colors = useColors();
  const isDark = colors.background === "#0F1923";
  const c = RISK_COLORS[level];
  const bg = isDark ? c.dark_bg : c.bg;
  const fg = isDark ? c.dark_text : c.text;

  const fontSize = size === "sm" ? 10 : size === "lg" ? 14 : 12;
  const paddingH = size === "sm" ? 6 : size === "lg" ? 12 : 8;
  const paddingV = size === "sm" ? 2 : size === "lg" ? 5 : 3;

  return (
    <View style={[styles.badge, { backgroundColor: bg, paddingHorizontal: paddingH, paddingVertical: paddingV }]}>
      <Text style={[styles.badgeText, { color: fg, fontSize }]}>
        {RISK_LABELS[level].toUpperCase()}
      </Text>
    </View>
  );
}

// ─── Sensor Value Card ────────────────────────────────────────────────────────

interface SensorCardProps {
  label: string;
  value: string | number;
  unit: string;
  icon: string;
  status?: "normal" | "warning" | "danger";
}

export function SensorCard({ label, value, unit, icon, status = "normal" }: SensorCardProps) {
  const colors = useColors();
  const statusColor = status === "danger" ? colors.error : status === "warning" ? colors.warning : colors.success;

  return (
    <View style={[styles.sensorCard, { backgroundColor: colors.surface, borderColor: colors.border }]}>
      <Text style={styles.sensorIcon}>{icon}</Text>
      <Text style={[styles.sensorValue, { color: statusColor }]}>
        {typeof value === "number" ? value.toFixed(1) : value}
        <Text style={[styles.sensorUnit, { color: colors.muted }]}> {unit}</Text>
      </Text>
      <Text style={[styles.sensorLabel, { color: colors.muted }]}>{label}</Text>
    </View>
  );
}

// ─── Section Header ───────────────────────────────────────────────────────────

export function SectionHeader({ title, subtitle }: { title: string; subtitle?: string }) {
  const colors = useColors();
  return (
    <View style={styles.sectionHeader}>
      <Text style={[styles.sectionTitle, { color: colors.foreground }]}>{title}</Text>
      {subtitle && <Text style={[styles.sectionSubtitle, { color: colors.muted }]}>{subtitle}</Text>}
    </View>
  );
}

// ─── KPI Card ─────────────────────────────────────────────────────────────────

export function KpiCard({ label, value, color }: { label: string; value: string | number; color?: string }) {
  const colors = useColors();
  return (
    <View style={[styles.kpiCard, { backgroundColor: colors.surface, borderColor: colors.border }]}>
      <Text style={[styles.kpiValue, { color: color ?? colors.primary }]}>{value}</Text>
      <Text style={[styles.kpiLabel, { color: colors.muted }]}>{label}</Text>
    </View>
  );
}

// ─── Divider ─────────────────────────────────────────────────────────────────

export function Divider() {
  const colors = useColors();
  return <View style={[styles.divider, { backgroundColor: colors.border }]} />;
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  badge: {
    borderRadius: 20,
    alignSelf: "flex-start",
  },
  badgeText: {
    fontWeight: "700",
    letterSpacing: 0.5,
  },
  sensorCard: {
    flex: 1,
    borderRadius: 12,
    padding: 12,
    alignItems: "center",
    borderWidth: 1,
    minWidth: 90,
  },
  sensorIcon: {
    fontSize: 22,
    marginBottom: 4,
  },
  sensorValue: {
    fontSize: 20,
    fontWeight: "700",
    lineHeight: 26,
  },
  sensorUnit: {
    fontSize: 12,
    fontWeight: "400",
  },
  sensorLabel: {
    fontSize: 11,
    marginTop: 2,
    fontWeight: "500",
  },
  sectionHeader: {
    marginBottom: 12,
    marginTop: 8,
  },
  sectionTitle: {
    fontSize: 17,
    fontWeight: "700",
    lineHeight: 22,
  },
  sectionSubtitle: {
    fontSize: 13,
    marginTop: 2,
    lineHeight: 18,
  },
  kpiCard: {
    flex: 1,
    borderRadius: 12,
    padding: 14,
    alignItems: "center",
    borderWidth: 1,
  },
  kpiValue: {
    fontSize: 24,
    fontWeight: "800",
    lineHeight: 30,
  },
  kpiLabel: {
    fontSize: 11,
    fontWeight: "500",
    marginTop: 2,
    textAlign: "center",
  },
  divider: {
    height: 1,
    marginVertical: 12,
  },
});
