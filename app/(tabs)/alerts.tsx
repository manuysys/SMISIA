// SMISIA Alerts Screen
// Prioritized list of all system alerts with filter tabs

import React, { useState, useCallback } from "react";
import {
  View,
  Text,
  FlatList,
  Pressable,
  StyleSheet,
} from "react-native";
import { useRouter } from "expo-router";
import * as Haptics from "expo-haptics";
import { Platform } from "react-native";

import { ScreenContainer } from "@/components/screen-container";
import { IconSymbol } from "@/components/ui/icon-symbol";
import { useColors } from "@/hooks/use-colors";
import { useSmisia } from "@/lib/smisia-context";
import { Alert, RiskLevel } from "@/lib/models";

type FilterLevel = "all" | "high" | "medium" | "low";

const FILTER_TABS: { key: FilterLevel; label: string }[] = [
  { key: "all",    label: "Todos" },
  { key: "high",   label: "Alto" },
  { key: "medium", label: "Medio" },
  { key: "low",    label: "Bajo" },
];

const RISK_ICONS: Record<Exclude<RiskLevel, "safe">, string> = {
  high:   "🔴",
  medium: "🟡",
  low:    "🟢",
};

const RISK_LABELS: Record<Exclude<RiskLevel, "safe">, string> = {
  high:   "ALTO",
  medium: "MEDIO",
  low:    "BAJO",
};

function formatDateTime(date: Date): string {
  const d = new Date(date);
  return d.toLocaleString("es-AR", {
    day: "2-digit", month: "2-digit",
    hour: "2-digit", minute: "2-digit",
  });
}

// ─── Alert Card ───────────────────────────────────────────────────────────────

function AlertCard({ alert, onPress, onAcknowledge }: {
  alert: Alert;
  onPress: () => void;
  onAcknowledge: () => void;
}) {
  const colors = useColors();

  const borderColor =
    alert.riskLevel === "high"   ? colors.error   :
    alert.riskLevel === "medium" ? colors.warning  :
    colors.success;

  const bgColor = alert.acknowledged
    ? colors.surface
    : alert.riskLevel === "high"   ? (colors.background === "#F4F7FB" ? "#FFF5F5" : "#2D1515")
    : alert.riskLevel === "medium" ? (colors.background === "#F4F7FB" ? "#FFFBEB" : "#2D2210")
    : colors.surface;

  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [
        styles.alertCard,
        { backgroundColor: bgColor, borderColor, borderLeftWidth: 4 },
        pressed && { opacity: 0.75 },
        alert.acknowledged && { opacity: 0.6 },
      ]}
    >
      <View style={styles.alertHeader}>
        <View style={styles.alertTitleRow}>
          <Text style={styles.alertIcon}>{RISK_ICONS[alert.riskLevel]}</Text>
          <View style={styles.alertTitleBlock}>
            <Text style={[styles.alertSilobag, { color: colors.foreground }]}>
              {alert.silobagName}
            </Text>
            <Text style={[styles.alertType, { color: colors.muted }]}>
              {alert.type === "temperature" ? "Temperatura" :
               alert.type === "humidity"    ? "Humedad"     :
               alert.type === "co2"         ? "CO₂"         : "Combinado"}
            </Text>
          </View>
          <View style={[styles.riskBadge, { borderColor }]}>
            <Text style={[styles.riskBadgeText, { color: borderColor }]}>
              {RISK_LABELS[alert.riskLevel]}
            </Text>
          </View>
        </View>
      </View>

      <Text style={[styles.alertMessage, { color: colors.foreground }]}>
        {alert.message}
      </Text>

      <View style={styles.alertFooter}>
        <Text style={[styles.alertTime, { color: colors.muted }]}>
          🕐 {formatDateTime(alert.timestamp)}
        </Text>
        {!alert.acknowledged ? (
          <Pressable
            onPress={onAcknowledge}
            style={({ pressed }) => [
              styles.ackBtn,
              { backgroundColor: colors.primary },
              pressed && { opacity: 0.75 },
            ]}
          >
            <Text style={styles.ackBtnText}>Confirmar</Text>
          </Pressable>
        ) : (
          <View style={[styles.ackDone, { backgroundColor: colors.border }]}>
            <Text style={[styles.ackDoneText, { color: colors.muted }]}>✓ Confirmado</Text>
          </View>
        )}
      </View>
    </Pressable>
  );
}

// ─── Alerts Screen ────────────────────────────────────────────────────────────

export default function AlertsScreen() {
  const colors = useColors();
  const router = useRouter();
  const { state, acknowledgeAlert } = useSmisia();
  const [filter, setFilter] = useState<FilterLevel>("all");

  const filtered = state.alerts.filter((a) =>
    filter === "all" ? true : a.riskLevel === filter
  );

  const handleAlertPress = useCallback((alert: Alert) => {
    if (Platform.OS !== "web") {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    }
    router.push(`/silobag/${alert.silobagId}` as any);
  }, [router]);

  const handleAcknowledge = useCallback((alertId: string) => {
    if (Platform.OS !== "web") {
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    }
    acknowledgeAlert(alertId);
  }, [acknowledgeAlert]);

  const highCount   = state.alerts.filter((a) => a.riskLevel === "high"   && !a.acknowledged).length;
  const mediumCount = state.alerts.filter((a) => a.riskLevel === "medium" && !a.acknowledged).length;

  return (
    <ScreenContainer containerClassName="bg-background">
      {/* Header */}
      <View style={[styles.header, { backgroundColor: colors.surface, borderBottomColor: colors.border }]}>
        <Text style={[styles.headerTitle, { color: colors.foreground }]}>Alertas</Text>
        <Text style={[styles.headerSub, { color: colors.muted }]}>
          {state.alerts.filter((a) => !a.acknowledged).length} sin confirmar
        </Text>
      </View>

      {/* Summary row */}
      {(highCount > 0 || mediumCount > 0) && (
        <View style={[styles.summaryRow, { backgroundColor: colors.surface, borderBottomColor: colors.border }]}>
          {highCount > 0 && (
            <View style={[styles.summaryBadge, { backgroundColor: "#FEE2E2" }]}>
              <Text style={[styles.summaryBadgeText, { color: colors.error }]}>
                🔴 {highCount} alto{highCount > 1 ? "s" : ""}
              </Text>
            </View>
          )}
          {mediumCount > 0 && (
            <View style={[styles.summaryBadge, { backgroundColor: "#FEF3C7" }]}>
              <Text style={[styles.summaryBadgeText, { color: colors.warning }]}>
                🟡 {mediumCount} medio{mediumCount > 1 ? "s" : ""}
              </Text>
            </View>
          )}
        </View>
      )}

      {/* Filter tabs */}
      <View style={[styles.filterRow, { backgroundColor: colors.surface, borderBottomColor: colors.border }]}>
        {FILTER_TABS.map((tab) => {
          const count = tab.key === "all"
            ? state.alerts.length
            : state.alerts.filter((a) => a.riskLevel === tab.key).length;
          const isActive = filter === tab.key;
          return (
            <Pressable
              key={tab.key}
              onPress={() => setFilter(tab.key)}
              style={({ pressed }) => [
                styles.filterTab,
                isActive && { borderBottomColor: colors.primary, borderBottomWidth: 2 },
                pressed && { opacity: 0.75 },
              ]}
            >
              <Text style={[
                styles.filterTabText,
                { color: isActive ? colors.primary : colors.muted },
              ]}>
                {tab.label}
              </Text>
              {count > 0 && (
                <View style={[styles.filterCount, { backgroundColor: isActive ? colors.primary : colors.border }]}>
                  <Text style={[styles.filterCountText, { color: isActive ? "#fff" : colors.muted }]}>
                    {count}
                  </Text>
                </View>
              )}
            </Pressable>
          );
        })}
      </View>

      <FlatList
        data={filtered}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.listContent}
        renderItem={({ item }) => (
          <AlertCard
            alert={item}
            onPress={() => handleAlertPress(item)}
            onAcknowledge={() => handleAcknowledge(item.id)}
          />
        )}
        ItemSeparatorComponent={() => <View style={{ height: 10 }} />}
        ListEmptyComponent={
          <View style={styles.empty}>
            <Text style={styles.emptyIcon}>✅</Text>
            <Text style={[styles.emptyTitle, { color: colors.foreground }]}>
              Sin alertas activas
            </Text>
            <Text style={[styles.emptyText, { color: colors.muted }]}>
              Todos los silobolsas están en condiciones normales.
            </Text>
          </View>
        }
      />
    </ScreenContainer>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  header: {
    paddingHorizontal: 16,
    paddingVertical: 14,
    borderBottomWidth: 1,
  },
  headerTitle: { fontSize: 22, fontWeight: "800", lineHeight: 28 },
  headerSub: { fontSize: 12, fontWeight: "500", marginTop: 2, lineHeight: 16 },
  summaryRow: {
    flexDirection: "row",
    gap: 8,
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderBottomWidth: 1,
  },
  summaryBadge: {
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  summaryBadgeText: { fontSize: 12, fontWeight: "700", lineHeight: 18 },
  filterRow: {
    flexDirection: "row",
    borderBottomWidth: 1,
  },
  filterTab: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 12,
    gap: 5,
    borderBottomWidth: 2,
    borderBottomColor: "transparent",
  },
  filterTabText: { fontSize: 13, fontWeight: "600", lineHeight: 18 },
  filterCount: {
    borderRadius: 8,
    minWidth: 18,
    height: 18,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 4,
  },
  filterCountText: { fontSize: 10, fontWeight: "700", lineHeight: 14 },
  listContent: { padding: 16, paddingBottom: 24 },
  alertCard: {
    borderRadius: 14,
    padding: 14,
    gap: 10,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06,
    shadowRadius: 4,
    elevation: 2,
  },
  alertHeader: {},
  alertTitleRow: { flexDirection: "row", alignItems: "center", gap: 10 },
  alertIcon: { fontSize: 20, lineHeight: 26 },
  alertTitleBlock: { flex: 1 },
  alertSilobag: { fontSize: 14, fontWeight: "700", lineHeight: 20 },
  alertType: { fontSize: 11, lineHeight: 16 },
  riskBadge: {
    borderRadius: 6,
    paddingHorizontal: 7,
    paddingVertical: 3,
    borderWidth: 1,
  },
  riskBadgeText: { fontSize: 10, fontWeight: "800", lineHeight: 14, letterSpacing: 0.5 },
  alertMessage: { fontSize: 13, lineHeight: 20 },
  alertFooter: { flexDirection: "row", alignItems: "center", justifyContent: "space-between" },
  alertTime: { fontSize: 11, lineHeight: 16 },
  ackBtn: {
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 5,
  },
  ackBtnText: { color: "#fff", fontSize: 12, fontWeight: "700", lineHeight: 16 },
  ackDone: {
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 5,
  },
  ackDoneText: { fontSize: 12, fontWeight: "500", lineHeight: 16 },
  empty: { alignItems: "center", paddingVertical: 60, gap: 8 },
  emptyIcon: { fontSize: 48, lineHeight: 56 },
  emptyTitle: { fontSize: 17, fontWeight: "700", lineHeight: 22 },
  emptyText: { fontSize: 13, textAlign: "center", lineHeight: 20, maxWidth: 260 },
});
