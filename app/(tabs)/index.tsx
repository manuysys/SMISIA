// SMISIA Dashboard Screen
// Main overview: KPI summary + silobag list

import React, { useCallback } from "react";
import {
  View,
  Text,
  FlatList,
  Pressable,
  RefreshControl,
  StyleSheet,
  ActivityIndicator,
} from "react-native";
import { useRouter } from "expo-router";
import * as Haptics from "expo-haptics";
import { Platform } from "react-native";

import { ScreenContainer } from "@/components/screen-container";
import { RiskBadge, KpiCard } from "@/components/smisia-ui";
import { IconSymbol } from "@/components/ui/icon-symbol";
import { useColors } from "@/hooks/use-colors";
import { useSmisia } from "@/lib/smisia-context";
import { Silobag, RiskLevel } from "@/lib/models";

// ─── Silobag List Card ────────────────────────────────────────────────────────

function SilobagCard({ silobag, onPress }: { silobag: Silobag; onPress: () => void }) {
  const colors = useColors();
  const r = silobag.lastReading;

  const borderColor =
    silobag.riskLevel === "high"   ? colors.error   :
    silobag.riskLevel === "medium" ? colors.warning  :
    silobag.riskLevel === "low"    ? "#D97706"       :
    colors.border;

  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [
        styles.card,
        { backgroundColor: colors.surface, borderColor, borderLeftWidth: 4 },
        pressed && { opacity: 0.75 },
      ]}
    >
      <View style={styles.cardHeader}>
        <View style={styles.cardTitleRow}>
          <Text style={[styles.cardTitle, { color: colors.foreground }]}>{silobag.name}</Text>
          <RiskBadge level={silobag.riskLevel} size="sm" />
        </View>
        <Text style={[styles.cardLocation, { color: colors.muted }]}>
          📍 {silobag.location} · {silobag.grainType}
        </Text>
      </View>

      {r && (
        <View style={styles.cardReadings}>
          <ReadingPill icon="🌡️" value={`${r.temperature.toFixed(1)}°C`} />
          <ReadingPill icon="💧" value={`${r.humidity.toFixed(1)}%`} />
          <ReadingPill icon="🌬️" value={`${r.co2.toFixed(0)} ppm`} />
        </View>
      )}

      <View style={styles.cardFooter}>
        <Text style={[styles.cardTimestamp, { color: colors.muted }]}>
          Última lectura: {r ? formatTime(r.timestamp) : "—"}
        </Text>
        <IconSymbol name="chevron.right" size={16} color={colors.muted} />
      </View>
    </Pressable>
  );
}

function ReadingPill({ icon, value }: { icon: string; value: string }) {
  const colors = useColors();
  return (
    <View style={[styles.pill, { backgroundColor: colors.background }]}>
      <Text style={styles.pillIcon}>{icon}</Text>
      <Text style={[styles.pillValue, { color: colors.foreground }]}>{value}</Text>
    </View>
  );
}

function formatTime(date: Date): string {
  const d = new Date(date);
  return d.toLocaleTimeString("es-AR", { hour: "2-digit", minute: "2-digit" });
}

// ─── Dashboard Screen ─────────────────────────────────────────────────────────

export default function DashboardScreen() {
  const colors = useColors();
  const router = useRouter();
  const { state, refreshData, unacknowledgedCount } = useSmisia();
  const [refreshing, setRefreshing] = React.useState(false);

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    refreshData();
    setTimeout(() => setRefreshing(false), 800);
  }, [refreshData]);

  const handleSilobagPress = useCallback((id: string) => {
    if (Platform.OS !== "web") {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    }
    router.push(`/silobag/${id}` as any);
  }, [router]);

  if (state.isLoading) {
    return (
      <ScreenContainer>
        <View style={styles.loading}>
          <ActivityIndicator size="large" color={colors.primary} />
          <Text style={[styles.loadingText, { color: colors.muted }]}>Cargando datos...</Text>
        </View>
      </ScreenContainer>
    );
  }

  const silobags = state.silobags;
  const totalActive = silobags.filter((s) => s.isActive).length;
  const highRisk = silobags.filter((s) => s.riskLevel === "high").length;
  const avgTemp = silobags.length > 0
    ? (silobags.reduce((sum, s) => sum + (s.lastReading?.temperature ?? 0), 0) / silobags.length).toFixed(1)
    : "—";

  const riskOrder: Record<RiskLevel, number> = { high: 0, medium: 1, low: 2, safe: 3 };
  const sorted = [...silobags].sort((a, b) => riskOrder[a.riskLevel] - riskOrder[b.riskLevel]);

  return (
    <ScreenContainer containerClassName="bg-background">
      <FlatList
        data={sorted}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.listContent}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor={colors.primary}
            colors={[colors.primary]}
          />
        }
        ListHeaderComponent={
          <View>
            {/* Header */}
            <View style={[styles.header, { backgroundColor: colors.surface, borderBottomColor: colors.border }]}>
              <View>
                <Text style={[styles.headerTitle, { color: colors.primary }]}>SMISIA</Text>
                <Text style={[styles.headerSubtitle, { color: colors.muted }]}>
                  Monitoreo de Silobolsas
                </Text>
              </View>
              <View style={styles.headerRight}>
                {unacknowledgedCount > 0 && (
                  <View style={[styles.alertDot, { backgroundColor: colors.error }]}>
                    <Text style={styles.alertDotText}>{unacknowledgedCount}</Text>
                  </View>
                )}
                <IconSymbol name="antenna.radiowaves.left.and.right" size={22} color={colors.primary} />
              </View>
            </View>

            {/* KPI Row */}
            <View style={styles.kpiRow}>
              <KpiCard label="Silobolsas" value={totalActive} />
              <KpiCard label="Alertas" value={unacknowledgedCount} color={unacknowledgedCount > 0 ? colors.error : colors.success} />
              <KpiCard label="Temp. Prom." value={`${avgTemp}°C`} color={highRisk > 0 ? colors.warning : colors.success} />
            </View>

            {/* Section title */}
            <View style={styles.sectionRow}>
              <Text style={[styles.sectionTitle, { color: colors.foreground }]}>
                Silobolsas Activas
              </Text>
              <Text style={[styles.sectionCount, { color: colors.muted }]}>
                {totalActive} total
              </Text>
            </View>
          </View>
        }
        renderItem={({ item }) => (
          <SilobagCard
            silobag={item}
            onPress={() => handleSilobagPress(item.id)}
          />
        )}
        ItemSeparatorComponent={() => <View style={{ height: 10 }} />}
        ListEmptyComponent={
          <View style={styles.empty}>
            <Text style={[styles.emptyText, { color: colors.muted }]}>
              No hay silobolsas registradas.
            </Text>
          </View>
        }
      />
    </ScreenContainer>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  loading: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: 12,
  },
  loadingText: {
    fontSize: 14,
    lineHeight: 20,
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 16,
    paddingVertical: 14,
    borderBottomWidth: 1,
    marginBottom: 16,
  },
  headerTitle: {
    fontSize: 22,
    fontWeight: "800",
    letterSpacing: 1,
    lineHeight: 28,
  },
  headerSubtitle: {
    fontSize: 12,
    fontWeight: "500",
    lineHeight: 16,
  },
  headerRight: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  alertDot: {
    borderRadius: 10,
    minWidth: 20,
    height: 20,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 4,
  },
  alertDotText: {
    color: "#fff",
    fontSize: 11,
    fontWeight: "700",
    lineHeight: 14,
  },
  kpiRow: {
    flexDirection: "row",
    gap: 10,
    paddingHorizontal: 16,
    marginBottom: 20,
  },
  sectionRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 16,
    marginBottom: 10,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: "700",
    lineHeight: 22,
  },
  sectionCount: {
    fontSize: 13,
    lineHeight: 18,
  },
  listContent: {
    paddingHorizontal: 16,
    paddingBottom: 24,
  },
  card: {
    borderRadius: 14,
    padding: 14,
    gap: 10,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06,
    shadowRadius: 4,
    elevation: 2,
  },
  cardHeader: {
    gap: 4,
  },
  cardTitleRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  cardTitle: {
    fontSize: 15,
    fontWeight: "700",
    lineHeight: 20,
  },
  cardLocation: {
    fontSize: 12,
    lineHeight: 16,
  },
  cardReadings: {
    flexDirection: "row",
    gap: 8,
  },
  pill: {
    flexDirection: "row",
    alignItems: "center",
    borderRadius: 8,
    paddingHorizontal: 8,
    paddingVertical: 4,
    gap: 4,
  },
  pillIcon: {
    fontSize: 13,
    lineHeight: 18,
  },
  pillValue: {
    fontSize: 12,
    fontWeight: "600",
    lineHeight: 16,
  },
  cardFooter: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  cardTimestamp: {
    fontSize: 11,
    lineHeight: 14,
  },
  empty: {
    alignItems: "center",
    paddingVertical: 40,
  },
  emptyText: {
    fontSize: 14,
    lineHeight: 20,
  },
});
