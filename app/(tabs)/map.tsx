// SMISIA Map Screen
// Geographic view of silobag locations with status markers

import React, { useState, useCallback } from "react";
import {
  View,
  Text,
  Pressable,
  StyleSheet,
  Platform,
  ScrollView,
  Dimensions,
} from "react-native";
import { useRouter } from "expo-router";
import * as Haptics from "expo-haptics";

import { ScreenContainer } from "@/components/screen-container";
import { RiskBadge } from "@/components/smisia-ui";
import { useColors } from "@/hooks/use-colors";
import { useSmisia } from "@/lib/smisia-context";
import { Silobag, RiskLevel } from "@/lib/models";

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get("window");

// ─── Map Marker Colors ────────────────────────────────────────────────────────

const MARKER_COLORS: Record<RiskLevel, string> = {
  safe:   "#16A34A",
  low:    "#D97706",
  medium: "#F59E0B",
  high:   "#DC2626",
};

// ─── Simple SVG-based Map Fallback (works on Web + Native) ───────────────────
// We use a visual grid map since react-native-maps requires native builds

function SimpleMap({
  silobags,
  selectedId,
  onMarkerPress,
}: {
  silobags: Silobag[];
  selectedId: string | null;
  onMarkerPress: (id: string) => void;
}) {
  const colors = useColors();

  // Normalize lat/lng to canvas coordinates
  const lats = silobags.map((s) => s.latitude);
  const lngs = silobags.map((s) => s.longitude);
  const minLat = Math.min(...lats) - 0.003;
  const maxLat = Math.max(...lats) + 0.003;
  const minLng = Math.min(...lngs) - 0.003;
  const maxLng = Math.max(...lngs) + 0.003;

  const mapW = SCREEN_WIDTH;
  const mapH = SCREEN_HEIGHT * 0.48;

  const toX = (lng: number) => ((lng - minLng) / (maxLng - minLng)) * (mapW - 60) + 30;
  const toY = (lat: number) => ((maxLat - lat) / (maxLat - minLat)) * (mapH - 80) + 40;

  return (
    <View style={[styles.mapContainer, { backgroundColor: colors.surface, width: mapW, height: mapH }]}>
      {/* Grid background */}
      <View style={[styles.mapGrid, { borderColor: colors.border }]}>
        {/* Field zones */}
        <View style={[styles.fieldZone, { backgroundColor: colors.background, borderColor: colors.border, top: 30, left: 20, width: mapW * 0.4, height: mapH * 0.35 }]}>
          <Text style={[styles.fieldLabel, { color: colors.muted }]}>Lote Norte</Text>
        </View>
        <View style={[styles.fieldZone, { backgroundColor: colors.background, borderColor: colors.border, bottom: 30, left: 20, width: mapW * 0.4, height: mapH * 0.35 }]}>
          <Text style={[styles.fieldLabel, { color: colors.muted }]}>Lote Sur</Text>
        </View>
        <View style={[styles.fieldZone, { backgroundColor: colors.background, borderColor: colors.border, top: mapH * 0.25, right: 20, width: mapW * 0.3, height: mapH * 0.4 }]}>
          <Text style={[styles.fieldLabel, { color: colors.muted }]}>Lote Este</Text>
        </View>

        {/* Silobag markers */}
        {silobags.map((sb) => {
          const x = toX(sb.longitude);
          const y = toY(sb.latitude);
          const markerColor = MARKER_COLORS[sb.riskLevel];
          const isSelected = sb.id === selectedId;

          return (
            <Pressable
              key={sb.id}
              onPress={() => onMarkerPress(sb.id)}
              style={({ pressed }) => [
                styles.marker,
                {
                  left: x - 18,
                  top: y - 18,
                  backgroundColor: markerColor,
                  borderColor: isSelected ? "#fff" : markerColor,
                  borderWidth: isSelected ? 3 : 2,
                  transform: [{ scale: isSelected ? 1.2 : 1 }],
                  shadowColor: markerColor,
                  shadowOpacity: isSelected ? 0.6 : 0.3,
                  shadowRadius: isSelected ? 8 : 4,
                  elevation: isSelected ? 8 : 4,
                },
                pressed && { opacity: 0.8 },
              ]}
            >
              <Text style={styles.markerText}>
                {sb.riskLevel === "high" ? "⚠" : sb.riskLevel === "medium" ? "!" : "✓"}
              </Text>
            </Pressable>
          );
        })}
      </View>

      {/* Compass */}
      <View style={[styles.compass, { backgroundColor: colors.surface, borderColor: colors.border }]}>
        <Text style={[styles.compassN, { color: colors.primary }]}>N</Text>
        <Text style={[styles.compassArrow, { color: colors.primary }]}>↑</Text>
      </View>

      {/* Scale */}
      <View style={[styles.scale, { backgroundColor: colors.surface, borderColor: colors.border }]}>
        <Text style={[styles.scaleText, { color: colors.muted }]}>~500m</Text>
      </View>
    </View>
  );
}

// ─── Silobag Bottom Sheet ─────────────────────────────────────────────────────

function SilobagSheet({ silobag, onClose, onDetail }: {
  silobag: Silobag;
  onClose: () => void;
  onDetail: () => void;
}) {
  const colors = useColors();
  const r = silobag.lastReading;

  return (
    <View style={[styles.sheet, { backgroundColor: colors.surface, borderColor: colors.border }]}>
      <View style={styles.sheetHandle}>
        <View style={[styles.handleBar, { backgroundColor: colors.border }]} />
      </View>

      <View style={styles.sheetHeader}>
        <View style={styles.sheetTitleRow}>
          <Text style={[styles.sheetTitle, { color: colors.foreground }]}>{silobag.name}</Text>
          <RiskBadge level={silobag.riskLevel} size="md" />
        </View>
        <Text style={[styles.sheetLocation, { color: colors.muted }]}>
          📍 {silobag.location} · {silobag.grainType}
        </Text>
      </View>

      {r && (
        <View style={styles.sheetReadings}>
          <SheetReading icon="🌡️" value={`${r.temperature.toFixed(1)}°C`} label="Temp" colors={colors} />
          <SheetReading icon="💧" value={`${r.humidity.toFixed(1)}%`} label="Hum" colors={colors} />
          <SheetReading icon="🌬️" value={`${r.co2.toFixed(0)} ppm`} label="CO₂" colors={colors} />
        </View>
      )}

      {silobag.prediction && (
        <Text style={[styles.sheetPrediction, { color: colors.muted }]} numberOfLines={2}>
          🤖 {silobag.prediction.explanation}
        </Text>
      )}

      <View style={styles.sheetActions}>
        <Pressable
          onPress={onClose}
          style={({ pressed }) => [
            styles.sheetBtnSecondary,
            { borderColor: colors.border },
            pressed && { opacity: 0.7 },
          ]}
        >
          <Text style={[styles.sheetBtnSecondaryText, { color: colors.muted }]}>Cerrar</Text>
        </Pressable>
        <Pressable
          onPress={onDetail}
          style={({ pressed }) => [
            styles.sheetBtnPrimary,
            { backgroundColor: colors.primary },
            pressed && { opacity: 0.8 },
          ]}
        >
          <Text style={styles.sheetBtnPrimaryText}>Ver Detalle →</Text>
        </Pressable>
      </View>
    </View>
  );
}

function SheetReading({ icon, value, label, colors }: { icon: string; value: string; label: string; colors: any }) {
  return (
    <View style={[styles.sheetReadingItem, { backgroundColor: colors.background }]}>
      <Text style={styles.sheetReadingIcon}>{icon}</Text>
      <Text style={[styles.sheetReadingValue, { color: colors.foreground }]}>{value}</Text>
      <Text style={[styles.sheetReadingLabel, { color: colors.muted }]}>{label}</Text>
    </View>
  );
}

// ─── Map Screen ───────────────────────────────────────────────────────────────

export default function MapScreen() {
  const colors = useColors();
  const router = useRouter();
  const { state } = useSmisia();
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const selectedSilobag = selectedId ? state.silobags.find((s) => s.id === selectedId) : null;

  const handleMarkerPress = useCallback((id: string) => {
    if (Platform.OS !== "web") {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    }
    setSelectedId((prev) => (prev === id ? null : id));
  }, []);

  const handleViewDetail = useCallback(() => {
    if (selectedId) {
      router.push(`/silobag/${selectedId}` as any);
    }
  }, [selectedId, router]);

  // Risk summary counts
  const counts = {
    high:   state.silobags.filter((s) => s.riskLevel === "high").length,
    medium: state.silobags.filter((s) => s.riskLevel === "medium").length,
    low:    state.silobags.filter((s) => s.riskLevel === "low").length,
    safe:   state.silobags.filter((s) => s.riskLevel === "safe").length,
  };

  return (
    <ScreenContainer containerClassName="bg-background" edges={["top", "left", "right"]}>
      {/* Header */}
      <View style={[styles.header, { backgroundColor: colors.surface, borderBottomColor: colors.border }]}>
        <Text style={[styles.headerTitle, { color: colors.foreground }]}>Mapa de Silobolsas</Text>
        <Text style={[styles.headerSub, { color: colors.muted }]}>
          {state.silobags.length} ubicaciones
        </Text>
      </View>

      {/* Legend */}
      <View style={[styles.legend, { backgroundColor: colors.surface, borderBottomColor: colors.border }]}>
        {([["safe", "Seguro", "#16A34A"], ["low", "Bajo", "#D97706"], ["medium", "Precaución", "#F59E0B"], ["high", "Peligro", "#DC2626"]] as const).map(([level, label, color]) => (
          <View key={level} style={styles.legendItem}>
            <View style={[styles.legendDot, { backgroundColor: color }]} />
            <Text style={[styles.legendText, { color: colors.muted }]}>
              {label} ({counts[level]})
            </Text>
          </View>
        ))}
      </View>

      {/* Map */}
      <SimpleMap
        silobags={state.silobags}
        selectedId={selectedId}
        onMarkerPress={handleMarkerPress}
      />

      {/* Selected silobag sheet */}
      {selectedSilobag && (
        <SilobagSheet
          silobag={selectedSilobag}
          onClose={() => setSelectedId(null)}
          onDetail={handleViewDetail}
        />
      )}

      {/* Tap hint */}
      {!selectedSilobag && (
        <View style={[styles.hint, { backgroundColor: colors.surface, borderColor: colors.border }]}>
          <Text style={[styles.hintText, { color: colors.muted }]}>
            Toca un marcador para ver detalles del silobolsa
          </Text>
        </View>
      )}
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
  legend: {
    flexDirection: "row",
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderBottomWidth: 1,
    gap: 12,
    flexWrap: "wrap",
  },
  legendItem: { flexDirection: "row", alignItems: "center", gap: 5 },
  legendDot: { width: 10, height: 10, borderRadius: 5 },
  legendText: { fontSize: 11, fontWeight: "500", lineHeight: 16 },
  mapContainer: {
    position: "relative",
    overflow: "hidden",
  },
  mapGrid: {
    flex: 1,
    position: "relative",
  },
  fieldZone: {
    position: "absolute",
    borderRadius: 8,
    borderWidth: 1,
    borderStyle: "dashed",
    padding: 6,
  },
  fieldLabel: {
    fontSize: 10,
    fontWeight: "600",
    lineHeight: 14,
  },
  marker: {
    position: "absolute",
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: "center",
    justifyContent: "center",
    shadowOffset: { width: 0, height: 2 },
  },
  markerText: {
    fontSize: 16,
    lineHeight: 20,
    color: "#fff",
    fontWeight: "800",
  },
  compass: {
    position: "absolute",
    top: 12,
    right: 12,
    width: 36,
    height: 36,
    borderRadius: 18,
    borderWidth: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  compassN: { fontSize: 9, fontWeight: "800", lineHeight: 12 },
  compassArrow: { fontSize: 12, lineHeight: 14 },
  scale: {
    position: "absolute",
    bottom: 12,
    right: 12,
    borderRadius: 6,
    borderWidth: 1,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  scaleText: { fontSize: 10, lineHeight: 14 },
  sheet: {
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    borderWidth: 1,
    borderBottomWidth: 0,
    padding: 16,
    gap: 12,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: -2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 8,
  },
  sheetHandle: { alignItems: "center", marginBottom: 4 },
  handleBar: { width: 36, height: 4, borderRadius: 2 },
  sheetHeader: { gap: 4 },
  sheetTitleRow: { flexDirection: "row", alignItems: "center", justifyContent: "space-between" },
  sheetTitle: { fontSize: 17, fontWeight: "700", lineHeight: 22 },
  sheetLocation: { fontSize: 12, lineHeight: 16 },
  sheetReadings: { flexDirection: "row", gap: 10 },
  sheetReadingItem: {
    flex: 1,
    borderRadius: 10,
    padding: 10,
    alignItems: "center",
    gap: 2,
  },
  sheetReadingIcon: { fontSize: 18, lineHeight: 24 },
  sheetReadingValue: { fontSize: 14, fontWeight: "700", lineHeight: 20 },
  sheetReadingLabel: { fontSize: 10, lineHeight: 14 },
  sheetPrediction: { fontSize: 12, lineHeight: 18 },
  sheetActions: { flexDirection: "row", gap: 10 },
  sheetBtnSecondary: {
    flex: 1,
    borderRadius: 10,
    borderWidth: 1,
    paddingVertical: 10,
    alignItems: "center",
  },
  sheetBtnSecondaryText: { fontSize: 14, fontWeight: "600", lineHeight: 20 },
  sheetBtnPrimary: {
    flex: 2,
    borderRadius: 10,
    paddingVertical: 10,
    alignItems: "center",
  },
  sheetBtnPrimaryText: { color: "#fff", fontSize: 14, fontWeight: "700", lineHeight: 20 },
  hint: {
    margin: 16,
    borderRadius: 10,
    borderWidth: 1,
    paddingVertical: 10,
    alignItems: "center",
  },
  hintText: { fontSize: 12, lineHeight: 18 },
});
