// SMISIA Silobag Detail Screen
// Deep-dive view: sensor readings, AI prediction, charts, history

import React, { useState, useMemo } from "react";
import {
  View,
  Text,
  ScrollView,
  Pressable,
  StyleSheet,
  Dimensions,
} from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import Svg, { Polyline, Line, Text as SvgText, Defs, LinearGradient, Stop, Rect } from "react-native-svg";

import { ScreenContainer } from "@/components/screen-container";
import { RiskBadge, SectionHeader, Divider } from "@/components/smisia-ui";
import { IconSymbol } from "@/components/ui/icon-symbol";
import { useColors } from "@/hooks/use-colors";
import { useSmisia } from "@/lib/smisia-context";
import { SensorReading } from "@/lib/models";

const { width: SCREEN_WIDTH } = Dimensions.get("window");
const CHART_WIDTH = SCREEN_WIDTH - 32;
const CHART_HEIGHT = 160;
const CHART_PAD = { top: 16, right: 16, bottom: 28, left: 44 };

// ─── Mini Line Chart ──────────────────────────────────────────────────────────

type ChartMetric = "temperature" | "humidity" | "co2";

interface LineChartProps {
  data: SensorReading[];
  metric: ChartMetric;
  color: string;
  unit: string;
}

function LineChart({ data, metric, color, unit }: LineChartProps) {
  const colors = useColors();
  if (data.length < 2) return null;

  const values = data.map((d) => d[metric] as number);
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);
  const range = maxVal - minVal || 1;

  const innerW = CHART_WIDTH - CHART_PAD.left - CHART_PAD.right;
  const innerH = CHART_HEIGHT - CHART_PAD.top - CHART_PAD.bottom;

  const toX = (i: number) => CHART_PAD.left + (i / (data.length - 1)) * innerW;
  const toY = (v: number) => CHART_PAD.top + innerH - ((v - minVal) / range) * innerH;

  const points = data.map((d, i) => `${toX(i)},${toY(d[metric] as number)}`).join(" ");

  const yLabels = [minVal, (minVal + maxVal) / 2, maxVal];

  return (
    <Svg width={CHART_WIDTH} height={CHART_HEIGHT}>
      <Defs>
        <LinearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
          <Stop offset="0" stopColor={color} stopOpacity="0.18" />
          <Stop offset="1" stopColor={color} stopOpacity="0.01" />
        </LinearGradient>
      </Defs>

      {/* Grid lines */}
      {yLabels.map((v, i) => (
        <React.Fragment key={i}>
          <Line
            x1={CHART_PAD.left}
            y1={toY(v)}
            x2={CHART_WIDTH - CHART_PAD.right}
            y2={toY(v)}
            stroke={colors.border}
            strokeWidth="1"
            strokeDasharray="4,4"
          />
          <SvgText
            x={CHART_PAD.left - 4}
            y={toY(v) + 4}
            fontSize="10"
            fill={colors.muted}
            textAnchor="end"
          >
            {v.toFixed(0)}
          </SvgText>
        </React.Fragment>
      ))}

      {/* Filled area */}
      <Polyline
        points={[
          `${CHART_PAD.left},${CHART_PAD.top + innerH}`,
          ...data.map((d, i) => `${toX(i)},${toY(d[metric] as number)}`),
          `${CHART_WIDTH - CHART_PAD.right},${CHART_PAD.top + innerH}`,
        ].join(" ")}
        fill={`url(#grad)`}
        stroke="none"
      />

      {/* Line */}
      <Polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="2.5"
        strokeLinejoin="round"
        strokeLinecap="round"
      />

      {/* X axis labels (first, middle, last) */}
      {[0, Math.floor(data.length / 2), data.length - 1].map((i) => (
        <SvgText
          key={i}
          x={toX(i)}
          y={CHART_HEIGHT - 4}
          fontSize="9"
          fill={colors.muted}
          textAnchor="middle"
        >
          {formatHour(data[i].timestamp)}
        </SvgText>
      ))}
    </Svg>
  );
}

function formatHour(date: Date): string {
  return new Date(date).toLocaleTimeString("es-AR", { hour: "2-digit", minute: "2-digit" });
}

function formatDateTime(date: Date): string {
  return new Date(date).toLocaleString("es-AR", {
    day: "2-digit", month: "2-digit",
    hour: "2-digit", minute: "2-digit",
  });
}

// ─── Detail Screen ────────────────────────────────────────────────────────────

const RANGE_OPTIONS = [
  { label: "12h", hours: 12 },
  { label: "24h", hours: 24 },
  { label: "7d", hours: 168 },
];

export default function SilobagDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();
  const colors = useColors();
  const { getSilobag } = useSmisia();
  const [activeMetric, setActiveMetric] = useState<ChartMetric>("temperature");
  const [rangeHours, setRangeHours] = useState(24);

  const silobag = getSilobag(id);

  const filteredHistory = useMemo(() => {
    if (!silobag) return [];
    const cutoff = new Date(Date.now() - rangeHours * 60 * 60 * 1000);
    return silobag.history.filter((r) => new Date(r.timestamp) >= cutoff);
  }, [silobag, rangeHours]);

  if (!silobag) {
    return (
      <ScreenContainer>
        <View style={styles.notFound}>
          <Text style={[styles.notFoundText, { color: colors.muted }]}>
            Silobolsa no encontrada.
          </Text>
          <Pressable onPress={() => router.back()} style={({ pressed }) => [pressed && { opacity: 0.7 }]}>
            <Text style={[styles.backLink, { color: colors.primary }]}>← Volver</Text>
          </Pressable>
        </View>
      </ScreenContainer>
    );
  }

  const r = silobag.lastReading;
  const pred = silobag.prediction;

  const metricConfig: Record<ChartMetric, { label: string; unit: string; color: string; icon: string }> = {
    temperature: { label: "Temperatura", unit: "°C", color: colors.error, icon: "🌡️" },
    humidity:    { label: "Humedad",      unit: "%",  color: colors.info,  icon: "💧" },
    co2:         { label: "CO₂",          unit: "ppm",color: colors.warning, icon: "🌬️" },
  };

  const riskColors: Record<string, string> = {
    safe:   colors.success,
    low:    colors.warning,
    medium: colors.warning,
    high:   colors.error,
  };

  return (
    <ScreenContainer containerClassName="bg-background">
      {/* Header */}
      <View style={[styles.header, { backgroundColor: colors.surface, borderBottomColor: colors.border }]}>
        <Pressable
          onPress={() => router.back()}
          style={({ pressed }) => [styles.backBtn, pressed && { opacity: 0.6 }]}
        >
          <IconSymbol name="chevron.left" size={22} color={colors.primary} />
        </Pressable>
        <View style={styles.headerCenter}>
          <Text style={[styles.headerTitle, { color: colors.foreground }]} numberOfLines={1}>
            {silobag.name}
          </Text>
          <Text style={[styles.headerSub, { color: colors.muted }]}>{silobag.location}</Text>
        </View>
        <RiskBadge level={silobag.riskLevel} size="sm" />
      </View>

      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>

        {/* Current Readings */}
        <SectionHeader title="Lecturas Actuales" subtitle={r ? `Actualizado: ${formatDateTime(r.timestamp)}` : undefined} />
        <View style={styles.readingsRow}>
          {r ? (
            <>
              <ReadingCard icon="🌡️" label="Temperatura" value={r.temperature.toFixed(1)} unit="°C" color={colors.error} />
              <ReadingCard icon="💧" label="Humedad"      value={r.humidity.toFixed(1)}     unit="%"  color={colors.info} />
              <ReadingCard icon="🌬️" label="CO₂"          value={r.co2.toFixed(0)}           unit="ppm" color={colors.warning} />
            </>
          ) : (
            <Text style={[styles.noData, { color: colors.muted }]}>Sin datos disponibles</Text>
          )}
        </View>

        <Divider />

        {/* AI Prediction */}
        {pred && (
          <>
            <SectionHeader title="Predicción IA" subtitle="Análisis basado en reglas + tendencias" />
            <View style={[styles.predCard, { backgroundColor: colors.surface, borderColor: colors.border }]}>
              <View style={styles.predHeader}>
                <View style={[styles.riskIndicator, { backgroundColor: riskColors[pred.riskLevel] }]} />
                <Text style={[styles.predRisk, { color: riskColors[pred.riskLevel] }]}>
                  {pred.riskLevel === "safe" ? "Sin riesgo" :
                   pred.riskLevel === "low"  ? "Riesgo Bajo" :
                   pred.riskLevel === "medium" ? "Riesgo Medio" : "Riesgo Alto"}
                </Text>
                <Text style={[styles.predConf, { color: colors.muted }]}>
                  Confianza: {pred.confidence}%
                </Text>
              </View>
              <Text style={[styles.predExplanation, { color: colors.foreground }]}>
                {pred.explanation}
              </Text>

              {pred.triggeredRules.length > 0 && (
                <View style={styles.rulesContainer}>
                  {pred.triggeredRules.map((rule, i) => (
                    <View key={i} style={[styles.ruleTag, { backgroundColor: colors.background }]}>
                      <Text style={[styles.ruleText, { color: colors.warning }]}>⚠ {rule}</Text>
                    </View>
                  ))}
                </View>
              )}

              <View style={styles.recsContainer}>
                <Text style={[styles.recsTitle, { color: colors.foreground }]}>Recomendaciones:</Text>
                {pred.recommendations.map((rec, i) => (
                  <View key={i} style={styles.recRow}>
                    <Text style={[styles.recBullet, { color: colors.primary }]}>→</Text>
                    <Text style={[styles.recText, { color: colors.foreground }]}>{rec}</Text>
                  </View>
                ))}
              </View>
            </View>

            <Divider />
          </>
        )}

        {/* Chart */}
        <SectionHeader title="Histórico de Sensores" />

        {/* Metric selector */}
        <View style={styles.metricSelector}>
          {(Object.keys(metricConfig) as ChartMetric[]).map((m) => (
            <Pressable
              key={m}
              onPress={() => setActiveMetric(m)}
              style={({ pressed }) => [
                styles.metricBtn,
                { backgroundColor: activeMetric === m ? colors.primary : colors.surface,
                  borderColor: colors.border },
                pressed && { opacity: 0.75 },
              ]}
            >
              <Text style={[
                styles.metricBtnText,
                { color: activeMetric === m ? "#fff" : colors.muted },
              ]}>
                {metricConfig[m].icon} {metricConfig[m].label}
              </Text>
            </Pressable>
          ))}
        </View>

        {/* Range selector */}
        <View style={styles.rangeSelector}>
          {RANGE_OPTIONS.map((opt) => (
            <Pressable
              key={opt.label}
              onPress={() => setRangeHours(opt.hours)}
              style={({ pressed }) => [
                styles.rangeBtn,
                { backgroundColor: rangeHours === opt.hours ? colors.primary : colors.background,
                  borderColor: colors.border },
                pressed && { opacity: 0.75 },
              ]}
            >
              <Text style={[
                styles.rangeBtnText,
                { color: rangeHours === opt.hours ? "#fff" : colors.muted },
              ]}>
                {opt.label}
              </Text>
            </Pressable>
          ))}
        </View>

        {/* Chart */}
        <View style={[styles.chartContainer, { backgroundColor: colors.surface, borderColor: colors.border }]}>
          {filteredHistory.length >= 2 ? (
            <>
              <LineChart
                data={filteredHistory}
                metric={activeMetric}
                color={metricConfig[activeMetric].color}
                unit={metricConfig[activeMetric].unit}
              />
              <Text style={[styles.chartCaption, { color: colors.muted }]}>
                {metricConfig[activeMetric].label} ({metricConfig[activeMetric].unit}) — {filteredHistory.length} lecturas
              </Text>
            </>
          ) : (
            <Text style={[styles.noData, { color: colors.muted }]}>
              No hay suficientes datos para el rango seleccionado.
            </Text>
          )}
        </View>

        <Divider />

        {/* Info */}
        <SectionHeader title="Información del Silobolsa" />
        <View style={[styles.infoCard, { backgroundColor: colors.surface, borderColor: colors.border }]}>
          <InfoRow label="Grano" value={silobag.grainType} colors={colors} />
          <InfoRow label="Capacidad" value={`${silobag.capacityTons} toneladas`} colors={colors} />
          <InfoRow label="Instalación" value={new Date(silobag.installDate).toLocaleDateString("es-AR")} colors={colors} />
          <InfoRow label="Coordenadas" value={`${silobag.latitude.toFixed(4)}, ${silobag.longitude.toFixed(4)}`} colors={colors} />
        </View>

        {/* History Table */}
        <SectionHeader title="Últimas Lecturas" />
        <View style={[styles.historyTable, { backgroundColor: colors.surface, borderColor: colors.border }]}>
          <View style={[styles.historyHeader, { borderBottomColor: colors.border }]}>
            <Text style={[styles.historyHeaderCell, { color: colors.muted, flex: 2 }]}>Hora</Text>
            <Text style={[styles.historyHeaderCell, { color: colors.muted }]}>Temp</Text>
            <Text style={[styles.historyHeaderCell, { color: colors.muted }]}>Hum</Text>
            <Text style={[styles.historyHeaderCell, { color: colors.muted }]}>CO₂</Text>
          </View>
          {silobag.history.slice(-8).reverse().map((reading) => (
            <View key={reading.id} style={[styles.historyRow, { borderBottomColor: colors.border }]}>
              <Text style={[styles.historyCell, { color: colors.muted, flex: 2 }]}>
                {formatHour(reading.timestamp)}
              </Text>
              <Text style={[styles.historyCell, { color: colors.foreground }]}>
                {reading.temperature.toFixed(1)}°
              </Text>
              <Text style={[styles.historyCell, { color: colors.foreground }]}>
                {reading.humidity.toFixed(1)}%
              </Text>
              <Text style={[styles.historyCell, { color: colors.foreground }]}>
                {reading.co2.toFixed(0)}
              </Text>
            </View>
          ))}
        </View>

        <View style={{ height: 32 }} />
      </ScrollView>
    </ScreenContainer>
  );
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function ReadingCard({ icon, label, value, unit, color }: {
  icon: string; label: string; value: string; unit: string; color: string;
}) {
  const colors = useColors();
  return (
    <View style={[styles.readingCard, { backgroundColor: colors.surface, borderColor: colors.border }]}>
      <Text style={styles.readingIcon}>{icon}</Text>
      <Text style={[styles.readingValue, { color }]}>{value}</Text>
      <Text style={[styles.readingUnit, { color: colors.muted }]}>{unit}</Text>
      <Text style={[styles.readingLabel, { color: colors.muted }]}>{label}</Text>
    </View>
  );
}

function InfoRow({ label, value, colors }: { label: string; value: string; colors: any }) {
  return (
    <View style={styles.infoRow}>
      <Text style={[styles.infoLabel, { color: colors.muted }]}>{label}</Text>
      <Text style={[styles.infoValue, { color: colors.foreground }]}>{value}</Text>
    </View>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  notFound: { flex: 1, alignItems: "center", justifyContent: "center", gap: 12 },
  notFoundText: { fontSize: 16, lineHeight: 22 },
  backLink: { fontSize: 15, fontWeight: "600", lineHeight: 20 },
  header: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 12,
    paddingVertical: 12,
    borderBottomWidth: 1,
    gap: 10,
  },
  backBtn: { padding: 4 },
  headerCenter: { flex: 1 },
  headerTitle: { fontSize: 16, fontWeight: "700", lineHeight: 22 },
  headerSub: { fontSize: 12, lineHeight: 16 },
  scrollContent: { padding: 16 },
  readingsRow: { flexDirection: "row", gap: 10, marginBottom: 4 },
  readingCard: {
    flex: 1,
    borderRadius: 12,
    padding: 12,
    alignItems: "center",
    borderWidth: 1,
  },
  readingIcon: { fontSize: 22, lineHeight: 28, marginBottom: 4 },
  readingValue: { fontSize: 22, fontWeight: "800", lineHeight: 28 },
  readingUnit: { fontSize: 11, lineHeight: 16 },
  readingLabel: { fontSize: 11, fontWeight: "500", marginTop: 2, lineHeight: 16 },
  noData: { fontSize: 13, lineHeight: 18, textAlign: "center", paddingVertical: 20 },
  predCard: {
    borderRadius: 14,
    padding: 16,
    borderWidth: 1,
    gap: 12,
    marginBottom: 4,
  },
  predHeader: { flexDirection: "row", alignItems: "center", gap: 8 },
  riskIndicator: { width: 10, height: 10, borderRadius: 5 },
  predRisk: { fontSize: 15, fontWeight: "700", flex: 1, lineHeight: 20 },
  predConf: { fontSize: 12, lineHeight: 16 },
  predExplanation: { fontSize: 13, lineHeight: 20 },
  rulesContainer: { flexDirection: "row", flexWrap: "wrap", gap: 6 },
  ruleTag: { borderRadius: 8, paddingHorizontal: 8, paddingVertical: 4 },
  ruleText: { fontSize: 11, fontWeight: "600", lineHeight: 16 },
  recsContainer: { gap: 6 },
  recsTitle: { fontSize: 13, fontWeight: "700", lineHeight: 18 },
  recRow: { flexDirection: "row", gap: 6, alignItems: "flex-start" },
  recBullet: { fontSize: 13, fontWeight: "700", lineHeight: 20 },
  recText: { fontSize: 13, flex: 1, lineHeight: 20 },
  metricSelector: { flexDirection: "row", gap: 8, marginBottom: 10 },
  metricBtn: {
    flex: 1,
    borderRadius: 10,
    paddingVertical: 8,
    alignItems: "center",
    borderWidth: 1,
  },
  metricBtnText: { fontSize: 11, fontWeight: "600", lineHeight: 16 },
  rangeSelector: { flexDirection: "row", gap: 8, marginBottom: 12 },
  rangeBtn: {
    flex: 1,
    borderRadius: 8,
    paddingVertical: 6,
    alignItems: "center",
    borderWidth: 1,
  },
  rangeBtnText: { fontSize: 12, fontWeight: "600", lineHeight: 16 },
  chartContainer: {
    borderRadius: 14,
    padding: 12,
    borderWidth: 1,
    marginBottom: 4,
    alignItems: "center",
  },
  chartCaption: { fontSize: 11, marginTop: 6, lineHeight: 16 },
  infoCard: {
    borderRadius: 14,
    padding: 16,
    borderWidth: 1,
    gap: 10,
    marginBottom: 4,
  },
  infoRow: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  infoLabel: { fontSize: 13, lineHeight: 18 },
  infoValue: { fontSize: 13, fontWeight: "600", lineHeight: 18 },
  historyTable: {
    borderRadius: 14,
    borderWidth: 1,
    overflow: "hidden",
    marginBottom: 4,
  },
  historyHeader: {
    flexDirection: "row",
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderBottomWidth: 1,
  },
  historyHeaderCell: { flex: 1, fontSize: 11, fontWeight: "700", lineHeight: 16 },
  historyRow: {
    flexDirection: "row",
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderBottomWidth: 0.5,
  },
  historyCell: { flex: 1, fontSize: 12, lineHeight: 18 },
});
