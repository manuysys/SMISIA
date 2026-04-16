// SMISIA Settings Screen
// Configure alert thresholds, sampling intervals, notifications, and simulation

import React, { useState, useCallback } from "react";
import {
  View,
  Text,
  ScrollView,
  Pressable,
  TextInput,
  Switch,
  StyleSheet,
  Alert,
  Platform,
} from "react-native";
import * as Haptics from "expo-haptics";

import { ScreenContainer } from "@/components/screen-container";
import { Divider } from "@/components/smisia-ui";
import { useColors } from "@/hooks/use-colors";
import { useSmisia } from "@/lib/smisia-context";
import { AppSettings } from "@/lib/models";

// ─── Setting Row ──────────────────────────────────────────────────────────────

function SettingRow({ label, subtitle, children }: {
  label: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  const colors = useColors();
  return (
    <View style={styles.settingRow}>
      <View style={styles.settingLabel}>
        <Text style={[styles.settingLabelText, { color: colors.foreground }]}>{label}</Text>
        {subtitle && (
          <Text style={[styles.settingSubtitle, { color: colors.muted }]}>{subtitle}</Text>
        )}
      </View>
      <View style={styles.settingControl}>{children}</View>
    </View>
  );
}

// ─── Number Input ─────────────────────────────────────────────────────────────

function NumberInput({
  value,
  onChangeText,
  unit,
}: {
  value: string;
  onChangeText: (v: string) => void;
  unit: string;
}) {
  const colors = useColors();
  return (
    <View style={styles.numberInputRow}>
      <TextInput
        value={value}
        onChangeText={onChangeText}
        keyboardType="numeric"
        returnKeyType="done"
        style={[
          styles.numberInput,
          {
            color: colors.foreground,
            backgroundColor: colors.background,
            borderColor: colors.border,
          },
        ]}
      />
      <Text style={[styles.numberUnit, { color: colors.muted }]}>{unit}</Text>
    </View>
  );
}

// ─── Section Header ───────────────────────────────────────────────────────────

function SettingSection({ title, icon }: { title: string; icon: string }) {
  const colors = useColors();
  return (
    <View style={styles.sectionHeader}>
      <Text style={styles.sectionIcon}>{icon}</Text>
      <Text style={[styles.sectionTitle, { color: colors.foreground }]}>{title}</Text>
    </View>
  );
}

// ─── Settings Screen ──────────────────────────────────────────────────────────

export default function SettingsScreen() {
  const colors = useColors();
  const { state, updateSettings, resetData, refreshData } = useSmisia();
  const s = state.settings;

  // Local state for numeric inputs (as strings for TextInput)
  const [maxTemp,     setMaxTemp]     = useState(String(s.maxTemperature));
  const [maxHumidity, setMaxHumidity] = useState(String(s.maxHumidity));
  const [maxCO2,      setMaxCO2]      = useState(String(s.maxCO2));

  const handleSaveThresholds = useCallback(() => {
    const temp = parseFloat(maxTemp);
    const hum  = parseFloat(maxHumidity);
    const co2  = parseFloat(maxCO2);

    if (isNaN(temp) || isNaN(hum) || isNaN(co2)) {
      Alert.alert("Error", "Por favor ingresa valores numéricos válidos.");
      return;
    }

    updateSettings({ maxTemperature: temp, maxHumidity: hum, maxCO2: co2 });

    if (Platform.OS !== "web") {
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    }
    Alert.alert("Guardado", "Umbrales actualizados correctamente.");
  }, [maxTemp, maxHumidity, maxCO2, updateSettings]);

  const handleIntervalChange = useCallback((hours: number) => {
    updateSettings({ samplingIntervalHours: hours });
    if (Platform.OS !== "web") {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    }
  }, [updateSettings]);

  const handleGenerateData = useCallback(() => {
    refreshData();
    if (Platform.OS !== "web") {
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    }
    Alert.alert("Datos generados", "Se simularon nuevas lecturas de sensores para todos los silobolsas.");
  }, [refreshData]);

  const handleResetData = useCallback(() => {
    Alert.alert(
      "Resetear datos",
      "¿Estás seguro? Esto eliminará todas las alertas y generará datos de prueba nuevos.",
      [
        { text: "Cancelar", style: "cancel" },
        {
          text: "Resetear",
          style: "destructive",
          onPress: () => {
            resetData();
            if (Platform.OS !== "web") {
              Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
            }
          },
        },
      ]
    );
  }, [resetData]);

  const INTERVAL_OPTIONS = [
    { label: "1 hora",  hours: 1 },
    { label: "2 horas", hours: 2 },
    { label: "4 horas", hours: 4 },
  ];

  return (
    <ScreenContainer containerClassName="bg-background">
      {/* Header */}
      <View style={[styles.header, { backgroundColor: colors.surface, borderBottomColor: colors.border }]}>
        <Text style={[styles.headerTitle, { color: colors.foreground }]}>Ajustes</Text>
        <Text style={[styles.headerSub, { color: colors.muted }]}>
          Configuración del sistema SMISIA
        </Text>
      </View>

      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>

        {/* Thresholds */}
        <SettingSection title="Umbrales de Alerta" icon="⚠️" />
        <View style={[styles.card, { backgroundColor: colors.surface, borderColor: colors.border }]}>
          <SettingRow label="Temperatura máxima" subtitle="Umbral para alerta de temperatura">
            <NumberInput value={maxTemp} onChangeText={setMaxTemp} unit="°C" />
          </SettingRow>
          <Divider />
          <SettingRow label="Humedad máxima" subtitle="Umbral para alerta de humedad">
            <NumberInput value={maxHumidity} onChangeText={setMaxHumidity} unit="%" />
          </SettingRow>
          <Divider />
          <SettingRow label="CO₂ máximo" subtitle="Umbral para alerta de gases">
            <NumberInput value={maxCO2} onChangeText={setMaxCO2} unit="ppm" />
          </SettingRow>

          <Pressable
            onPress={handleSaveThresholds}
            style={({ pressed }) => [
              styles.saveBtn,
              { backgroundColor: colors.primary },
              pressed && { opacity: 0.8 },
            ]}
          >
            <Text style={styles.saveBtnText}>Guardar Umbrales</Text>
          </Pressable>
        </View>

        <View style={{ height: 20 }} />

        {/* Sampling interval */}
        <SettingSection title="Intervalo de Muestreo" icon="🕐" />
        <View style={[styles.card, { backgroundColor: colors.surface, borderColor: colors.border }]}>
          <Text style={[styles.cardSubtitle, { color: colors.muted }]}>
            Frecuencia de envío de datos desde los sensores LoRa
          </Text>
          <View style={styles.intervalRow}>
            {INTERVAL_OPTIONS.map((opt) => (
              <Pressable
                key={opt.hours}
                onPress={() => handleIntervalChange(opt.hours)}
                style={({ pressed }) => [
                  styles.intervalBtn,
                  {
                    backgroundColor: s.samplingIntervalHours === opt.hours ? colors.primary : colors.background,
                    borderColor: s.samplingIntervalHours === opt.hours ? colors.primary : colors.border,
                  },
                  pressed && { opacity: 0.75 },
                ]}
              >
                <Text style={[
                  styles.intervalBtnText,
                  { color: s.samplingIntervalHours === opt.hours ? "#fff" : colors.muted },
                ]}>
                  {opt.label}
                </Text>
              </Pressable>
            ))}
          </View>
        </View>

        <View style={{ height: 20 }} />

        {/* Notifications */}
        <SettingSection title="Notificaciones" icon="🔔" />
        <View style={[styles.card, { backgroundColor: colors.surface, borderColor: colors.border }]}>
          <SettingRow label="Alertas push" subtitle="Recibir notificaciones de nuevas alertas">
            <Switch
              value={s.pushNotificationsEnabled}
              onValueChange={(v) => updateSettings({ pushNotificationsEnabled: v })}
              trackColor={{ false: colors.border, true: colors.primary }}
              thumbColor="#fff"
            />
          </SettingRow>
          <Divider />
          <SettingRow label="Solo alertas de alto riesgo" subtitle="Filtrar notificaciones por severidad">
            <Switch
              value={s.highRiskAlertsOnly}
              onValueChange={(v) => updateSettings({ highRiskAlertsOnly: v })}
              trackColor={{ false: colors.border, true: colors.primary }}
              thumbColor="#fff"
            />
          </SettingRow>
        </View>

        <View style={{ height: 20 }} />

        {/* Simulation */}
        <SettingSection title="Simulación IoT" icon="📡" />
        <View style={[styles.card, { backgroundColor: colors.surface, borderColor: colors.border }]}>
          <Text style={[styles.cardSubtitle, { color: colors.muted }]}>
            Genera datos de prueba para simular lecturas de sensores LoRa
          </Text>

          <Pressable
            onPress={handleGenerateData}
            style={({ pressed }) => [
              styles.simBtn,
              { backgroundColor: colors.primary },
              pressed && { opacity: 0.8 },
            ]}
          >
            <Text style={styles.simBtnText}>📡 Generar Datos de Prueba</Text>
          </Pressable>

          <Pressable
            onPress={handleResetData}
            style={({ pressed }) => [
              styles.simBtn,
              { backgroundColor: colors.error },
              pressed && { opacity: 0.8 },
            ]}
          >
            <Text style={styles.simBtnText}>🔄 Resetear Todos los Datos</Text>
          </Pressable>
        </View>

        <View style={{ height: 20 }} />

        {/* About */}
        <SettingSection title="Acerca de SMISIA" icon="ℹ️" />
        <View style={[styles.card, { backgroundColor: colors.surface, borderColor: colors.border }]}>
          <InfoItem label="Sistema" value="SMISIA v1.0.0" colors={colors} />
          <Divider />
          <InfoItem label="Protocolo IoT" value="LoRa / HTTP" colors={colors} />
          <Divider />
          <InfoItem label="IA" value="Motor de reglas + ML" colors={colors} />
          <Divider />
          <InfoItem label="Base de datos" value="Local (AsyncStorage)" colors={colors} />
          <Divider />
          <InfoItem label="Silobolsas activas" value={String(state.silobags.filter((s) => s.isActive).length)} colors={colors} />
        </View>

        <View style={{ height: 40 }} />
      </ScrollView>
    </ScreenContainer>
  );
}

function InfoItem({ label, value, colors }: { label: string; value: string; colors: any }) {
  return (
    <View style={styles.infoRow}>
      <Text style={[styles.infoLabel, { color: colors.muted }]}>{label}</Text>
      <Text style={[styles.infoValue, { color: colors.foreground }]}>{value}</Text>
    </View>
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
  scrollContent: { padding: 16 },
  sectionHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 10,
    marginTop: 4,
  },
  sectionIcon: { fontSize: 18, lineHeight: 24 },
  sectionTitle: { fontSize: 15, fontWeight: "700", lineHeight: 20 },
  card: {
    borderRadius: 14,
    padding: 16,
    borderWidth: 1,
    gap: 0,
  },
  cardSubtitle: { fontSize: 12, lineHeight: 18, marginBottom: 12 },
  settingRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: 10,
    gap: 12,
  },
  settingLabel: { flex: 1 },
  settingLabelText: { fontSize: 14, fontWeight: "600", lineHeight: 20 },
  settingSubtitle: { fontSize: 11, lineHeight: 16, marginTop: 2 },
  settingControl: { alignItems: "flex-end" },
  numberInputRow: { flexDirection: "row", alignItems: "center", gap: 6 },
  numberInput: {
    width: 64,
    height: 36,
    borderRadius: 8,
    borderWidth: 1,
    paddingHorizontal: 8,
    fontSize: 14,
    fontWeight: "600",
    textAlign: "center",
    lineHeight: 20,
  },
  numberUnit: { fontSize: 12, fontWeight: "500", lineHeight: 18 },
  saveBtn: {
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: "center",
    marginTop: 12,
  },
  saveBtnText: { color: "#fff", fontSize: 14, fontWeight: "700", lineHeight: 20 },
  intervalRow: { flexDirection: "row", gap: 10 },
  intervalBtn: {
    flex: 1,
    borderRadius: 10,
    paddingVertical: 10,
    alignItems: "center",
    borderWidth: 1,
  },
  intervalBtnText: { fontSize: 12, fontWeight: "600", lineHeight: 18 },
  simBtn: {
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: "center",
    marginTop: 10,
  },
  simBtnText: { color: "#fff", fontSize: 14, fontWeight: "700", lineHeight: 20 },
  infoRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 8,
  },
  infoLabel: { fontSize: 13, lineHeight: 18 },
  infoValue: { fontSize: 13, fontWeight: "600", lineHeight: 18 },
});
