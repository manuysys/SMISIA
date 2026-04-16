// SMISIA Mock IoT Data Engine
// Simulates sensor readings from silobags and applies AI rule-based analysis

import { Silobag, SensorReading, Alert, AIPrediction, RiskLevel, AppSettings, DEFAULT_SETTINGS } from "./models";

// ─── Seed Data ────────────────────────────────────────────────────────────────

const SILOBAG_SEEDS = [
  { id: "sb-001", name: "Silobolsa A1", location: "Lote Norte", lat: -34.6037, lng: -58.3816, grain: "Soja", capacity: 200 },
  { id: "sb-002", name: "Silobolsa A2", location: "Lote Norte", lat: -34.6045, lng: -58.3820, grain: "Maíz", capacity: 180 },
  { id: "sb-003", name: "Silobolsa B1", location: "Lote Sur",   lat: -34.6120, lng: -58.3790, grain: "Trigo", capacity: 220 },
  { id: "sb-004", name: "Silobolsa B2", location: "Lote Sur",   lat: -34.6130, lng: -58.3795, grain: "Soja", capacity: 200 },
  { id: "sb-005", name: "Silobolsa C1", location: "Lote Este",  lat: -34.6080, lng: -58.3700, grain: "Girasol", capacity: 150 },
];

// ─── Reading Generator ────────────────────────────────────────────────────────

function randomBetween(min: number, max: number, decimals = 1): number {
  return parseFloat((Math.random() * (max - min) + min).toFixed(decimals));
}

/**
 * Generates a realistic sensor reading for a silobag.
 * Some silobags are seeded with elevated values to trigger alerts.
 */
function generateReading(silobagId: string, hoursAgo: number, scenario: "normal" | "warning" | "danger"): SensorReading {
  const now = new Date();
  now.setHours(now.getHours() - hoursAgo);

  let temp: number, humidity: number, co2: number;

  switch (scenario) {
    case "danger":
      temp     = randomBetween(30, 36);
      humidity = randomBetween(16, 22);
      co2      = randomBetween(2500, 4000);
      break;
    case "warning":
      temp     = randomBetween(26, 31);
      humidity = randomBetween(13, 17);
      co2      = randomBetween(1500, 2600);
      break;
    default:
      temp     = randomBetween(18, 26);
      humidity = randomBetween(10, 14);
      co2      = randomBetween(400, 1500);
  }

  // Add slight trend over time (older readings slightly lower)
  const trendFactor = hoursAgo * 0.02;
  temp     = Math.max(10, temp - trendFactor);
  humidity = Math.max(8, humidity - trendFactor * 0.5);
  co2      = Math.max(300, co2 - trendFactor * 20);

  return {
    id: `${silobagId}-r-${hoursAgo}`,
    silobagId,
    timestamp: now,
    temperature: parseFloat(temp.toFixed(1)),
    humidity:    parseFloat(humidity.toFixed(1)),
    co2:         parseFloat(co2.toFixed(0)),
  };
}

// ─── AI Alert Engine ──────────────────────────────────────────────────────────

export function analyzeReading(reading: SensorReading, settings: AppSettings = DEFAULT_SETTINGS): AIPrediction {
  const { temperature, humidity, co2 } = reading;
  const { maxTemperature, maxHumidity, maxCO2 } = settings;

  const triggeredRules: string[] = [];
  const recommendations: string[] = [];
  let riskScore = 0;

  // Rule 1: High temperature
  if (temperature > maxTemperature + 5) {
    riskScore += 40;
    triggeredRules.push("Temperatura crítica");
    recommendations.push("Verificar ventilación del silobolsa inmediatamente.");
  } else if (temperature > maxTemperature) {
    riskScore += 20;
    triggeredRules.push("Temperatura elevada");
    recommendations.push("Monitorear temperatura cada 30 minutos.");
  }

  // Rule 2: High humidity
  if (humidity > maxHumidity + 4) {
    riskScore += 40;
    triggeredRules.push("Humedad crítica");
    recommendations.push("Riesgo de hongos y deterioro. Considerar extracción urgente.");
  } else if (humidity > maxHumidity) {
    riskScore += 20;
    triggeredRules.push("Humedad elevada");
    recommendations.push("Revisar hermeticidad del silobolsa.");
  }

  // Rule 3: High CO2 (biological activity)
  if (co2 > maxCO2 * 1.5) {
    riskScore += 30;
    triggeredRules.push("CO₂ crítico — actividad biológica alta");
    recommendations.push("Alta actividad biológica detectada. Inspección física recomendada.");
  } else if (co2 > maxCO2) {
    riskScore += 15;
    triggeredRules.push("CO₂ elevado");
    recommendations.push("Posible inicio de fermentación. Aumentar frecuencia de monitoreo.");
  }

  // Rule 4: Combined temperature + humidity (synergistic risk)
  if (temperature > maxTemperature && humidity > maxHumidity) {
    riskScore += 20;
    triggeredRules.push("Combinación temperatura + humedad elevadas");
    recommendations.push("Condiciones favorables para deterioro acelerado del grano.");
  }

  // Determine risk level
  let riskLevel: RiskLevel;
  let explanation: string;

  if (riskScore === 0) {
    riskLevel = "safe";
    explanation = "Todos los parámetros dentro de los rangos normales. El grano está en condiciones óptimas.";
    recommendations.push("Continuar monitoreo de rutina.");
  } else if (riskScore <= 20) {
    riskLevel = "low";
    explanation = "Se detectaron valores ligeramente elevados. Riesgo bajo de deterioro.";
  } else if (riskScore <= 50) {
    riskLevel = "medium";
    explanation = "Parámetros fuera de rango. Posible inicio de deterioro del grano. Se recomienda atención.";
  } else {
    riskLevel = "high";
    explanation = "Condiciones críticas detectadas. Alto riesgo de deterioro severo del grano. Acción inmediata requerida.";
  }

  const confidence = Math.min(95, 60 + riskScore * 0.5);

  return {
    riskLevel,
    confidence: parseFloat(confidence.toFixed(0)),
    explanation,
    recommendations: recommendations.length > 0 ? recommendations : ["Mantener monitoreo regular."],
    triggeredRules,
  };
}

export function generateAlert(silobag: Silobag, prediction: AIPrediction): Alert | null {
  if (prediction.riskLevel === "safe") return null;
  if (!silobag.lastReading) return null;

  const { temperature, humidity, co2 } = silobag.lastReading;
  let type: Alert["type"] = "combined";
  let message = "";

  const hasHighTemp = temperature > DEFAULT_SETTINGS.maxTemperature;
  const hasHighHumidity = humidity > DEFAULT_SETTINGS.maxHumidity;
  const hasHighCO2 = co2 > DEFAULT_SETTINGS.maxCO2;

  if (hasHighTemp && hasHighHumidity) {
    type = "combined";
    message = `Posible deterioro: temperatura ${temperature}°C y humedad ${humidity}%`;
  } else if (hasHighHumidity) {
    type = "humidity";
    message = `Humedad elevada: ${humidity}% (máx. ${DEFAULT_SETTINGS.maxHumidity}%)`;
  } else if (hasHighTemp) {
    type = "temperature";
    message = `Temperatura elevada: ${temperature}°C (máx. ${DEFAULT_SETTINGS.maxTemperature}°C)`;
  } else if (hasHighCO2) {
    type = "co2";
    message = `Actividad biológica detectada: CO₂ ${co2} ppm`;
  } else {
    message = prediction.explanation;
  }

  return {
    id: `alert-${silobag.id}-${Date.now()}`,
    silobagId: silobag.id,
    silobagName: silobag.name,
    timestamp: new Date(),
    riskLevel: prediction.riskLevel as Exclude<RiskLevel, "safe">,
    message,
    type,
    acknowledged: false,
  };
}

// ─── Initial Dataset ──────────────────────────────────────────────────────────

const SCENARIOS: Array<"normal" | "warning" | "danger"> = [
  "normal", "normal", "warning", "danger", "normal",
];

export function generateInitialSilobags(): Silobag[] {
  return SILOBAG_SEEDS.map((seed, index) => {
    const scenario = SCENARIOS[index];
    const history: SensorReading[] = [];

    // Generate 30 readings (every 2 hours = 60 hours of history)
    for (let h = 60; h >= 0; h -= 2) {
      history.push(generateReading(seed.id, h, scenario));
    }

    const lastReading = history[history.length - 1];
    const prediction = analyzeReading(lastReading);

    return {
      id: seed.id,
      name: seed.name,
      location: seed.location,
      latitude: seed.lat,
      longitude: seed.lng,
      grainType: seed.grain,
      capacityTons: seed.capacity,
      installDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
      lastReading,
      riskLevel: prediction.riskLevel,
      prediction,
      history,
      isActive: true,
    };
  });
}

export function generateNewReading(silobag: Silobag): SensorReading {
  const scenario = SCENARIOS[SILOBAG_SEEDS.findIndex((s) => s.id === silobag.id)] ?? "normal";
  // Small random variation from last reading
  const last = silobag.lastReading;
  if (!last) return generateReading(silobag.id, 0, scenario);

  const variation = () => randomBetween(-0.5, 0.5);
  return {
    id: `${silobag.id}-r-${Date.now()}`,
    silobagId: silobag.id,
    timestamp: new Date(),
    temperature: parseFloat(Math.max(10, last.temperature + variation()).toFixed(1)),
    humidity:    parseFloat(Math.max(5, last.humidity + variation() * 0.3).toFixed(1)),
    co2:         parseFloat(Math.max(300, last.co2 + randomBetween(-50, 50)).toFixed(0)),
  };
}

export function generateInitialAlerts(silobags: Silobag[]): Alert[] {
  const alerts: Alert[] = [];
  silobags.forEach((sb) => {
    if (sb.prediction && sb.prediction.riskLevel !== "safe") {
      const alert = generateAlert(sb, sb.prediction);
      if (alert) alerts.push(alert);
    }
  });
  return alerts;
}
