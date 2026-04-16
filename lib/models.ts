// SMISIA Data Models
// Defines shared types for silobags, sensor readings, and alerts

export type RiskLevel = "safe" | "low" | "medium" | "high";

export interface SensorReading {
  id: string;
  silobagId: string;
  timestamp: Date;
  temperature: number; // °C
  humidity: number;    // % RH
  co2: number;         // ppm
}

export interface AIPrediction {
  riskLevel: RiskLevel;
  confidence: number;       // 0–100
  explanation: string;
  recommendations: string[];
  triggeredRules: string[];
}

export interface Alert {
  id: string;
  silobagId: string;
  silobagName: string;
  timestamp: Date;
  riskLevel: Exclude<RiskLevel, "safe">;
  message: string;
  type: "temperature" | "humidity" | "co2" | "combined";
  acknowledged: boolean;
}

export interface Silobag {
  id: string;
  name: string;
  location: string;
  latitude: number;
  longitude: number;
  grainType: string;
  capacityTons: number;
  installDate: Date;
  lastReading: SensorReading | null;
  riskLevel: RiskLevel;
  prediction: AIPrediction | null;
  history: SensorReading[];
  isActive: boolean;
}

export interface AppSettings {
  maxTemperature: number;    // °C
  maxHumidity: number;       // %
  maxCO2: number;            // ppm
  samplingIntervalHours: number;
  pushNotificationsEnabled: boolean;
  highRiskAlertsOnly: boolean;
}

export const DEFAULT_SETTINGS: AppSettings = {
  maxTemperature: 28,
  maxHumidity: 14,
  maxCO2: 2000,
  samplingIntervalHours: 2,
  pushNotificationsEnabled: true,
  highRiskAlertsOnly: false,
};
