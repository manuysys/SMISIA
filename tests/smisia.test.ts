// SMISIA Unit Tests
// Tests for mock data engine and AI alert analysis logic

import { describe, it, expect } from "vitest";
import { analyzeReading, generateInitialSilobags, generateInitialAlerts, generateNewReading } from "../lib/mock-data";
import { SensorReading, DEFAULT_SETTINGS } from "../lib/models";

function makeReading(overrides: Partial<SensorReading> = {}): SensorReading {
  return {
    id: "test-r-1",
    silobagId: "test-sb",
    timestamp: new Date(),
    temperature: 22,
    humidity: 12,
    co2: 800,
    ...overrides,
  };
}

describe("analyzeReading — safe conditions", () => {
  it("returns safe when all values are within thresholds", () => {
    const reading = makeReading({ temperature: 22, humidity: 12, co2: 800 });
    const result = analyzeReading(reading, DEFAULT_SETTINGS);
    expect(result.riskLevel).toBe("safe");
    expect(result.confidence).toBeGreaterThan(0);
    expect(result.recommendations.length).toBeGreaterThan(0);
  });
});

describe("analyzeReading — temperature alerts", () => {
  it("returns low risk when temperature slightly above threshold", () => {
    const reading = makeReading({ temperature: DEFAULT_SETTINGS.maxTemperature + 1 });
    const result = analyzeReading(reading, DEFAULT_SETTINGS);
    expect(["low", "medium"]).toContain(result.riskLevel);
  });

  it("returns medium or high risk when temperature critically high", () => {
    const reading = makeReading({ temperature: DEFAULT_SETTINGS.maxTemperature + 8 });
    const result = analyzeReading(reading, DEFAULT_SETTINGS);
    expect(["medium", "high"]).toContain(result.riskLevel);
  });
});

describe("analyzeReading — humidity alerts", () => {
  it("returns medium or high risk when humidity critically high", () => {
    const reading = makeReading({ humidity: DEFAULT_SETTINGS.maxHumidity + 6 });
    const result = analyzeReading(reading, DEFAULT_SETTINGS);
    expect(["medium", "high"]).toContain(result.riskLevel);
  });
});

describe("analyzeReading — CO2 alerts", () => {
  it("returns non-safe when CO2 above threshold", () => {
    const reading = makeReading({ co2: DEFAULT_SETTINGS.maxCO2 + 200 });
    const result = analyzeReading(reading, DEFAULT_SETTINGS);
    expect(result.riskLevel).not.toBe("safe");
  });
});

describe("analyzeReading — combined conditions", () => {
  it("returns high risk when temperature, humidity, and CO2 are all critical", () => {
    const reading = makeReading({
      temperature: DEFAULT_SETTINGS.maxTemperature + 8,
      humidity: DEFAULT_SETTINGS.maxHumidity + 6,
      co2: DEFAULT_SETTINGS.maxCO2 * 2,
    });
    const result = analyzeReading(reading, DEFAULT_SETTINGS);
    expect(result.riskLevel).toBe("high");
    expect(result.triggeredRules.length).toBeGreaterThan(1);
  });

  it("includes triggered rules in the result", () => {
    const reading = makeReading({ temperature: 40, humidity: 22 });
    const result = analyzeReading(reading, DEFAULT_SETTINGS);
    expect(result.triggeredRules.length).toBeGreaterThan(0);
  });
});

describe("generateInitialSilobags", () => {
  it("generates 5 silobags", () => {
    const silobags = generateInitialSilobags();
    expect(silobags.length).toBe(5);
  });

  it("each silobag has a lastReading", () => {
    const silobags = generateInitialSilobags();
    silobags.forEach((sb) => {
      expect(sb.lastReading).not.toBeNull();
    });
  });

  it("each silobag has a prediction", () => {
    const silobags = generateInitialSilobags();
    silobags.forEach((sb) => {
      expect(sb.prediction).not.toBeNull();
    });
  });

  it("each silobag has history with at least 10 readings", () => {
    const silobags = generateInitialSilobags();
    silobags.forEach((sb) => {
      expect(sb.history.length).toBeGreaterThanOrEqual(10);
    });
  });

  it("at least one silobag has a non-safe risk level (seeded scenarios)", () => {
    const silobags = generateInitialSilobags();
    const nonSafe = silobags.filter((sb) => sb.riskLevel !== "safe");
    expect(nonSafe.length).toBeGreaterThan(0);
  });
});

describe("generateInitialAlerts", () => {
  it("generates alerts for non-safe silobags", () => {
    const silobags = generateInitialSilobags();
    const alerts = generateInitialAlerts(silobags);
    const nonSafeSilobags = silobags.filter((sb) => sb.riskLevel !== "safe");
    expect(alerts.length).toBe(nonSafeSilobags.length);
  });

  it("all alerts have valid risk levels", () => {
    const silobags = generateInitialSilobags();
    const alerts = generateInitialAlerts(silobags);
    alerts.forEach((alert) => {
      expect(["low", "medium", "high"]).toContain(alert.riskLevel);
    });
  });
});

describe("generateNewReading", () => {
  it("generates a new reading based on last reading", () => {
    const silobags = generateInitialSilobags();
    const sb = silobags[0];
    const newReading = generateNewReading(sb);
    expect(newReading.silobagId).toBe(sb.id);
    expect(newReading.temperature).toBeGreaterThan(0);
    expect(newReading.humidity).toBeGreaterThan(0);
    expect(newReading.co2).toBeGreaterThan(0);
  });

  it("new reading timestamp is recent", () => {
    const silobags = generateInitialSilobags();
    const newReading = generateNewReading(silobags[0]);
    const now = Date.now();
    const readingTime = new Date(newReading.timestamp).getTime();
    expect(Math.abs(now - readingTime)).toBeLessThan(5000); // within 5 seconds
  });
});
