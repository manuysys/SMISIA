// SMISIA Global State Context
// Manages silobags, alerts, and settings across the app

import React, { createContext, useContext, useReducer, useCallback, useEffect } from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { Silobag, Alert, AppSettings, DEFAULT_SETTINGS } from "./models";
import {
  generateInitialSilobags,
  generateInitialAlerts,
  generateNewReading,
  analyzeReading,
  generateAlert,
} from "./mock-data";

// ─── State ────────────────────────────────────────────────────────────────────

interface SmisiaState {
  silobags: Silobag[];
  alerts: Alert[];
  settings: AppSettings;
  lastRefresh: Date | null;
  isLoading: boolean;
}

type SmisiaAction =
  | { type: "INIT"; payload: { silobags: Silobag[]; alerts: Alert[] } }
  | { type: "REFRESH_DATA" }
  | { type: "UPDATE_SETTINGS"; payload: Partial<AppSettings> }
  | { type: "ACKNOWLEDGE_ALERT"; payload: string }
  | { type: "RESET_DATA" }
  | { type: "SET_LOADING"; payload: boolean };

function reducer(state: SmisiaState, action: SmisiaAction): SmisiaState {
  switch (action.type) {
    case "INIT":
      return {
        ...state,
        silobags: action.payload.silobags,
        alerts: action.payload.alerts,
        lastRefresh: new Date(),
        isLoading: false,
      };

    case "REFRESH_DATA": {
      const updatedSilobags = state.silobags.map((sb) => {
        const newReading = generateNewReading(sb);
        const updatedHistory = [...sb.history.slice(-29), newReading];
        const prediction = analyzeReading(newReading, state.settings);
        return {
          ...sb,
          lastReading: newReading,
          history: updatedHistory,
          riskLevel: prediction.riskLevel,
          prediction,
        };
      });

      const newAlerts: Alert[] = [];
      updatedSilobags.forEach((sb) => {
        if (sb.prediction && sb.prediction.riskLevel !== "safe") {
          const alert = generateAlert(sb, sb.prediction);
          if (alert) newAlerts.push(alert);
        }
      });

      // Keep existing alerts (max 50) + new ones
      const allAlerts = [...newAlerts, ...state.alerts].slice(0, 50);

      return {
        ...state,
        silobags: updatedSilobags,
        alerts: allAlerts,
        lastRefresh: new Date(),
      };
    }

    case "UPDATE_SETTINGS":
      return {
        ...state,
        settings: { ...state.settings, ...action.payload },
      };

    case "ACKNOWLEDGE_ALERT":
      return {
        ...state,
        alerts: state.alerts.map((a) =>
          a.id === action.payload ? { ...a, acknowledged: true } : a
        ),
      };

    case "RESET_DATA": {
      const freshSilobags = generateInitialSilobags();
      const freshAlerts = generateInitialAlerts(freshSilobags);
      return {
        ...state,
        silobags: freshSilobags,
        alerts: freshAlerts,
        lastRefresh: new Date(),
      };
    }

    case "SET_LOADING":
      return { ...state, isLoading: action.payload };

    default:
      return state;
  }
}

// ─── Context ──────────────────────────────────────────────────────────────────

interface SmisiaContextValue {
  state: SmisiaState;
  refreshData: () => void;
  updateSettings: (settings: Partial<AppSettings>) => void;
  acknowledgeAlert: (alertId: string) => void;
  resetData: () => void;
  getSilobag: (id: string) => Silobag | undefined;
  unacknowledgedCount: number;
}

const SmisiaContext = createContext<SmisiaContextValue | null>(null);

const SETTINGS_KEY = "@smisia_settings";

export function SmisiaProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(reducer, {
    silobags: [],
    alerts: [],
    settings: DEFAULT_SETTINGS,
    lastRefresh: null,
    isLoading: true,
  });

  // Load settings from AsyncStorage and initialize data
  useEffect(() => {
    async function init() {
      try {
        const stored = await AsyncStorage.getItem(SETTINGS_KEY);
        const settings: AppSettings = stored ? JSON.parse(stored) : DEFAULT_SETTINGS;
        const silobags = generateInitialSilobags();
        const alerts = generateInitialAlerts(silobags);
        dispatch({ type: "INIT", payload: { silobags, alerts } });
        if (stored) dispatch({ type: "UPDATE_SETTINGS", payload: settings });
      } catch {
        const silobags = generateInitialSilobags();
        const alerts = generateInitialAlerts(silobags);
        dispatch({ type: "INIT", payload: { silobags, alerts } });
      }
    }
    init();
  }, []);

  // Persist settings whenever they change
  useEffect(() => {
    if (!state.isLoading) {
      AsyncStorage.setItem(SETTINGS_KEY, JSON.stringify(state.settings)).catch(() => {});
    }
  }, [state.settings, state.isLoading]);

  const refreshData = useCallback(() => {
    dispatch({ type: "REFRESH_DATA" });
  }, []);

  const updateSettings = useCallback((settings: Partial<AppSettings>) => {
    dispatch({ type: "UPDATE_SETTINGS", payload: settings });
  }, []);

  const acknowledgeAlert = useCallback((alertId: string) => {
    dispatch({ type: "ACKNOWLEDGE_ALERT", payload: alertId });
  }, []);

  const resetData = useCallback(() => {
    dispatch({ type: "RESET_DATA" });
  }, []);

  const getSilobag = useCallback(
    (id: string) => state.silobags.find((sb) => sb.id === id),
    [state.silobags]
  );

  const unacknowledgedCount = state.alerts.filter((a) => !a.acknowledged).length;

  return (
    <SmisiaContext.Provider
      value={{ state, refreshData, updateSettings, acknowledgeAlert, resetData, getSilobag, unacknowledgedCount }}
    >
      {children}
    </SmisiaContext.Provider>
  );
}

export function useSmisia(): SmisiaContextValue {
  const ctx = useContext(SmisiaContext);
  if (!ctx) throw new Error("useSmisia must be used within SmisiaProvider");
  return ctx;
}
