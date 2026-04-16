// Icon mapping: SF Symbols (iOS) → Material Icons (Android/Web)
// Add new mappings here before using in tabs or screens

import MaterialIcons from "@expo/vector-icons/MaterialIcons";
import { SymbolWeight, SymbolViewProps } from "expo-symbols";
import { ComponentProps } from "react";
import { OpaqueColorValue, type StyleProp, type TextStyle } from "react-native";

type IconMapping = Record<SymbolViewProps["name"], ComponentProps<typeof MaterialIcons>["name"]>;
type IconSymbolName = keyof typeof MAPPING;

const MAPPING = {
  // Navigation
  "house.fill":                            "home",
  "bell.fill":                             "notifications",
  "map.fill":                              "map",
  "gearshape.fill":                        "settings",
  // General
  "paperplane.fill":                       "send",
  "chevron.left.forwardslash.chevron.right": "code",
  "chevron.right":                         "chevron-right",
  "chevron.left":                          "chevron-left",
  "xmark":                                 "close",
  "checkmark":                             "check",
  "checkmark.circle.fill":                 "check-circle",
  "exclamationmark.triangle.fill":         "warning",
  "exclamationmark.circle.fill":           "error",
  "info.circle.fill":                      "info",
  // Sensors
  "thermometer":                           "thermostat",
  "drop.fill":                             "water-drop",
  "wind":                                  "air",
  // Data
  "chart.line.uptrend.xyaxis":             "show-chart",
  "clock.fill":                            "schedule",
  "arrow.clockwise":                       "refresh",
  "location.fill":                         "location-on",
  // Status
  "shield.fill":                           "shield",
  "antenna.radiowaves.left.and.right":     "wifi",
} as IconMapping;

export function IconSymbol({
  name,
  size = 24,
  color,
  style,
}: {
  name: IconSymbolName;
  size?: number;
  color: string | OpaqueColorValue;
  style?: StyleProp<TextStyle>;
  weight?: SymbolWeight;
}) {
  return <MaterialIcons color={color} size={size} name={MAPPING[name]} style={style} />;
}
