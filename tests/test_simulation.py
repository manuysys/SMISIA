"""
AGRILION — Simulation Tests
==============================
Validates the full pipeline against 4 synthetic risk scenarios.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import json
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ─── Scenario Generators ──────────────────────────────────────────────────────

def _base_ts(n: int = 72, freq: str = "1h") -> pd.DatetimeIndex:
    return pd.date_range("2025-06-01", periods=n, freq=freq, tz="UTC")


def scenario_normal(n: int = 72) -> pd.DataFrame:
    """Normal grain storage conditions."""
    rng = np.random.default_rng(1)
    ts = _base_ts(n)
    return pd.DataFrame({
        "temperature": rng.normal(22, 1, n),
        "humidity": rng.normal(55, 2, n),
        "co2": rng.normal(400, 15, n),
    }, index=ts)


def scenario_gradual_temp_rise(n: int = 72) -> pd.DataFrame:
    """Temperature increases gradually — risk scenario for heat damage."""
    rng = np.random.default_rng(2)
    ts = _base_ts(n)
    temp = np.linspace(22, 38, n) + rng.normal(0, 0.5, n)  # 22→38°C over 72h
    return pd.DataFrame({
        "temperature": temp,
        "humidity": rng.normal(58, 2, n),
        "co2": rng.normal(420, 20, n),
    }, index=ts)


def scenario_co2_spike(n: int = 72) -> pd.DataFrame:
    """CO2 spike simulating biological activity / fermentation."""
    rng = np.random.default_rng(3)
    ts = _base_ts(n)
    co2 = np.full(n, 420.0)
    co2[40:55] = np.linspace(420, 1200, 15)  # spike at step 40
    co2[55:] = np.linspace(1200, 900, n - 55)
    return pd.DataFrame({
        "temperature": rng.normal(25, 1, n),
        "humidity": rng.normal(60, 2, n),
        "co2": co2 + rng.normal(0, 10, n),
    }, index=ts)


def scenario_combined_risk(n: int = 72) -> pd.DataFrame:
    """High humidity + high temperature sustained — worst case."""
    rng = np.random.default_rng(4)
    ts = _base_ts(n)
    temp = rng.normal(34, 1, n)   # sustained high temp
    hum = rng.normal(82, 2, n)    # sustained high humidity
    co2 = np.linspace(400, 800, n) + rng.normal(0, 20, n)
    return pd.DataFrame({
        "temperature": temp,
        "humidity": hum,
        "co2": co2,
    }, index=ts)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def risk_engine():
    from src.risk_engine import RiskEngine
    return RiskEngine()


@pytest.fixture(scope="module")
def alert_system():
    from src.alerts import AlertSystem
    return AlertSystem()


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestScenarioNormal:
    def test_risk_stays_low(self, risk_engine):
        df = scenario_normal()
        scores = []
        for _, row in df.iterrows():
            s = risk_engine.calculate_risk_score(row.to_dict())
            scores.append(s)
        avg = np.mean(scores)
        assert avg < 40, f"Normal scenario avg score {avg:.1f} should be < 40"

    def test_no_critical_alerts(self, risk_engine, alert_system):
        df = scenario_normal()
        critical_count = 0
        for _, row in df.iterrows():
            factors = risk_engine.get_risk_factors(row.to_dict())
            alerts = alert_system.generate_alerts(factors)
            critical_count += sum(1 for a in alerts if a.level == "CRITICAL")
        assert critical_count == 0, f"Normal scenario triggered {critical_count} CRITICAL alerts"


class TestScenarioGradualTempRise:
    def test_risk_increases_over_time(self, risk_engine):
        df = scenario_gradual_temp_rise()
        scores = [risk_engine.calculate_risk_score(row.to_dict()) for _, row in df.iterrows()]
        first_half_avg = np.mean(scores[:36])
        second_half_avg = np.mean(scores[36:])
        assert second_half_avg > first_half_avg, \
            f"Risk should increase: {first_half_avg:.1f} → {second_half_avg:.1f}"

    def test_eventually_triggers_warning(self, risk_engine):
        df = scenario_gradual_temp_rise()
        scores = [risk_engine.calculate_risk_score(row.to_dict()) for _, row in df.iterrows()]
        assert max(scores) >= 30, f"Should trigger at least WARNING (max={max(scores)})"

    def test_early_detection(self, risk_engine):
        """System should detect risk BEFORE last 10% of time series."""
        df = scenario_gradual_temp_rise()
        scores = [risk_engine.calculate_risk_score(row.to_dict()) for _, row in df.iterrows()]
        first_warning_step = next((i for i, s in enumerate(scores) if s >= 30), None)
        total = len(scores)
        assert first_warning_step is not None, "Never triggered WARNING"
        assert first_warning_step < int(total * 0.9), \
            f"Warning only at step {first_warning_step}/{total} — too late"


class TestScenarioCO2Spike:
    def test_co2_triggers_alert(self, risk_engine, alert_system):
        df = scenario_co2_spike()
        found_co2_alert = False
        for _, row in df.iterrows():
            factors = risk_engine.get_risk_factors(row.to_dict())
            alerts = alert_system.generate_alerts(factors)
            if any(a.category == "fermentacion" for a in alerts):
                found_co2_alert = True
                break
        assert found_co2_alert, "CO2 spike scenario should trigger fermentation alert"

    def test_max_score_during_spike(self, risk_engine):
        df = scenario_co2_spike()
        scores = [risk_engine.calculate_risk_score(row.to_dict()) for _, row in df.iterrows()]
        spike_scores = scores[40:55]
        assert max(spike_scores) > max(scores[:40]), \
            "Risk during spike should exceed pre-spike risk"


class TestScenarioCombinedRisk:
    def test_sustained_critical(self, risk_engine):
        df = scenario_combined_risk()
        scores = [risk_engine.calculate_risk_score(row.to_dict()) for _, row in df.iterrows()]
        critical_pct = sum(1 for s in scores if s >= 70) / len(scores) * 100
        assert critical_pct > 20, \
            f"Combined risk should have >20% CRITICAL steps, got {critical_pct:.1f}%"

    def test_multiple_alert_categories(self, risk_engine, alert_system):
        df = scenario_combined_risk()
        categories = set()
        for _, row in df.iterrows():
            factors = risk_engine.get_risk_factors(row.to_dict())
            alerts = alert_system.generate_alerts(factors)
            for a in alerts:
                categories.add(a.category)
            if len(categories) >= 2:
                break
        assert len(categories) >= 2, \
            f"Combined scenario should trigger multiple alert types, got: {categories}"

    def test_recommendations_present(self, risk_engine, alert_system):
        df = scenario_combined_risk()
        row = df.iloc[-1]
        factors = risk_engine.get_risk_factors(row.to_dict())
        alerts = alert_system.generate_alerts(factors)
        for a in alerts:
            assert a.recommendation, f"Alert {a.category} missing recommendation"


# ─── Simulation Report ────────────────────────────────────────────────────────

def run_simulation_report(risk_engine, alert_system) -> dict:
    """
    Run all 4 scenarios and return structured evaluation report.
    Useful for calling programmatically without pytest.
    """
    scenarios = {
        "normal": scenario_normal,
        "gradual_temp_rise": scenario_gradual_temp_rise,
        "co2_spike": scenario_co2_spike,
        "combined_risk": scenario_combined_risk,
    }

    report = {"generated_at": datetime.now().isoformat(), "scenarios": {}}

    for name, gen_fn in scenarios.items():
        df = gen_fn()
        scores = []
        alert_counts = {"NORMAL": 0, "WARNING": 0, "CRITICAL": 0}
        first_warning = None
        first_critical = None

        for step, (_, row) in enumerate(df.iterrows()):
            factors = risk_engine.get_risk_factors(row.to_dict())
            score = factors["total_score"]
            level = factors["level"]
            scores.append(score)
            alerts = alert_system.generate_alerts(factors)

            if level == "WARNING" and first_warning is None:
                first_warning = step
            if level == "CRITICAL" and first_critical is None:
                first_critical = step

            for a in alerts:
                alert_counts[a.level] = alert_counts.get(a.level, 0) + 1

        report["scenarios"][name] = {
            "total_steps": len(scores),
            "avg_risk_score": round(float(np.mean(scores)), 1),
            "max_risk_score": int(max(scores)),
            "min_risk_score": int(min(scores)),
            "first_warning_step": first_warning,
            "first_critical_step": first_critical,
            "alert_counts": alert_counts,
            "pct_critical": round(sum(1 for s in scores if s >= 70) / len(scores) * 100, 1),
        }

    return report


if __name__ == "__main__":
    from src.risk_engine import RiskEngine
    from src.alerts import AlertSystem

    rengine = RiskEngine()
    asystem = AlertSystem()
    report = run_simulation_report(rengine, asystem)
    print(json.dumps(report, indent=2))
