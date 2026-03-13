"""
SMISIA — Monitoreo de Deriva (Drift)
Implementa cálculo de PSI (Population Stability Index) y logging de métricas.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("smisia.monitoring")

def compute_psi(expected, actual, buckets=10):
    """
    Calcula el Population Stability Index (PSI) entre dos distribuciones.
    PSI = sum((Actual % - Expected %) * ln(Actual % / Expected %))
    Ranges:
    - < 0.1: No change
    - 0.1 - 0.25: Slight change
    - > 0.25: Significant change (Retrain recommended)
    """
    def scale_range(input_data, min_val, max_val):
        return (input_data - min_val) / (max_val - min_val + 1e-7)

    def sub_compute_psi(e_perc, a_perc):
        if a_perc == 0:
            a_perc = 0.0001
        if e_perc == 0:
            e_perc = 0.0001
        return (a_perc - e_perc) * np.log(a_perc / e_perc)

    # Bucketing
    min_val = min(np.min(expected), np.min(actual))
    max_val = max(np.max(expected), np.max(actual))
    
    e_scaled = scale_range(expected, min_val, max_val)
    a_scaled = scale_range(actual, min_val, max_val)
    
    e_counts = np.histogram(e_scaled, bins=buckets, range=(0, 1))[0]
    a_counts = np.histogram(a_scaled, bins=buckets, range=(0, 1))[0]
    
    e_perc = e_counts / len(expected)
    a_perc = a_counts / len(actual)
    
    psi_val = np.sum([sub_compute_psi(e_perc[i], a_perc[i]) for i in range(len(e_perc))])
    return psi_val

def check_feature_drift(df_baseline, df_actual, feature_cols, threshold=0.25):
    """
    Verifica drift en múltiples features.
    """
    results = {}
    for col in feature_cols:
        if col in df_baseline.columns and col in df_actual.columns:
            psi = compute_psi(df_baseline[col].values, df_actual[col].values)
            results[col] = {
                "psi": round(psi, 4),
                "alert": psi > threshold
            }
    return results

def log_silo_metrics(df, silo_id):
    """
    Calcula y loguea métricas específicas por silo.
    """
    silo_data = df[df["silo_id"] == silo_id]
    if silo_data.empty:
        return None
        
    metrics = {
        "silo_id": silo_id,
        "n_readings": len(silo_data),
        "prediction_dist": silo_data["label"].value_counts(normalize=True).to_dict(),
        "missing_rate": silo_data[["temperature_c", "humidity_pct", "co2_ppm"]].isna().mean().mean(),
        "avg_sensor_health": silo_data["sensor_health"].mean() if "sensor_health" in silo_data.columns else None
    }
    
    logger.info(f"Métricas Silo {silo_id}: {metrics}")
    return metrics


def generate_monitoring_report(psi_results, output_path="models/monitoring_report.html"):
    """
    Genera un Dashboard HTML simple para visualizar el drift.
    """
    rows = ""
    for feat, data in psi_results.items():
        psi = data["psi"]
        alert = data["alert"]
        status_color = "#ef4444" if alert else ("#f59e0b" if psi > 0.1 else "#22c55e")
        status_text = "DANGER" if alert else ("WARNING" if psi > 0.1 else "STABLE")
        
        rows += f"""
        <tr>
            <td>{feat}</td>
            <td style="font-weight:bold;">{psi}</td>
            <td style="background-color:{status_color}; color:white; text-align:center; font-weight:bold; border-radius:4px;">{status_text}</td>
        </tr>
        """
        
    html = f"""
    <html>
    <head>
        <title>SMISIA Monitoring Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0f172a; color: #f8fafc; padding: 40px; line-height: 1.6; }}
            table {{ width: 100%; border-collapse: separate; border-spacing: 0 8px; margin-top: 20px; }}
            th, td {{ padding: 16px; text-align: left; }}
            th {{ background-color: #1e293b; color: #38bdf8; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.1em; border-bottom: 2px solid #334155; }}
            tr {{ background-color: #1e293b; transition: transform 0.2s; }}
            tr:hover {{ transform: scale(1.005); background-color: #334155; }}
            td {{ border-top: 1px solid #334155; border-bottom: 1px solid #334155; }}
            td:first-child {{ border-left: 1px solid #334155; border-top-left-radius: 8px; border-bottom-left-radius: 8px; }}
            td:last-child {{ border-right: 1px solid #334155; border-top-right-radius: 8px; border-bottom-right-radius: 8px; }}
            h1 {{ color: #38bdf8; margin: 0; font-size: 2.5rem; }}
            .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #38bdf8; padding-bottom: 20px; margin-bottom: 30px; }}
            .card {{ background: #1e293b; padding: 25px; border-radius: 12px; margin-top: 30px; border: 1px solid #334155; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }}
            .badge {{ padding: 4px 12px; border-radius: 9999px; font-size: 0.875rem; font-weight: 600; }}
        </style>
    </head>
    <body>
        <div class="header">
            <div>
                <h1>SMISIA</h1>
                <p style="color: #94a3b8; margin: 5px 0 0 0; font-size: 1.1rem;">Drift Monitoring & Population Stability Dashboard</p>
            </div>
            <div style="text-align: right;">
                <span style="color: #38bdf8; font-weight: bold; font-size: 1.2rem;">System Status: OPERATIONAL</span><br/>
                <small style="color: #64748b;">Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
            </div>
        </div>
        
        <table>
            <thead>
                <tr>
                    <th>Feature / Variable Analysis</th>
                    <th>PSI Metric Score</th>
                    <th>Risk Classification</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        
        <div class="card">
            <h3 style="margin-top:0; color: #38bdf8; font-size: 1.5rem;">Maintenance Interpretation Guide</h3>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 20px;">
                <div style="border-left: 4px solid #22c55e; padding-left: 15px;">
                    <h4 style="color: #22c55e; margin: 0 0 10px 0;">STABLE (PSI < 0.1)</h4>
                    <p style="color: #94a3b8; font-size: 0.9rem;">No significant changes. The model remains highly reliable for current data distributions.</p>
                </div>
                <div style="border-left: 4px solid #f59e0b; padding-left: 15px;">
                    <h4 style="color: #f59e0b; margin: 0 0 10px 0;">WARNING (0.1 - 0.25)</h4>
                    <p style="color: #94a3b8; font-size: 0.9rem;">Minor drift detected. Increased monitoring frequency advised. Investigation into sensor health may be needed.</p>
                </div>
                <div style="border-left: 4px solid #ef4444; padding-left: 15px;">
                    <h4 style="color: #ef4444; margin: 0 0 10px 0;">DANGER (PSI > 0.25)</h4>
                    <p style="color: #94a3b8; font-size: 0.9rem;">Significant distribution shift. Potential for model degradation. <b>Automated retraining triggered or manually recommended.</b></p>
                </div>
            </div>
        </div>
        
        <footer style="margin-top: 50px; text-align: center; color: #475569; font-size: 0.875rem; letter-spacing: 0.025em;">
            &copy; 2026 SMISIA - Advanced Silobolsa AI Monitoring System | Phase 2 Execution
        </footer>
    </body>
    </html>
    """
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Monitoring dashboard generated at {output_path}")
