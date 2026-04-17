"""
AGRILION — Visualización Profesional
=======================================

Gráficos de alta calidad para análisis de series temporales:
- Series temporales con subplots por sensor
- Predicciones LSTM vs valores reales
- Anomalías marcadas sobre series temporales
- Timeline de riesgo con zonas de color
- Historial de entrenamiento del modelo
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend no interactivo para servidores
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

from .config import PLOT_CONFIG, SENSOR_COLUMNS, RISK_CONFIG, OUTPUTS_DIR

logger = logging.getLogger(__name__)


def _setup_style():
    """Configura el estilo profesional de los gráficos."""
    plt.style.use(PLOT_CONFIG["style"])
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#0f3460",
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.color": "#e94560",
        "text.color": "#e0e0e0",
        "xtick.color": "#a0a0a0",
        "ytick.color": "#a0a0a0",
    })


def plot_timeseries(
    df: pd.DataFrame,
    columns: List[str] = None,
    title: str = "AGRILION — Series Temporales de Sensores",
    save_path: str = None,
    show: bool = False,
) -> str:
    """
    Gráfico de series temporales con subplots por sensor.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con DatetimeIndex y datos de sensores.
    columns : list of str
        Columnas a graficar.
    title : str
        Título del gráfico.
    save_path : str, optional
        Ruta para guardar. Default: outputs/timeseries.png.
    show : bool
        Si True, muestra el gráfico en pantalla.

    Returns
    -------
    str
        Ruta donde se guardó el gráfico.
    """
    _setup_style()
    columns = columns or [c for c in SENSOR_COLUMNS if c in df.columns]
    colors = PLOT_CONFIG["color_palette"]

    n_cols = len(columns)
    fig, axes = plt.subplots(n_cols, 1, figsize=PLOT_CONFIG["figsize_timeseries"],
                              sharex=True)

    if n_cols == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=16, fontweight="bold", color="#e94560", y=0.98)

    units = {"temperature": "°C", "humidity": "%", "co2": "ppm"}

    for i, col in enumerate(columns):
        ax = axes[i]
        color = colors.get(col, "#74b9ff")

        # Línea principal con gradiente
        ax.plot(df.index, df[col], color=color, linewidth=1.2, alpha=0.9, label=col)

        # Relleno con gradiente sutil
        ax.fill_between(df.index, df[col].min() * 0.95, df[col],
                        alpha=0.1, color=color)

        # Etiquetas
        unit = units.get(col, "")
        ax.set_ylabel(f"{col.capitalize()} ({unit})", color=color, fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.3)

        # Estadísticas en el gráfico
        mean_val = df[col].mean()
        ax.axhline(y=mean_val, color=color, linestyle="--", alpha=0.3, linewidth=0.8)
        ax.text(df.index[0], mean_val, f"  μ={mean_val:.1f}",
                color=color, alpha=0.6, fontsize=9, va="bottom")

    axes[-1].set_xlabel("Tiempo", fontweight="bold")

    # Formato de fechas
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Guardar
    save_path = save_path or str(OUTPUTS_DIR / "timeseries.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight",
                facecolor=fig.get_facecolor())

    if show:
        plt.show()
    plt.close(fig)

    logger.info(f"📊 Gráfico de series temporales guardado: {save_path}")
    return save_path


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex = None,
    feature_names: List[str] = None,
    title: str = "AGRILION — Predicciones LSTM vs Valores Reales",
    save_path: str = None,
    show: bool = False,
) -> str:
    """
    Gráfico de predicciones superpuestas sobre valores reales.

    Parameters
    ----------
    y_true : np.ndarray
        Valores reales.
    y_pred : np.ndarray
        Predicciones del modelo.
    timestamps : pd.DatetimeIndex
        Timestamps para el eje X.
    feature_names : list of str
        Nombres de las features.
    title : str
        Título del gráfico.
    save_path : str
        Ruta para guardar.
    show : bool
        Si True, muestra el gráfico.

    Returns
    -------
    str
        Ruta donde se guardó el gráfico.
    """
    _setup_style()

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    feature_names = feature_names or SENSOR_COLUMNS
    n_features = min(y_true.shape[1], y_pred.shape[1], len(feature_names))
    colors = PLOT_CONFIG["color_palette"]

    fig, axes = plt.subplots(n_features, 1,
                              figsize=(PLOT_CONFIG["figsize_predictions"][0],
                                       PLOT_CONFIG["figsize_predictions"][1] * n_features / 2),
                              sharex=True)

    if n_features == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=16, fontweight="bold", color="#e94560", y=0.98)

    x_axis = timestamps[:len(y_true)] if timestamps is not None else range(len(y_true))

    for i in range(n_features):
        ax = axes[i]
        name = feature_names[i]
        color_actual = colors.get(name, "#ff6b6b")
        color_pred = colors.get("prediction", "#74b9ff")

        # Valores reales
        ax.plot(x_axis, y_true[:, i], color=color_actual, linewidth=1.2,
                alpha=0.8, label=f"Real ({name})")

        # Predicciones
        ax.plot(x_axis, y_pred[:, i], color=color_pred, linewidth=1.2,
                alpha=0.8, linestyle="--", label=f"Predicción ({name})")

        # Banda de error
        error = np.abs(y_true[:, i] - y_pred[:, i])
        ax.fill_between(x_axis, y_true[:, i] - error, y_true[:, i] + error,
                        alpha=0.1, color=color_pred)

        ax.set_ylabel(name.capitalize(), fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.3)

        # Métricas en el gráfico
        from sklearn.metrics import mean_absolute_error, r2_score
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        ax.text(0.02, 0.95, f"MAE={mae:.2f}  R²={r2:.3f}",
                transform=ax.transAxes, fontsize=9, color="#ffd93d",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#16213e",
                          edgecolor="#ffd93d", alpha=0.8))

    if timestamps is not None:
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
        plt.xticks(rotation=45)

    axes[-1].set_xlabel("Tiempo", fontweight="bold")
    plt.tight_layout()

    save_path = save_path or str(OUTPUTS_DIR / "predictions_vs_actual.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight",
                facecolor=fig.get_facecolor())

    if show:
        plt.show()
    plt.close(fig)

    logger.info(f"📊 Gráfico de predicciones guardado: {save_path}")
    return save_path


def plot_anomalies(
    df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    columns: List[str] = None,
    title: str = "AGRILION — Detección de Anomalías",
    save_path: str = None,
    show: bool = False,
) -> str:
    """
    Gráfico de series temporales con anomalías marcadas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos de sensores.
    anomaly_df : pd.DataFrame
        DataFrame con columnas de anomalía (is_anomaly o *_anomaly_consensus).
    columns : list of str
        Columnas a graficar.
    title : str
        Título del gráfico.
    save_path : str
        Ruta para guardar.
    show : bool
        Si True, muestra el gráfico.

    Returns
    -------
    str
        Ruta donde se guardó el gráfico.
    """
    _setup_style()
    columns = columns or [c for c in SENSOR_COLUMNS if c in df.columns]
    colors = PLOT_CONFIG["color_palette"]

    n_cols = len(columns)
    fig, axes = plt.subplots(n_cols, 1, figsize=PLOT_CONFIG["figsize_anomalies"],
                              sharex=True)

    if n_cols == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=16, fontweight="bold", color="#e94560", y=0.98)

    for i, col in enumerate(columns):
        ax = axes[i]
        color = colors.get(col, "#74b9ff")
        anomaly_color = colors.get("anomaly", "#e74c3c")

        # Serie temporal normal
        ax.plot(df.index, df[col], color=color, linewidth=1.0, alpha=0.7)

        # Determinar columna de anomalía
        anomaly_col = None
        candidates = [
            f"{col}_anomaly_consensus",
            "is_anomaly",
        ]
        for candidate in candidates:
            if candidate in anomaly_df.columns:
                anomaly_col = candidate
                break

        if anomaly_col is not None:
            # Alinear índices
            aligned_anomaly = anomaly_df[anomaly_col].reindex(df.index, fill_value=False)
            anomaly_mask = aligned_anomaly.astype(bool)

            if anomaly_mask.any():
                # Marcar anomalías con puntos grandes rojos
                ax.scatter(
                    df.index[anomaly_mask],
                    df[col][anomaly_mask],
                    color=anomaly_color,
                    s=50,
                    zorder=5,
                    alpha=0.8,
                    label="Anomalía",
                    edgecolors="white",
                    linewidths=0.5,
                )

                # Franjas verticales en zonas de anomalía
                for idx in df.index[anomaly_mask]:
                    ax.axvline(x=idx, color=anomaly_color, alpha=0.05, linewidth=3)

        ax.set_ylabel(f"{col.capitalize()}", color=color, fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.3)

    # Conteo de anomalías
    if "is_anomaly" in anomaly_df.columns:
        n_anom = anomaly_df["is_anomaly"].sum()
        total = len(anomaly_df)
        fig.text(0.99, 0.01,
                 f"Anomalías: {n_anom}/{total} ({n_anom/total*100:.1f}%)",
                 ha="right", va="bottom", fontsize=10, color="#e94560",
                 style="italic")

    axes[-1].set_xlabel("Tiempo", fontweight="bold")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()

    save_path = save_path or str(OUTPUTS_DIR / "anomalies.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight",
                facecolor=fig.get_facecolor())

    if show:
        plt.show()
    plt.close(fig)

    logger.info(f"📊 Gráfico de anomalías guardado: {save_path}")
    return save_path


def plot_risk_timeline(
    df: pd.DataFrame,
    title: str = "AGRILION — Timeline de Riesgo",
    save_path: str = None,
    show: bool = False,
) -> str:
    """
    Timeline de riesgo con zonas de color por nivel.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas risk_score y risk_level.
    title : str
        Título del gráfico.
    save_path : str
        Ruta para guardar.
    show : bool
        Si True, muestra el gráfico.

    Returns
    -------
    str
        Ruta donde se guardó el gráfico.
    """
    _setup_style()
    colors = PLOT_CONFIG["color_palette"]

    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize_risk"])
    fig.suptitle(title, fontsize=16, fontweight="bold", color="#e94560")

    timestamps = df.index
    scores = df["risk_score"].values

    # Zonas de fondo por nivel
    ax.axhspan(0, 30, alpha=0.08, color=colors["normal"], label="NORMAL (0-30)")
    ax.axhspan(30, 70, alpha=0.08, color=colors["warning"], label="WARNING (30-70)")
    ax.axhspan(70, 100, alpha=0.08, color=colors["critical"], label="CRITICAL (70-100)")

    # Líneas de separación
    ax.axhline(y=30, color=colors["warning"], linestyle=":", alpha=0.5, linewidth=1)
    ax.axhline(y=70, color=colors["critical"], linestyle=":", alpha=0.5, linewidth=1)

    # Score como línea coloreada
    for j in range(len(timestamps) - 1):
        score = scores[j]
        if score < 30:
            c = colors["normal"]
        elif score < 70:
            c = colors["warning"]
        else:
            c = colors["critical"]
        ax.plot(timestamps[j:j+2], scores[j:j+2], color=c, linewidth=2, alpha=0.8)

    # Relleno bajo la curva
    ax.fill_between(timestamps, 0, scores, alpha=0.15, color="#e94560")

    ax.set_ylabel("Score de Riesgo", fontweight="bold")
    ax.set_xlabel("Tiempo", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", framealpha=0.3)

    # Formato de fechas
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    plt.xticks(rotation=45)

    # Estadísticas
    avg_score = np.mean(scores)
    max_score = np.max(scores)
    ax.text(0.98, 0.95,
            f"Score promedio: {avg_score:.1f}\nScore máximo: {max_score:.0f}",
            transform=ax.transAxes, fontsize=10, color="#ffd93d",
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#16213e",
                      edgecolor="#ffd93d", alpha=0.8))

    plt.tight_layout()

    save_path = save_path or str(OUTPUTS_DIR / "risk_timeline.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight",
                facecolor=fig.get_facecolor())

    if show:
        plt.show()
    plt.close(fig)

    logger.info(f"📊 Gráfico de riesgo guardado: {save_path}")
    return save_path


def plot_training_history(
    history: Dict,
    title: str = "AGRILION — Historial de Entrenamiento LSTM",
    save_path: str = None,
    show: bool = False,
) -> str:
    """
    Gráfico del historial de entrenamiento del modelo LSTM.

    Parameters
    ----------
    history : dict
        Historial de Keras con keys: loss, val_loss, mae, val_mae.
    title : str
        Título del gráfico.
    save_path : str
        Ruta para guardar.
    show : bool
        Si True, muestra el gráfico.

    Returns
    -------
    str
        Ruta donde se guardó el gráfico.
    """
    _setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=16, fontweight="bold", color="#e94560")

    epochs = range(1, len(history["loss"]) + 1)

    # Loss
    ax1 = axes[0]
    ax1.plot(epochs, history["loss"], color="#ff6b6b", linewidth=2, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], color="#74b9ff", linewidth=2,
             linestyle="--", label="Val Loss")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss (MSE)")
    ax1.set_title("Función de Pérdida")
    ax1.legend(framealpha=0.3)

    # Marcar mejor época
    best_epoch = np.argmin(history["val_loss"]) + 1
    best_val_loss = min(history["val_loss"])
    ax1.axvline(x=best_epoch, color="#ffd93d", linestyle=":", alpha=0.7)
    ax1.scatter([best_epoch], [best_val_loss], color="#ffd93d", s=80, zorder=5,
                marker="*", label=f"Mejor: epoch {best_epoch}")
    ax1.legend(framealpha=0.3)

    # MAE
    if "mae" in history:
        ax2 = axes[1]
        ax2.plot(epochs, history["mae"], color="#ff6b6b", linewidth=2, label="Train MAE")
        if "val_mae" in history:
            ax2.plot(epochs, history["val_mae"], color="#74b9ff", linewidth=2,
                     linestyle="--", label="Val MAE")
        ax2.set_xlabel("Época")
        ax2.set_ylabel("MAE")
        ax2.set_title("Mean Absolute Error")
        ax2.legend(framealpha=0.3)
    else:
        axes[1].set_visible(False)

    plt.tight_layout()

    save_path = save_path or str(OUTPUTS_DIR / "training_history.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight",
                facecolor=fig.get_facecolor())

    if show:
        plt.show()
    plt.close(fig)

    logger.info(f"📊 Gráfico de entrenamiento guardado: {save_path}")
    return save_path


def generate_all_plots(
    df: pd.DataFrame,
    anomaly_df: pd.DataFrame = None,
    risk_df: pd.DataFrame = None,
    y_true: np.ndarray = None,
    y_pred: np.ndarray = None,
    timestamps: pd.DatetimeIndex = None,
    training_history: Dict = None,
    show: bool = False,
) -> Dict[str, str]:
    """
    Genera todos los gráficos del sistema.

    Parameters
    ----------
    df : pd.DataFrame
        Datos de sensores.
    anomaly_df : pd.DataFrame
        Resultados de detección de anomalías.
    risk_df : pd.DataFrame
        Datos con scores de riesgo.
    y_true, y_pred : np.ndarray
        Valores reales y predichos.
    timestamps : pd.DatetimeIndex
        Timestamps para predicciones.
    training_history : dict
        Historial de entrenamiento.
    show : bool
        Si True, muestra todos los gráficos.

    Returns
    -------
    dict
        Paths de todos los gráficos generados.
    """
    paths = {}

    logger.info("\n📊 Generando todos los gráficos...")

    # 1. Series temporales
    paths["timeseries"] = plot_timeseries(df, show=show)

    # 2. Anomalías
    if anomaly_df is not None:
        paths["anomalies"] = plot_anomalies(df, anomaly_df, show=show)

    # 3. Predicciones vs reales
    if y_true is not None and y_pred is not None:
        paths["predictions"] = plot_predictions_vs_actual(
            y_true, y_pred, timestamps=timestamps, show=show
        )

    # 4. Timeline de riesgo
    if risk_df is not None and "risk_score" in risk_df.columns:
        paths["risk"] = plot_risk_timeline(risk_df, show=show)

    # 5. Historial de entrenamiento
    if training_history is not None:
        paths["training"] = plot_training_history(training_history, show=show)

    logger.info(f"✅ {len(paths)} gráficos generados en: {OUTPUTS_DIR}")
    return paths
