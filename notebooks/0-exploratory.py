# ---
# SMISIA — Notebook Exploratorio (EDA)
# Convertir a .ipynb con: jupytext --to ipynb notebooks/0-exploratory.py
# ---

# %% [markdown]
# # SMISIA — Análisis Exploratorio de Datos (EDA)
# 
# Este notebook analiza el dataset sintético de silobolsas.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(".")), ""))
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# %% [markdown]
# ## 1. Carga de datos

# %%
df = pd.read_csv("data/synthetic_silo_dataset.csv")
print(f"Shape: {df.shape}")
print(f"Silos: {df['silo_id'].nunique()}")
print(f"Rango temporal: {df['timestamp'].min()} — {df['timestamp'].max()}")
df.head()

# %% [markdown]
# ## 2. Distribución de labels

# %%
fig, ax = plt.subplots(figsize=(8, 5))
colors = {"bien": "#2ecc71", "tolerable": "#f39c12", "problema": "#e74c3c", "critico": "#8e44ad"}
label_counts = df["label"].value_counts()
label_counts.plot(kind="bar", color=[colors.get(l, "#95a5a6") for l in label_counts.index], ax=ax)
ax.set_title("Distribución de Labels", fontsize=14)
ax.set_ylabel("Cantidad de registros")
for i, (label, count) in enumerate(label_counts.items()):
    ax.text(i, count + 200, f"{100*count/len(df):.1f}%", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("docs/label_distribution.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. Distribuciones de sensores

# %%
sensor_cols = ["temperature_c", "humidity_pct", "co2_ppm", "nh3_ppm", "battery_pct"]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, col in enumerate(sensor_cols):
    ax = axes[i // 3, i % 3]
    df[col].dropna().hist(bins=50, ax=ax, alpha=0.7, color=sns.color_palette("husl")[i])
    ax.set_title(col, fontsize=12)
    ax.set_xlabel("")
axes[1, 2].axis("off")
plt.suptitle("Distribuciones de Sensores", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig("docs/sensor_distributions.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4. Correlaciones

# %%
corr = df[sensor_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
ax.set_title("Matriz de Correlación", fontsize=14)
plt.tight_layout()
plt.savefig("docs/correlation_matrix.png", dpi=150)
plt.show()

# %% [markdown]
# ## 5. Evolución temporal por silo (ejemplo)

# %%
example_silo = df["silo_id"].unique()[5]  # Tomar un silo con variación
silo_data = df[df["silo_id"] == example_silo].sort_values("timestamp")
silo_data["timestamp"] = pd.to_datetime(silo_data["timestamp"])

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
axes[0].plot(silo_data["timestamp"], silo_data["temperature_c"], color="#e74c3c", alpha=0.7)
axes[0].set_ylabel("Temperatura (°C)")
axes[0].set_title(f"Evolución temporal — {example_silo}", fontsize=14)

axes[1].plot(silo_data["timestamp"], silo_data["humidity_pct"], color="#3498db", alpha=0.7)
axes[1].set_ylabel("Humedad (%)")

axes[2].plot(silo_data["timestamp"], silo_data["co2_ppm"], color="#2ecc71", alpha=0.7)
axes[2].set_ylabel("CO₂ (ppm)")
axes[2].set_xlabel("Fecha")

plt.tight_layout()
plt.savefig("docs/temporal_evolution.png", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Missing values

# %%
missing = df[sensor_cols].isnull().mean() * 100
print("Porcentaje de valores faltantes:")
print(missing.round(2))

# %% [markdown]
# ## 7. Sensores por label

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, col in enumerate(["temperature_c", "humidity_pct", "co2_ppm"]):
    df.boxplot(column=col, by="label", ax=axes[i])
    axes[i].set_title(col, fontsize=12)
    axes[i].set_xlabel("Label")
plt.suptitle("Distribución por Label", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("docs/boxplot_by_label.png", dpi=150)
plt.show()

# %% [markdown]
# ## Conclusiones
# 
# - El dataset contiene ~72k registros de 50 silobolsas
# - Distribución de labels: ~70% bien, ~15% tolerable, ~10% problema, ~5% crítico
# - Se observan patrones de deterioro realistas en los silos con escenarios de riesgo
# - Valores faltantes inyectados al ~3%
# - Correlación entre humedad, CO₂ y temperatura en escenarios de deterioro
