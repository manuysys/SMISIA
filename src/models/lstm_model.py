"""
SMISIA — Modelo LSTM/GRU (Fase B)
Predicción temporal de riesgo para silobolsas.
"""

import logging
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib

logger = logging.getLogger("smisia.lstm")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SiloSequenceDataset(Dataset):
    """Dataset de secuencias temporales por silo."""

    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray = None,
    ):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets) if targets is not None else None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.sequences[idx], self.targets[idx]
        return self.sequences[idx]


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
class SiloLSTM(nn.Module):
    """LSTM bidireccional para predicción de riesgo temporal."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list = None,
        dropout: float = 0.2,
        bidirectional: bool = True,
        n_horizons: int = 3,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        self.bidirectional = bidirectional
        dir_factor = 2 if bidirectional else 1

        # Capa LSTM 1
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_sizes[0],
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if len(hidden_sizes) > 1 else 0,
        )

        # Capa LSTM 2
        self.lstm2 = nn.LSTM(
            input_size=hidden_sizes[0] * dir_factor,
            hidden_size=hidden_sizes[1],
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)

        # Cabeza de clasificación: probabilidad de "problema"/"critico"
        # para cada horizonte (1d, 3d, 7d)
        self.fc = nn.Linear(hidden_sizes[1] * dir_factor, n_horizons)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = self.dropout(out)

        # Tomar último timestep
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out


# ---------------------------------------------------------------------------
# Preparación de secuencias
# ---------------------------------------------------------------------------
def prepare_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    seq_length: int = 360,
    horizons_days: list = None,
    readings_per_day: int = 12,
) -> tuple:
    """
    Prepara secuencias temporales por silo para LSTM.

    Returns:
        (sequences, targets, metadata)
    """
    if horizons_days is None:
        horizons_days = [1, 3, 7]

    label_map = {"bien": 0, "tolerable": 0, "problema": 1, "critico": 1}
    sequences = []
    targets = []
    meta = []

    for silo_id, group in df.groupby("silo_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        values = group[feature_cols].values.astype(np.float32)
        values = np.nan_to_num(values, nan=0.0)

        # Normalizar por silo
        means = values.mean(axis=0)
        stds = values.std(axis=0)
        stds[stds == 0] = 1.0
        values = (values - means) / stds

        labels = group["label"].map(label_map).fillna(0).values

        for i in range(seq_length, len(group) - max(horizons_days) * readings_per_day):
            seq = values[i - seq_length : i]
            target = []
            for h_days in horizons_days:
                future_idx = i + h_days * readings_per_day
                if future_idx < len(labels):
                    # ¿Hay evento problema/critico en la ventana futura?
                    window = labels[i:future_idx]
                    target.append(float(window.max()))
                else:
                    target.append(0.0)

            sequences.append(seq)
            targets.append(target)
            meta.append(
                {
                    "silo_id": silo_id,
                    "timestamp": group.iloc[i]["timestamp"],
                }
            )

    if not sequences:
        return np.array([]), np.array([]), []

    return np.array(sequences), np.array(targets), meta


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def train_lstm(
    df: pd.DataFrame,
    feature_cols: list,
    config: dict,
) -> dict:
    """Entrena el modelo LSTM."""
    lstm_cfg = config["lstm"]
    seed = config["project"]["random_seed"]
    torch.manual_seed(seed)

    logger.info("Preparando secuencias LSTM...")
    seq_length = min(lstm_cfg["sequence_length_hours"] // 2, 360)
    horizons = lstm_cfg["prediction_horizons_days"]

    sequences, targets, meta = prepare_sequences(
        df,
        feature_cols,
        seq_length=seq_length,
        horizons_days=horizons,
    )

    if len(sequences) == 0:
        logger.warning("No hay suficientes secuencias para LSTM")
        return {"model": None, "trained": False}

    logger.info(f"Secuencias: {len(sequences)}, Shape: {sequences.shape}")

    # Split temporal (80/20)
    split_idx = int(0.8 * len(sequences))
    train_dataset = SiloSequenceDataset(sequences[:split_idx], targets[:split_idx])
    val_dataset = SiloSequenceDataset(sequences[split_idx:], targets[split_idx:])

    train_loader = DataLoader(
        train_dataset, batch_size=lstm_cfg["batch_size"], shuffle=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=lstm_cfg["batch_size"], shuffle=False
    )

    # Modelo
    input_size = sequences.shape[2]
    model = SiloLSTM(
        input_size=input_size,
        hidden_sizes=lstm_cfg["hidden_units"],
        dropout=lstm_cfg["dropout"],
        bidirectional=lstm_cfg["bidirectional"],
        n_horizons=len(horizons),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(lstm_cfg["epochs"]):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                output = model(batch_x)
                loss = criterion(output, batch_y)
                val_loss += loss.item()

        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= lstm_cfg["early_stopping_patience"]:
                logger.info(f"Early stopping en epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "trained": True,
        "input_size": input_size,
        "seq_length": seq_length,
        "horizons": horizons,
        "best_val_loss": best_val_loss,
        "feature_cols": feature_cols,
    }


def predict_lstm_with_mc_dropout(
    model: SiloLSTM,
    sequence: np.ndarray,
    n_samples: int = 20,
) -> dict:
    """
    Predicción con MC Dropout para estimación de incertidumbre.
    """
    model.train()  # Mantener dropout activo
    sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(DEVICE)

    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            # Forward con dropout activo
            model.lstm1.dropout = 0.2  # Ensure dropout is active
            output = model(sequence_tensor)
            predictions.append(output.cpu().numpy()[0])

    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)

    return {
        "probabilities": mean_pred.tolist(),
        "uncertainty_std": std_pred.tolist(),
    }


def save_lstm_model(result: dict, models_dir: str = "models"):
    """Guarda modelo LSTM."""
    os.makedirs(models_dir, exist_ok=True)
    if result.get("model") is not None:
        torch.save(
            result["model"].state_dict(),
            os.path.join(models_dir, "lstm_model.pt"),
        )
        joblib.dump(
            {k: v for k, v in result.items() if k != "model"},
            os.path.join(models_dir, "lstm_metadata.joblib"),
        )
        logger.info(f"Modelo LSTM guardado en {models_dir}/")
