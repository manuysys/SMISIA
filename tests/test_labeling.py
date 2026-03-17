"""
SMISIA — Tests de Labeling
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def make_df_for_labeling(n=100):
    """Crea DataFrame para testing de labels."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "silo_id": "TEST_001",
        "temperature_c": rng.normal(25, 3, n),
        "humidity_pct": rng.normal(14, 2, n),
        "co2_ppm": rng.normal(500, 50, n),
        "humidity_pct_24h_slope": rng.normal(0, 0.1, n),
    })


class TestHeuristicLabeling:
    def test_labels_assigned(self):
        from src.labeling.heuristic import apply_heuristic_labels
        df = make_df_for_labeling()
        result = apply_heuristic_labels(df)
        assert "heuristic_label" in result.columns
        assert result["heuristic_label"].isin(
            ["bien", "tolerable", "problema", "critico"]
        ).all()

    def test_critical_detection(self):
        from src.labeling.heuristic import apply_heuristic_labels
        df = make_df_for_labeling(10)
        # Forzar condiciones críticas
        df.loc[0, "humidity_pct"] = 30
        df.loc[0, "co2_ppm"] = 3000
        df.loc[0, "humidity_pct_24h_slope"] = 0.5  # 0.5 * 24 = 12 > 3
        result = apply_heuristic_labels(df)
        assert result.loc[0, "heuristic_label"] == "critico"

    def test_label_encoding(self):
        from src.labeling.heuristic import encode_labels, decode_labels
        labels = pd.Series(["bien", "problema", "critico", "tolerable"])
        encoded, mapping = encode_labels(labels)
        assert encoded.tolist() == [0, 2, 3, 1]
        decoded = decode_labels(encoded.values, mapping)
        assert decoded.tolist() == labels.tolist()

    def test_active_learning_selection(self):
        from src.labeling.heuristic import select_for_active_learning
        df = make_df_for_labeling(200)
        rng = np.random.default_rng(42)
        probs = rng.dirichlet([1, 1, 1, 1], size=200)
        selected = select_for_active_learning(df, probs, top_n=20)
        assert len(selected) == 20
        assert "model_confidence" in selected.columns
        assert "needs_review" in selected.columns
