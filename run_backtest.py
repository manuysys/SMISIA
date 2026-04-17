"""
AGRILION — Run Backtesting
============================
Usage:
    python run_backtest.py
    python run_backtest.py --data data/sensor_data.csv --silo SILO_001
"""

import argparse
import sys
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_and_prepare
from src.preprocessing import DataPreprocessor
from src.lstm_model import AgrilionLSTM
from src.risk_engine import RiskEngine
from src.alerts import AlertSystem
from src.config import DEFAULT_MODEL_PATH, SCALER_PATH, SENSOR_COLUMNS, LSTM_CONFIG
from evaluation.backtest import BacktestEngine, print_backtest_report
from evaluation.report_generator import ReportGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--silo", default="SILO_001")
    parser.add_argument("--threshold", type=int, default=70)
    parser.add_argument("--save-report", action="store_true")
    args = parser.parse_args()

    # Load data
    data_path = args.data or "data/sensor_data.csv"
    print(f"Loading data from: {data_path}")
    df = load_and_prepare(data_path)

    # Load model
    preprocessor = DataPreprocessor()
    lstm = AgrilionLSTM(n_features=len(SENSOR_COLUMNS))

    if not Path(DEFAULT_MODEL_PATH).exists():
        print("No trained model found. Train the model first with: python main.py")
        sys.exit(1)

    lstm.load()
    preprocessor.load_scaler()

    risk_engine = RiskEngine()
    alert_system = AlertSystem()

    # Run backtest
    engine = BacktestEngine(preprocessor, lstm, risk_engine, alert_system)
    report = engine.run(
        df,
        seq_length=LSTM_CONFIG["sequence_length"],
        risk_threshold=args.threshold,
        silo_id=args.silo,
    )

    print_backtest_report(report)

    # Generate quality report
    if args.save_report:
        gen = ReportGenerator()
        quality = gen.generate_from_backtest(report, save=True)
        gen.print_summary(quality)

    return report


if __name__ == "__main__":
    main()
