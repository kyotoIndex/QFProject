from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.qf_project.data import download_market_data
from src.qf_project.dataset import build_dataloaders, build_model_inputs
from src.qf_project.evaluate import evaluate_model
from src.qf_project.features import engineer_features
from src.qf_project.model import QuantumFinanceModel
from src.qf_project.quantum_encoding import encode_quantum_states
from src.qf_project.train import train_model
from src.qf_project.backtest import run_backtest
from src.qf_project.utils import create_run_directory, ensure_directory, get_device, load_config, set_seed


def run_pipeline(config_path: str) -> None:
    config = load_config(config_path)
    set_seed(config["project"]["seed"])

    output_root = ensure_directory(config["project"]["output_dir"])
    run_dir = create_run_directory(output_root)
    device = get_device()

    market_data = download_market_data(
        symbols=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
        interval=config["data"]["interval"],
        cache_dir=config["data"]["cache_dir"],
    )

    summary = {}

    for symbol, frame in market_data.items():
        symbol_dir = Path(run_dir) / symbol.replace("^", "")
        symbol_dir.mkdir(parents=True, exist_ok=True)

        feature_frame = engineer_features(
            frame,
            volatility_window=config["features"]["volatility_window"],
            momentum_window=config["features"]["momentum_window"],
            bollinger_window=config["features"]["bollinger_window"],
            rsi_window=config["features"]["rsi_window"],
        )
        quantum_frame = encode_quantum_states(feature_frame, temperature=config["quantum"]["temperature"])
        dataset_frame = pd.concat([feature_frame, quantum_frame], axis=1).dropna().copy()

        inputs = build_model_inputs(
            dataset_frame,
            lookback=config["features"]["lookback"],
            horizon=config["features"]["horizon"],
            train_ratio=config["features"]["train_ratio"],
            val_ratio=config["features"]["val_ratio"],
        )
        dataloaders = build_dataloaders(inputs["splits"], batch_size=config["training"]["batch_size"])

        model = QuantumFinanceModel(
            input_size=len(inputs["feature_columns"]),
            recurrent_type=config["model"]["recurrent_type"],
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
            input_projection_size=config["model"]["input_projection_size"],
            mlp_hidden_size=config["model"]["mlp_hidden_size"],
        )

        model, history = train_model(model, dataloaders, device, config["training"])
        torch.save(model.state_dict(), symbol_dir / "best_model.pt")
        pd.DataFrame(history).to_csv(symbol_dir / "training_history.csv", index=False)

        val_results = evaluate_model(model, dataloaders["val"], device, symbol_dir, "val")
        test_results = evaluate_model(model, dataloaders["test"], device, symbol_dir, "test")
        backtest_summary = run_backtest(test_results["collected"], config["strategy"], symbol_dir, "test")

        summary[symbol] = {
            "validation_metrics": val_results["metrics"],
            "test_metrics": test_results["metrics"],
            "backtest": backtest_summary,
        }

    with open(Path(run_dir) / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Run complete. Outputs saved to {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run_pipeline(args.config)
