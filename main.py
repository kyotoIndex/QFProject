from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.qf_project.backtest import run_backtest
from src.qf_project.data import download_market_data
from src.qf_project.dataset import build_dataloaders, build_model_inputs
from src.qf_project.evaluate import evaluate_model
from src.qf_project.features import engineer_features
from src.qf_project.model import build_model
from src.qf_project.quantum_circuit import circuit_metadata
from src.qf_project.quantum_encoding import QUANTUM_REGIME_COLUMNS, encode_quantum_states
from src.qf_project.train import train_model
from src.qf_project.utils import create_run_directory, ensure_directory, get_device, load_config, set_seed


def _save_regime_plot(quantum_frame: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Warning: matplotlib unavailable for regime plot: {exc}")
        return

    try:
        plot_frame = quantum_frame[QUANTUM_REGIME_COLUMNS].tail(252)
        figure, axis = plt.subplots(figsize=(10, 4))
        plot_frame.plot.area(ax=axis, stacked=True, alpha=0.85)
        axis.set_title("Born-rule market regime probabilities")
        axis.set_ylabel("P(regime)")
        axis.set_ylim(0.0, 1.0)
        axis.legend(loc="upper left", fontsize=8)
        figure.tight_layout()
        figure.savefig(output_path, dpi=120)
        plt.close(figure)
    except Exception as exc:
        print(f"Warning: could not save regime plot to {output_path}: {exc}")
        try:
            plt.close("all")
        except Exception:
            pass


def run_pipeline(config_path: str) -> None:
    config = load_config(config_path)
    set_seed(config["project"]["seed"])

    output_root = ensure_directory(config["project"]["output_dir"])
    run_dir = create_run_directory(output_root)
    device = get_device()
    metadata = circuit_metadata(config["quantum"])

    with open(run_dir / "config_used.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    with open(run_dir / "quantum_circuit.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    (run_dir / "quantum_circuit.txt").write_text(
        metadata["feature_map_diagram"] + "\n\n" + metadata["vqc_diagram"] + "\n",
        encoding="utf-8",
    )

    market_data = download_market_data(
        symbols=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
        interval=config["data"]["interval"],
        cache_dir=config["data"]["cache_dir"],
    )

    summary = {
        "model_type": config["model"].get("type", "vqc"),
        "quantum": {
            "n_qubits": metadata["n_qubits"],
            "n_layers": metadata["n_layers"],
            "n_reuploads": metadata["n_reuploads"],
            "hilbert_dimension": metadata["hilbert_dimension"],
            "trainable_circuit_parameters": metadata["trainable_circuit_parameters"],
        },
        "symbols": {},
    }

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
        quantum_frame.to_csv(symbol_dir / "quantum_state_probabilities.csv")
        _save_regime_plot(quantum_frame, symbol_dir / "quantum_regime_probabilities.png")

        inputs = build_model_inputs(
            dataset_frame,
            lookback=config["features"]["lookback"],
            horizon=config["features"]["horizon"],
            train_ratio=config["features"]["train_ratio"],
            val_ratio=config["features"]["val_ratio"],
        )
        dataloaders = build_dataloaders(inputs["splits"], batch_size=config["training"]["batch_size"])

        model = build_model(config, input_size=len(inputs["feature_columns"]))
        model, history = train_model(model, dataloaders, device, config["training"])
        torch.save(model.state_dict(), symbol_dir / "best_model.pt")
        pd.DataFrame(history).to_csv(symbol_dir / "training_history.csv", index=False)

        val_results = evaluate_model(model, dataloaders["val"], device, symbol_dir, "val")
        test_results = evaluate_model(model, dataloaders["test"], device, symbol_dir, "test")
        backtest_summary = run_backtest(test_results["collected"], config["strategy"], symbol_dir, "test")

        summary["symbols"][symbol] = {
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
