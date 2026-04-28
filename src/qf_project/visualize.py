from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RISK_LABELS = ["Low", "Medium", "High"]
SIGNAL_LABELS = {-1: "Sell", 0: "Hold", 1: "Buy"}


def _prepare_axis_dates(frame: pd.DataFrame) -> pd.Series:
    if "timestamp" in frame.columns:
        return pd.to_datetime(frame["timestamp"])
    return pd.Series(np.arange(len(frame)))


def plot_training_history(history: pd.DataFrame, output_dir: str | Path) -> None:
    figure, axis_left = plt.subplots(figsize=(10, 5))
    axis_right = axis_left.twinx()

    axis_left.plot(history["epoch"], history["train_loss"], color="tab:blue", label="Train Loss")
    axis_right.plot(history["epoch"], history["validation_score"], color="tab:orange", label="Validation Score")

    axis_left.set_xlabel("Epoch")
    axis_left.set_ylabel("Train Loss", color="tab:blue")
    axis_right.set_ylabel("Validation Score", color="tab:orange")
    axis_left.set_title("Training History")

    lines_left, labels_left = axis_left.get_legend_handles_labels()
    lines_right, labels_right = axis_right.get_legend_handles_labels()
    axis_left.legend(lines_left + lines_right, labels_left + labels_right, loc="best")

    figure.tight_layout()
    figure.savefig(Path(output_dir) / "training_history.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_prediction_results(predictions: pd.DataFrame, output_dir: str | Path, split_name: str) -> None:
    x_axis = _prepare_axis_dates(predictions)
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, height_ratios=[2, 1])

    axes[0].plot(x_axis, predictions["actual_return"], label="Actual Return", color="tab:blue", linewidth=1.2)
    axes[0].plot(x_axis, predictions["predicted_return"], label="Predicted Return", color="tab:red", linewidth=1.0, alpha=0.8)
    axes[0].set_ylabel("Return")
    axes[0].set_title(f"{split_name.upper()} Return Prediction")
    axes[0].legend(loc="best")

    axes[1].plot(x_axis, predictions["predicted_direction_probability"], color="tab:green", label="Predicted Up Probability")
    axes[1].fill_between(x_axis, 0.5, predictions["actual_direction"], color="tab:gray", alpha=0.2, label="Actual Direction")
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Direction Prob.")
    axes[1].set_xlabel("Time")
    axes[1].legend(loc="best")

    figure.tight_layout()
    figure.savefig(Path(output_dir) / f"{split_name}_predictions.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_risk_confusion_matrix(predictions: pd.DataFrame, output_dir: str | Path) -> None:
    confusion = pd.crosstab(
        predictions["actual_risk_class"],
        predictions["predicted_risk_class"],
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False,
    ).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(confusion.values, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axis.set_xticks(range(3), RISK_LABELS)
    axis.set_yticks(range(3), RISK_LABELS)
    axis.set_xlabel("Predicted Risk")
    axis.set_ylabel("Actual Risk")
    axis.set_title("Risk Classification Confusion Matrix")

    for row in range(3):
        for column in range(3):
            axis.text(column, row, str(confusion.iloc[row, column]), ha="center", va="center", color="black")

    figure.tight_layout()
    figure.savefig(Path(output_dir) / "risk_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_backtest_results(backtest: pd.DataFrame, output_dir: str | Path) -> None:
    x_axis = _prepare_axis_dates(backtest)
    buy_and_hold = np.cumprod(1 + backtest["realized_return"].to_numpy())

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(x_axis, backtest["cumulative_curve"], label="Strategy", color="tab:purple", linewidth=1.3)
    axis.plot(x_axis, buy_and_hold, label="Buy and Hold", color="tab:gray", linewidth=1.0, linestyle="--")
    axis.set_title("Backtest Cumulative Return")
    axis.set_xlabel("Time")
    axis.set_ylabel("Cumulative Wealth")
    axis.legend(loc="best")

    figure.tight_layout()
    figure.savefig(Path(output_dir) / "test_backtest.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_signal_distribution(backtest: pd.DataFrame, output_dir: str | Path) -> None:
    counts = backtest["signal"].value_counts().reindex([-1, 0, 1], fill_value=0)
    labels = [SIGNAL_LABELS[value] for value in counts.index]

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(labels, counts.values, color=["tab:red", "tab:gray", "tab:green"])
    axis.set_title("Trading Signal Distribution")
    axis.set_ylabel("Count")

    figure.tight_layout()
    figure.savefig(Path(output_dir) / "signal_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def generate_visualizations(symbol_dir: str | Path) -> None:
    output_path = Path(symbol_dir)
    history = pd.read_csv(output_path / "training_history.csv")
    val_predictions = pd.read_csv(output_path / "val_predictions.csv")
    test_predictions = pd.read_csv(output_path / "test_predictions.csv")
    test_backtest = pd.read_csv(output_path / "test_backtest.csv")

    plot_training_history(history, output_path)
    plot_prediction_results(val_predictions, output_path, "val")
    plot_prediction_results(test_predictions, output_path, "test")
    plot_risk_confusion_matrix(test_predictions, output_path)
    plot_backtest_results(test_backtest, output_path)
    plot_signal_distribution(test_backtest, output_path)
