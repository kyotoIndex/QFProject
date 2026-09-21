from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def max_drawdown(curve: np.ndarray) -> float:
    running_max = np.maximum.accumulate(curve)
    return float((curve / running_max - 1.0).min())


def backtest_signal(signal: np.ndarray, realized: np.ndarray) -> dict[str, float]:
    strategy_return = signal * realized
    curve = np.cumprod(1.0 + strategy_return)
    denom = np.std(strategy_return) + 1e-8
    return {
        "cumulative_return": float(curve[-1] - 1.0),
        "annualized_sharpe": float((np.mean(strategy_return) / denom) * np.sqrt(252)),
        "max_drawdown": max_drawdown(curve),
        "hit_rate": float(np.mean(strategy_return > 0)),
        "active_ratio": float(np.mean(signal != 0)),
    }


def evaluate_prediction_frame(
    frame: pd.DataFrame,
    buy_threshold: float = 0.002,
    sell_threshold: float = -0.002,
) -> dict:
    predicted = frame["predicted_return"].to_numpy()
    actual = frame["actual_return"].to_numpy()
    pred_dir_prob = frame["predicted_direction_probability"].to_numpy()
    pred_dir = (pred_dir_prob >= 0.5).astype(int)
    actual_dir = frame["actual_direction"].to_numpy().astype(int)
    n = len(frame)

    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    mae = float(np.mean(np.abs(predicted - actual)))
    zero_rmse = float(np.sqrt(np.mean(actual**2)))
    mean_rmse = float(np.sqrt(np.mean((actual - actual.mean()) ** 2)))
    corr = float("nan")
    if n > 2 and float(np.std(predicted)) > 1e-12 and float(np.std(actual)) > 1e-12:
        corr = float(np.corrcoef(predicted, actual)[0, 1])

    direction_acc = float(np.mean(pred_dir == actual_dir))
    up_rate = float(np.mean(actual_dir == 1))
    pred_up_rate = float(np.mean(pred_dir == 1))
    majority_acc = max(up_rate, 1.0 - up_rate)
    binomial = stats.binomtest(int(np.sum(pred_dir == actual_dir)), n, p=0.5, alternative="greater")

    pred_risk = frame["predicted_risk_class"].to_numpy()
    actual_risk = frame["actual_risk_class"].to_numpy()
    risk_acc = float(np.mean(pred_risk == actual_risk))

    strategy_signal = np.where(
        predicted >= buy_threshold,
        1,
        np.where(predicted <= sell_threshold, -1, 0),
    )
    strategy_signal = np.where((strategy_signal == 1) & (pred_risk >= 2), 0, strategy_signal)

    return {
        "n_samples": n,
        "prediction": {
            "rmse": rmse,
            "mae": mae,
            "zero_predictor_rmse": zero_rmse,
            "mean_predictor_rmse": mean_rmse,
            "return_correlation": corr,
            "predicted_return_mean": float(predicted.mean()),
            "predicted_return_std": float(predicted.std()),
            "actual_return_mean": float(actual.mean()),
            "actual_return_std": float(actual.std()),
        },
        "direction": {
            "accuracy": direction_acc,
            "predicted_up_rate": pred_up_rate,
            "actual_up_rate": up_rate,
            "majority_class_accuracy": majority_acc,
            "accuracy_minus_majority": direction_acc - majority_acc,
            "binomial_p_value_vs_50pct": float(binomial.pvalue),
        },
        "risk": {
            "accuracy": risk_acc,
            "predicted_class_share": {str(cls): float(np.mean(pred_risk == cls)) for cls in (0, 1, 2)},
            "actual_class_share": {str(cls): float(np.mean(actual_risk == cls)) for cls in (0, 1, 2)},
        },
        "backtest": {
            "model_strategy": backtest_signal(strategy_signal, actual),
            "buy_and_hold": backtest_signal(np.ones(n), actual),
            "always_short": backtest_signal(-np.ones(n), actual),
            "long_if_predicted_up": backtest_signal(np.where(pred_dir == 1, 1, 0), actual),
        },
    }


def evaluate_predictions(path: Path, buy_threshold: float = 0.002, sell_threshold: float = -0.002) -> dict:
    return evaluate_prediction_frame(pd.read_csv(path), buy_threshold, sell_threshold)


def evaluate_run(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    symbols = {}
    for symbol_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        pred_path = symbol_dir / "test_predictions.csv"
        if pred_path.exists():
            symbols[symbol_dir.name] = evaluate_predictions(pred_path)
    return {
        "run_dir": str(run_dir),
        "model_type": payload.get("model_type"),
        "quantum": payload.get("quantum"),
        "symbols": symbols,
    }


def verdict_lines(report: dict) -> list[str]:
    lines = []
    for symbol, result in report["symbols"].items():
        direction = result["direction"]
        prediction = result["prediction"]
        backtest = result["backtest"]
        better_than_majority = direction["accuracy"] > direction["majority_class_accuracy"] + 0.01
        better_than_chance = direction["binomial_p_value_vs_50pct"] < 0.05
        corr_ok = prediction["return_correlation"] > 0.05
        rmse_ok = prediction["rmse"] < prediction["zero_predictor_rmse"]
        beat_hold = backtest["model_strategy"]["cumulative_return"] > backtest["buy_and_hold"]["cumulative_return"]
        score = sum([better_than_majority, better_than_chance, corr_ok, rmse_ok, beat_hold])
        if score >= 4:
            label = "useful edge"
        elif score >= 2:
            label = "weak / mixed"
        else:
            label = "not better than naive baselines"
        lines.append(
            f"{symbol}: {label} "
            f"(dir {direction['accuracy']:.1%} vs majority {direction['majority_class_accuracy']:.1%}, "
            f"corr {prediction['return_correlation']:.3f}, "
            f"strategy {backtest['model_strategy']['cumulative_return']:.1%} vs "
            f"hold {backtest['buy_and_hold']['cumulative_return']:.1%})"
        )
    return lines
