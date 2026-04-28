from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


def _max_drawdown(cumulative_returns: np.ndarray) -> float:
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = cumulative_returns / running_max - 1.0
    return float(drawdown.min())


def run_backtest(collected: dict, strategy_config: dict, output_dir: str | Path, split_name: str) -> dict[str, float]:
    predicted_return = collected["predictions"]["return"]
    predicted_risk = np.argmax(collected["predictions"]["risk"], axis=1)
    realized_return = collected["targets"]["realized_return"]

    signal = np.where(
        (predicted_return >= strategy_config["buy_threshold"]) & (predicted_risk < strategy_config["high_risk_class"]),
        1,
        np.where(predicted_return <= strategy_config["sell_threshold"], -1, 0),
    )

    strategy_return = signal * realized_return
    cumulative_curve = np.cumprod(1 + strategy_return)
    sharpe_denominator = np.std(strategy_return) + 1e-8

    summary = {
        "cumulative_return": float(cumulative_curve[-1] - 1.0),
        "annualized_sharpe": float((np.mean(strategy_return) / sharpe_denominator) * np.sqrt(252)),
        "max_drawdown": _max_drawdown(cumulative_curve),
        "hit_rate": float(np.mean(strategy_return > 0)),
        "active_ratio": float(np.mean(signal != 0)),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "signal": signal,
            "realized_return": realized_return,
            "strategy_return": strategy_return,
            "cumulative_curve": cumulative_curve,
        }
    ).to_csv(output_path / f"{split_name}_backtest.csv", index=False)

    with open(output_path / f"{split_name}_backtest.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary
