from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qf_project.benchmark import evaluate_run, verdict_lines


def _row(symbol: str, name: str, report: dict) -> dict:
    result = report["symbols"][symbol]
    return {
        "model": name,
        "symbol": symbol,
        "direction_accuracy": result["direction"]["accuracy"],
        "majority_accuracy": result["direction"]["majority_class_accuracy"],
        "predicted_up_rate": result["direction"]["predicted_up_rate"],
        "return_correlation": result["prediction"]["return_correlation"],
        "rmse": result["prediction"]["rmse"],
        "zero_predictor_rmse": result["prediction"]["zero_predictor_rmse"],
        "strategy_return": result["backtest"]["model_strategy"]["cumulative_return"],
        "buy_and_hold_return": result["backtest"]["buy_and_hold"]["cumulative_return"],
        "strategy_sharpe": result["backtest"]["model_strategy"]["annualized_sharpe"],
        "hold_sharpe": result["backtest"]["buy_and_hold"]["annualized_sharpe"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two saved runs against naive baselines.")
    parser.add_argument("left_run", type=Path)
    parser.add_argument("right_run", type=Path)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    left = evaluate_run(args.left_run)
    right = evaluate_run(args.right_run)
    symbols = sorted(set(left["symbols"]) & set(right["symbols"]))
    ordered_rows = []
    for symbol in symbols:
        ordered_rows.append(_row(symbol, args.left_name, left))
        ordered_rows.append(_row(symbol, args.right_name, right))

    payload = {
        "left": {"name": args.left_name, "run_dir": str(args.left_run), "model_type": left.get("model_type")},
        "right": {"name": args.right_name, "run_dir": str(args.right_run), "model_type": right.get("model_type")},
        "rows": ordered_rows,
        "left_verdict": verdict_lines(left),
        "right_verdict": verdict_lines(right),
    }
    print(json.dumps(payload, indent=2))
    print("\nComparison")
    header = (
        f"{'symbol':<6} {'model':<12} {'dir_acc':>8} {'majority':>9} "
        f"{'pred_up':>8} {'corr':>7} {'rmse':>8} {'zero_rmse':>9} "
        f"{'strat':>8} {'hold':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in ordered_rows:
        print(
            f"{row['symbol']:<6} {row['model']:<12} "
            f"{row['direction_accuracy']:>7.1%} {row['majority_accuracy']:>8.1%} "
            f"{row['predicted_up_rate']:>7.1%} {row['return_correlation']:>7.3f} "
            f"{row['rmse']:>8.4f} {row['zero_predictor_rmse']:>9.4f} "
            f"{row['strategy_return']:>7.1%} {row['buy_and_hold_return']:>7.1%}"
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
