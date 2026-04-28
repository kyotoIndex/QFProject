from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .train import collect_predictions, summarize_predictions


def evaluate_model(model, data_loader, device, output_dir: str | Path, split_name: str) -> dict:
    collected = collect_predictions(model, data_loader, device)
    metrics = summarize_predictions(collected)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prediction_frame = pd.DataFrame(
        {
            "predicted_return": collected["predictions"]["return"],
            "predicted_direction_probability": collected["predictions"]["direction"],
            "predicted_risk_class": np.argmax(collected["predictions"]["risk"], axis=1),
            "actual_return": collected["targets"]["return"],
            "actual_direction": collected["targets"]["direction"],
            "actual_risk_class": collected["targets"]["risk"],
        }
    )
    prediction_frame.to_csv(output_path / f"{split_name}_predictions.csv", index=False)

    with open(output_path / f"{split_name}_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return {"metrics": metrics, "collected": collected}
