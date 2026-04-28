from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


FEATURE_COLUMNS = [
    "return",
    "log_return",
    "momentum",
    "rolling_mean",
    "volatility",
    "volume_change",
    "high_low_range",
    "rsi",
    "macd",
    "macd_signal",
    "bollinger_position",
]

QUANTUM_COLUMNS = [
    "q_bullish_low_vol",
    "q_bullish_high_vol",
    "q_neutral",
    "q_bearish_low_vol",
    "q_bearish_high_vol",
]


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class MarketSequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: dict[str, np.ndarray]):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.return_target = torch.tensor(targets["return"], dtype=torch.float32)
        self.direction_target = torch.tensor(targets["direction"], dtype=torch.float32)
        self.risk_target = torch.tensor(targets["risk"], dtype=torch.long)
        self.realized_return = torch.tensor(targets["realized_return"], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[index],
            "return_target": self.return_target[index],
            "direction_target": self.direction_target[index],
            "risk_target": self.risk_target[index],
            "realized_return": self.realized_return[index],
        }


def time_split(frame: pd.DataFrame, train_ratio: float, val_ratio: float) -> SplitData:
    train_end = int(len(frame) * train_ratio)
    val_end = int(len(frame) * (train_ratio + val_ratio))
    return SplitData(
        train=frame.iloc[:train_end].copy(),
        val=frame.iloc[train_end:val_end].copy(),
        test=frame.iloc[val_end:].copy(),
    )


def assign_risk_classes(frame: pd.DataFrame, train_end_index: int, horizon: int) -> pd.DataFrame:
    labeled = frame.copy()
    future_volatility_window = max(horizon, 5)
    labeled["future_volatility"] = labeled["return"].rolling(future_volatility_window).std().shift(-future_volatility_window)
    train_future_volatility = labeled.iloc[:train_end_index]["future_volatility"].dropna()
    lower = train_future_volatility.quantile(0.33)
    upper = train_future_volatility.quantile(0.67)

    def classify(value: float) -> float:
        if pd.isna(value):
            return np.nan
        if value <= lower:
            return 0
        if value <= upper:
            return 1
        return 2

    labeled["risk_target"] = labeled["future_volatility"].apply(classify)
    return labeled


def build_model_inputs(
    frame: pd.DataFrame,
    lookback: int,
    horizon: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, Any]:
    dataset_frame = frame.copy()
    dataset_frame["target_return"] = dataset_frame["return"].shift(-horizon)
    dataset_frame["target_direction"] = (dataset_frame["target_return"] > 0).astype(int)

    split_indices = time_split(dataset_frame, train_ratio, val_ratio)
    train_end_index = len(split_indices.train)
    dataset_frame = assign_risk_classes(dataset_frame, train_end_index, horizon)
    dataset_frame = dataset_frame.dropna().copy()

    if dataset_frame.empty:
        raise ValueError("No samples available after preprocessing. Try a longer date range or smaller windows.")

    adjusted_train_end_index = max(int(len(dataset_frame) * train_ratio), 1)

    feature_scaler = StandardScaler()
    feature_scaler.fit(dataset_frame.iloc[:adjusted_train_end_index][FEATURE_COLUMNS])
    dataset_frame[FEATURE_COLUMNS] = feature_scaler.transform(dataset_frame[FEATURE_COLUMNS])

    sequences = []
    target_return = []
    target_direction = []
    target_risk = []
    realized_return = []
    timestamps = []

    input_columns = FEATURE_COLUMNS + QUANTUM_COLUMNS

    for end_index in range(lookback, len(dataset_frame)):
        window = dataset_frame.iloc[end_index - lookback:end_index]
        target_row = dataset_frame.iloc[end_index]
        sequences.append(window[input_columns].to_numpy(dtype=np.float32))
        target_return.append(target_row["target_return"])
        target_direction.append(target_row["target_direction"])
        target_risk.append(target_row["risk_target"])
        realized_return.append(target_row["target_return"])
        timestamps.append(target_row.name)

    if not sequences:
        raise ValueError("No rolling windows were created. Reduce lookback or expand the dataset.")

    sequence_array = np.stack(sequences)
    targets = {
        "return": np.asarray(target_return, dtype=np.float32),
        "direction": np.asarray(target_direction, dtype=np.float32),
        "risk": np.asarray(target_risk, dtype=np.int64),
        "realized_return": np.asarray(realized_return, dtype=np.float32),
    }

    sequence_count = len(sequence_array)
    train_cutoff = int(sequence_count * train_ratio)
    val_cutoff = int(sequence_count * (train_ratio + val_ratio))

    return {
        "splits": {
            "train": (sequence_array[:train_cutoff], {key: value[:train_cutoff] for key, value in targets.items()}),
            "val": (sequence_array[train_cutoff:val_cutoff], {key: value[train_cutoff:val_cutoff] for key, value in targets.items()}),
            "test": (sequence_array[val_cutoff:], {key: value[val_cutoff:] for key, value in targets.items()}),
        },
        "timestamps": {
            "train": timestamps[:train_cutoff],
            "val": timestamps[train_cutoff:val_cutoff],
            "test": timestamps[val_cutoff:],
        },
        "feature_columns": input_columns,
        "feature_scaler": feature_scaler,
    }


def build_dataloaders(splits: dict[str, Any], batch_size: int) -> dict[str, DataLoader]:
    loaders = {}
    for split_name, (features, targets) in splits.items():
        dataset = MarketSequenceDataset(features, targets)
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
        )
    return loaders
