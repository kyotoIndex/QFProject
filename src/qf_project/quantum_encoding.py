from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .quantum_circuit import BASIS_LABELS, REGIME_BASIS_MAP, qaoa_inspired_market_circuit


QUANTUM_BASIS_COLUMNS = [f"q_p{label}" for label in BASIS_LABELS]
QUANTUM_REGIME_COLUMNS = [
    "q_bullish_low_vol",
    "q_bullish_high_vol",
    "q_neutral",
    "q_bearish_low_vol",
    "q_bearish_high_vol",
]
QUANTUM_COLUMNS = QUANTUM_BASIS_COLUMNS + QUANTUM_REGIME_COLUMNS


def _feature_angles(series: pd.Series, temperature: float, center: float = 0.0) -> torch.Tensor:
    values = series.to_numpy(dtype=np.float64)
    scale = np.std(values) + 1e-6
    normalized = (values - center) / (scale * max(temperature, 1e-6))
    angles = np.pi * np.tanh(normalized)
    return torch.tensor(angles, dtype=torch.float32)


def _aggregate_regimes(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    label_to_index = {label: index for index, label in enumerate(BASIS_LABELS)}
    regimes: dict[str, np.ndarray] = {}
    for regime, labels in REGIME_BASIS_MAP.items():
        selected = probabilities[:, [label_to_index[label] for label in labels]]
        regimes[f"q_{regime}"] = selected.sum(axis=1)
    return regimes


def encode_quantum_states(frame: pd.DataFrame, temperature: float) -> pd.DataFrame:
    """Encode market conditions into a real 3-qubit quantum state.

    Unlike a classical softmax over hand-crafted scores, this prepares a
    superposition with Hadamard gates, applies data-dependent unitaries, and
    reads probabilities from the Born rule.
    """
    momentum_angle = _feature_angles(frame["momentum"], temperature)
    volatility_angle = _feature_angles(frame["volatility"], temperature)
    rsi_center = 50.0 if "rsi" in frame.columns else 0.0
    rsi_angle = _feature_angles(frame["rsi"] if "rsi" in frame.columns else frame["momentum"], temperature, center=rsi_center)

    with torch.no_grad():
        probabilities = qaoa_inspired_market_circuit(momentum_angle, volatility_angle, rsi_angle).cpu().numpy()

    encoded = pd.DataFrame(probabilities, index=frame.index, columns=QUANTUM_BASIS_COLUMNS)
    for column, values in _aggregate_regimes(probabilities).items():
        encoded[column] = values
    return encoded
