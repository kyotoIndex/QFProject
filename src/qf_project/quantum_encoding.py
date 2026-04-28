from __future__ import annotations

import numpy as np
import pandas as pd


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    shifted = values / max(temperature, 1e-6)
    shifted = shifted - np.max(shifted, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def encode_quantum_states(frame: pd.DataFrame, temperature: float) -> pd.DataFrame:
    momentum = frame["momentum"].to_numpy()
    volatility = frame["volatility"].to_numpy()

    momentum_scale = np.std(momentum) + 1e-6
    volatility_scale = np.std(volatility) + 1e-6

    normalized_momentum = momentum / momentum_scale
    normalized_volatility = volatility / volatility_scale
    neutrality = -np.abs(normalized_momentum) - 0.25 * np.abs(normalized_volatility)

    scores = np.column_stack(
        [
            normalized_momentum - normalized_volatility,
            normalized_momentum + normalized_volatility,
            neutrality,
            -normalized_momentum - normalized_volatility,
            -normalized_momentum + normalized_volatility,
        ]
    )

    probabilities = _softmax(scores, temperature)
    columns = [
        "q_bullish_low_vol",
        "q_bullish_high_vol",
        "q_neutral",
        "q_bearish_low_vol",
        "q_bearish_high_vol",
    ]
    encoded = pd.DataFrame(probabilities, index=frame.index, columns=columns)
    return encoded
