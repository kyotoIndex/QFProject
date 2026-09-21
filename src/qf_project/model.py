from __future__ import annotations

import torch
from torch import nn

from .quantum_circuit import HardwareEfficientVQC


class ClassicalRecurrentFinanceModel(nn.Module):
    """Optional classical baseline: GRU/LSTM multi-task predictor."""

    def __init__(
        self,
        input_size: int,
        recurrent_type: str,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        input_projection_size: int,
        mlp_hidden_size: int,
    ):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, input_projection_size),
            nn.ReLU(),
        )
        recurrent_cls = nn.GRU if recurrent_type.lower() == "gru" else nn.LSTM
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.recurrent = recurrent_cls(
            input_size=input_projection_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.return_head = nn.Linear(mlp_hidden_size, 1)
        self.direction_head = nn.Linear(mlp_hidden_size, 1)
        self.risk_head = nn.Linear(mlp_hidden_size, 3)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        projected = self.input_projection(features)
        recurrent_output, _ = self.recurrent(projected)
        shared_features = self.shared_head(recurrent_output[:, -1, :])
        return {
            "return": self.return_head(shared_features).squeeze(-1),
            "direction": self.direction_head(shared_features).squeeze(-1),
            "risk": self.risk_head(shared_features),
        }


class VariationalQuantumFinanceModel(nn.Module):
    """Hybrid quantum-classical model built around a variational quantum circuit.

    A small classical linear map only compresses the lookback window onto n
    qubits. The sequential structure is handled by quantum data re-uploading,
    and predictions are read out from Pauli observables of the final state.
    """

    def __init__(
        self,
        input_size: int,
        n_qubits: int,
        n_layers: int,
        n_reuploads: int,
        dropout: float,
        readout_hidden_size: int,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_reuploads = n_reuploads
        self.angle_encoder = nn.Linear(input_size, n_qubits)
        self.circuit = HardwareEfficientVQC(
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_reuploads=n_reuploads,
        )
        self.readout = nn.Sequential(
            nn.Linear(self.circuit.n_observables, readout_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.return_head = nn.Linear(readout_hidden_size, 1)
        self.direction_head = nn.Linear(readout_hidden_size, 1)
        self.risk_head = nn.Linear(readout_hidden_size, 3)

    def _window_to_angles(self, features: torch.Tensor) -> torch.Tensor:
        _, lookback, _ = features.shape
        projected = self.angle_encoder(features)
        if lookback >= self.n_reuploads:
            indices = torch.linspace(0, lookback - 1, self.n_reuploads, device=features.device)
            index = indices.round().long().clamp(0, lookback - 1)
            sampled = projected.index_select(1, index)
        else:
            pad = projected[:, -1:, :].repeat(1, self.n_reuploads - lookback, 1)
            sampled = torch.cat([projected, pad], dim=1)
        return torch.pi * torch.tanh(sampled)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        angles = self._window_to_angles(features)
        circuit_output = self.circuit(angles)
        shared_features = self.readout(circuit_output["observables"])
        return {
            "return": self.return_head(shared_features).squeeze(-1),
            "direction": self.direction_head(shared_features).squeeze(-1),
            "risk": self.risk_head(shared_features),
            "quantum_observables": circuit_output["observables"],
            "quantum_probabilities": circuit_output["probabilities"],
        }


def build_model(config: dict, input_size: int) -> nn.Module:
    model_config = config["model"]
    model_type = str(model_config.get("type", "vqc")).lower()
    if model_type in {"gru", "lstm", "classical"}:
        recurrent_type = model_config.get("recurrent_type", model_type if model_type in {"gru", "lstm"} else "gru")
        return ClassicalRecurrentFinanceModel(
            input_size=input_size,
            recurrent_type=recurrent_type,
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
            input_projection_size=model_config["input_projection_size"],
            mlp_hidden_size=model_config["mlp_hidden_size"],
        )

    quantum_config = config["quantum"]
    return VariationalQuantumFinanceModel(
        input_size=input_size,
        n_qubits=quantum_config["n_qubits"],
        n_layers=quantum_config["n_layers"],
        n_reuploads=quantum_config["n_reuploads"],
        dropout=model_config.get("dropout", 0.1),
        readout_hidden_size=model_config.get("readout_hidden_size", 32),
    )


# Backwards-compatible alias used by older imports.
QuantumFinanceModel = VariationalQuantumFinanceModel
