from __future__ import annotations

import torch
from torch import nn


class QuantumFinanceModel(nn.Module):
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
