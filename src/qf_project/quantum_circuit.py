from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .quantum_gates import (
    apply_cnot,
    apply_hadamard,
    apply_ry,
    apply_rz,
    born_probabilities,
    initial_state,
    num_qubits_from_dim,
    pauli_z_expectation,
    pauli_zz_expectation,
)


BASIS_LABELS = ["000", "001", "010", "011", "100", "101", "110", "111"]

# Computational-basis assignment used by the 3-qubit market feature map.
REGIME_BASIS_MAP = {
    "bullish_low_vol": ["000"],
    "bullish_high_vol": ["001"],
    "neutral": ["010", "011", "100"],
    "bearish_low_vol": ["101"],
    "bearish_high_vol": ["110", "111"],
}


def _broadcast_angle(theta: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    if theta.dim() == 0:
        return theta.reshape(1).expand(batch_size).to(device)
    return theta.to(device)


def apply_hardware_efficient_layer(states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """One hardware-efficient ansatz layer: RZ-RY-RZ on each qubit, then a CNOT ring.

    This is the same connectivity pattern used on superconducting NISQ devices:
    single-qubit rotations plus nearest-neighbour entanglement.
    """
    n_qubits = num_qubits_from_dim(states.shape[-1])
    for wire in range(n_qubits):
        states = apply_rz(states, weights[:, wire, 0], wire)
        states = apply_ry(states, weights[:, wire, 1], wire)
        states = apply_rz(states, weights[:, wire, 2], wire)
    for wire in range(n_qubits):
        states = apply_cnot(states, wire, (wire + 1) % n_qubits)
    return states


def encode_angles(states: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Angle encoding: RY(x_i) on qubit i."""
    n_qubits = angles.shape[-1]
    for wire in range(n_qubits):
        states = apply_ry(states, angles[:, wire], wire)
    return states


class HardwareEfficientVQC(nn.Module):
    """Variational quantum circuit with data re-uploading.

    For each re-upload step the circuit:
      1. encodes a feature vector by RY rotations (quantum feature map)
      2. applies a trainable hardware-efficient ansatz with entanglement

    Readout uses Pauli-Z and nearest-neighbour ZZ expectations, i.e. the same
    class of observables used in VQE-style hybrid algorithms.
    """

    def __init__(self, n_qubits: int, n_layers: int, n_reuploads: int):
        super().__init__()
        if n_qubits < 2:
            raise ValueError("The variational circuit needs at least 2 qubits for entanglement.")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_reuploads = n_reuploads
        self.weights = nn.Parameter(0.05 * torch.randn(n_layers, n_qubits, 3))

    @property
    def n_observables(self) -> int:
        return 2 * self.n_qubits

    def _evolve(self, angles: torch.Tensor) -> torch.Tensor:
        batch_size = angles.shape[0]
        states = initial_state(batch_size, self.n_qubits, angles.device)
        batched_weights = self.weights.unsqueeze(0).expand(batch_size, -1, -1, -1)
        for step in range(self.n_reuploads):
            states = encode_angles(states, angles[:, step, :])
            for layer in range(self.n_layers):
                states = apply_hardware_efficient_layer(states, batched_weights[:, layer])
        return states

    def forward(self, angles: torch.Tensor) -> dict[str, torch.Tensor]:
        states = self._evolve(angles)
        z_expectations = [pauli_z_expectation(states, wire) for wire in range(self.n_qubits)]
        zz_expectations = [
            pauli_zz_expectation(states, wire, (wire + 1) % self.n_qubits) for wire in range(self.n_qubits)
        ]
        observables = torch.stack(z_expectations + zz_expectations, dim=1)
        return {
            "observables": observables,
            "probabilities": born_probabilities(states),
            "states": states,
        }


def qaoa_inspired_market_circuit(
    momentum_angle: torch.Tensor,
    volatility_angle: torch.Tensor,
    rsi_angle: torch.Tensor,
) -> torch.Tensor:
    """Prepare a 3-qubit market state and return Born-rule probabilities.

    Algorithm (QAOA / IQP-style feature map, not a classical softmax):
      1. Hadamard on every qubit -> uniform superposition over 8 basis states
      2. Data-dependent RZ phases (cost-like diagonal unitary)
      3. CNOT ring (entanglement / mixer-style interactions)
      4. Data-dependent RY rotations (second encoding layer)
      5. Another CNOT ring
      6. Measure in the computational basis: P(i) = |<i|psi>|^2
    """
    batch_size = momentum_angle.shape[0]
    device = momentum_angle.device
    states = initial_state(batch_size, 3, device)

    for wire in range(3):
        states = apply_hadamard(states, wire)

    states = apply_rz(states, _broadcast_angle(momentum_angle, batch_size, device), 0)
    states = apply_rz(states, _broadcast_angle(volatility_angle, batch_size, device), 1)
    states = apply_rz(states, _broadcast_angle(rsi_angle, batch_size, device), 2)
    states = apply_cnot(states, 0, 1)
    states = apply_cnot(states, 1, 2)
    states = apply_cnot(states, 2, 0)

    states = apply_ry(states, _broadcast_angle(momentum_angle, batch_size, device), 0)
    states = apply_ry(states, _broadcast_angle(volatility_angle, batch_size, device), 1)
    states = apply_ry(states, _broadcast_angle(rsi_angle, batch_size, device), 2)
    states = apply_cnot(states, 0, 1)
    states = apply_cnot(states, 1, 2)
    states = apply_cnot(states, 2, 0)

    return born_probabilities(states)


def describe_vqc(n_qubits: int, n_layers: int, n_reuploads: int) -> str:
    cnot_ring = " -> ".join(f"q{i}" for i in range(n_qubits)) + f" -> q0"
    lines = [
        "Variational Quantum Circuit",
        f"- qubits: {n_qubits}",
        f"- ansatz layers per re-upload: {n_layers}",
        f"- data re-uploads: {n_reuploads}",
        f"- Hilbert-space dimension: {2 ** n_qubits}",
        "",
        "for t in 1..n_reuploads:",
        "  angle encoding:",
    ]
    for wire in range(n_qubits):
        lines.append(f"    RY(x[t, {wire}]) | q{wire}")
    lines.extend(
        [
            "  hardware-efficient ansatz (repeat n_layers):",
            "    for each qubit qi: RZ(theta) RY(theta) RZ(theta)",
            f"    CNOT ring: {cnot_ring}",
            "",
            "measurement:",
            "  <Z_i> on every qubit",
            "  <Z_i Z_{i+1}> on every neighbouring pair",
            "",
            "This circuit is NISQ-native: only single-qubit rotations and CNOTs.",
            "Training is a hybrid quantum-classical loop (VQE / QNN style).",
        ]
    )
    return "\n".join(lines)


def describe_market_feature_map() -> str:
    regime_lines = []
    for regime, kets in REGIME_BASIS_MAP.items():
        pretty = " + ".join(f"|{label}>" for label in kets)
        regime_lines.append(f"  {regime}: {pretty}")
    return "\n".join(
        [
            "3-qubit QAOA-inspired market feature map",
            "- q0: trend / direction qubit",
            "- q1: volatility / risk qubit",
            "- q2: RSI / oscillator qubit",
            "",
            "H^⊗3 |000>                     # superposition over 8 market basis states",
            "RZ(momentum) RZ(volatility) RZ(rsi)",
            "CNOT ring                      # entanglement between market factors",
            "RY(momentum) RY(volatility) RY(rsi)",
            "CNOT ring",
            "measure computational basis    # Born rule P(i)=|<i|psi>|^2",
            "",
            "Basis-to-regime map:",
            *regime_lines,
        ]
    )


def circuit_metadata(quantum_config: dict[str, Any]) -> dict[str, Any]:
    n_qubits = int(quantum_config["n_qubits"])
    n_layers = int(quantum_config["n_layers"])
    n_reuploads = int(quantum_config["n_reuploads"])
    return {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "n_reuploads": n_reuploads,
        "hilbert_dimension": 2**n_qubits,
        "trainable_circuit_parameters": n_layers * n_qubits * 3,
        "encoding": quantum_config.get("encoding", "qaoa_feature_map"),
        "ansatz": quantum_config.get("ansatz", "hardware_efficient"),
        "regime_basis_map": REGIME_BASIS_MAP,
        "vqc_diagram": describe_vqc(n_qubits, n_layers, n_reuploads),
        "feature_map_diagram": describe_market_feature_map(),
    }
