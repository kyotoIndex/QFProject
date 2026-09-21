from __future__ import annotations

import math

import torch


def num_qubits_from_dim(dimension: int) -> int:
    n_qubits = int(round(math.log2(dimension)))
    if 2**n_qubits != dimension:
        raise ValueError(f"Hilbert-space dimension {dimension} is not a power of 2.")
    return n_qubits


def basis_bit(index: torch.Tensor, wire: int, n_qubits: int) -> torch.Tensor:
    """Return the computational-basis value of `wire` for each basis index.

    Qubit 0 is the leftmost / most significant bit in ket notation |q0 q1 ...>.
    """
    shift = n_qubits - 1 - wire
    return (index >> shift) & 1


def initial_state(
    batch_size: int,
    n_qubits: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """Prepare |0...0> for a batch of quantum registers."""
    states = torch.zeros(batch_size, 2**n_qubits, device=device, dtype=dtype)
    states[:, 0] = 1.0 + 0.0j
    return states


def hadamard_matrix(device: torch.device | str, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    scale = 1.0 / math.sqrt(2.0)
    return torch.tensor([[scale, scale], [scale, -scale]], device=device, dtype=dtype)


def ry_gate(theta: torch.Tensor) -> torch.Tensor:
    """Batched RY(theta) matrices with shape (batch, 2, 2)."""
    half = theta.reshape(-1) * 0.5
    cosine = torch.cos(half).to(torch.complex64)
    sine = torch.sin(half).to(torch.complex64)
    gate = torch.zeros(theta.shape[0], 2, 2, device=theta.device, dtype=torch.complex64)
    gate[:, 0, 0] = cosine
    gate[:, 0, 1] = -sine
    gate[:, 1, 0] = sine
    gate[:, 1, 1] = cosine
    return gate


def rz_gate(theta: torch.Tensor) -> torch.Tensor:
    """Batched RZ(theta) matrices with shape (batch, 2, 2)."""
    half = theta.reshape(-1) * 0.5
    phase_zero = torch.exp(-1j * half).to(torch.complex64)
    phase_one = torch.exp(1j * half).to(torch.complex64)
    gate = torch.zeros(theta.shape[0], 2, 2, device=theta.device, dtype=torch.complex64)
    gate[:, 0, 0] = phase_zero
    gate[:, 1, 1] = phase_one
    return gate


def apply_single_qubit_gate(states: torch.Tensor, gate: torch.Tensor, wire: int) -> torch.Tensor:
    """Apply a 2x2 gate to one qubit of a batched statevector.

    Args:
        states: (batch, 2**n) complex amplitudes
        gate: (2, 2) or (batch, 2, 2) complex matrix
        wire: target qubit
    """
    n_qubits = num_qubits_from_dim(states.shape[-1])
    batch_size = states.shape[0]
    reshaped = states.reshape(batch_size, *([2] * n_qubits))
    qubit_axis = wire + 1
    permutation = [0] + [axis for axis in range(1, n_qubits + 1) if axis != qubit_axis] + [qubit_axis]
    moved = reshaped.permute(*permutation).contiguous()
    flattened = moved.reshape(batch_size, -1, 2)

    if gate.dim() == 2:
        updated = torch.einsum("ij,bsj->bsi", gate.to(dtype=states.dtype), flattened)
    else:
        updated = torch.einsum("bij,bsj->bsi", gate.to(dtype=states.dtype), flattened)

    updated = updated.reshape(moved.shape)
    inverse = [0] * (n_qubits + 1)
    for destination, source in enumerate(permutation):
        inverse[source] = destination
    restored = updated.permute(*inverse).contiguous()
    return restored.reshape(batch_size, 2**n_qubits)


def apply_hadamard(states: torch.Tensor, wire: int) -> torch.Tensor:
    return apply_single_qubit_gate(states, hadamard_matrix(states.device, states.dtype), wire)


def apply_ry(states: torch.Tensor, theta: torch.Tensor, wire: int) -> torch.Tensor:
    return apply_single_qubit_gate(states, ry_gate(theta), wire)


def apply_rz(states: torch.Tensor, theta: torch.Tensor, wire: int) -> torch.Tensor:
    return apply_single_qubit_gate(states, rz_gate(theta), wire)


def apply_cnot(states: torch.Tensor, control: int, target: int) -> torch.Tensor:
    """Flip `target` when `control` is |1>, as a permutation of the computational basis."""
    n_qubits = num_qubits_from_dim(states.shape[-1])
    dimension = states.shape[-1]
    indices = torch.arange(dimension, device=states.device)
    control_is_one = basis_bit(indices, control, n_qubits).bool()
    permutation = indices.clone()
    permutation[control_is_one] = indices[control_is_one] ^ (1 << (n_qubits - 1 - target))
    return states[:, permutation]


def born_probabilities(states: torch.Tensor) -> torch.Tensor:
    """Return P(i) = |<i|psi>|^2 for each computational-basis state."""
    return torch.real(states.conj() * states)


def pauli_z_eigenvalues(n_qubits: int, wire: int, device: torch.device | str) -> torch.Tensor:
    indices = torch.arange(2**n_qubits, device=device)
    bits = basis_bit(indices, wire, n_qubits)
    return (1 - 2 * bits).to(torch.float32)


def pauli_z_expectation(states: torch.Tensor, wire: int) -> torch.Tensor:
    probabilities = born_probabilities(states)
    eigenvalues = pauli_z_eigenvalues(num_qubits_from_dim(states.shape[-1]), wire, states.device)
    return torch.sum(probabilities * eigenvalues, dim=-1)


def pauli_zz_expectation(states: torch.Tensor, wire_a: int, wire_b: int) -> torch.Tensor:
    n_qubits = num_qubits_from_dim(states.shape[-1])
    probabilities = born_probabilities(states)
    eigenvalues = pauli_z_eigenvalues(n_qubits, wire_a, states.device) * pauli_z_eigenvalues(
        n_qubits, wire_b, states.device
    )
    return torch.sum(probabilities * eigenvalues, dim=-1)


def state_norm(states: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(born_probabilities(states).sum(dim=-1), min=0.0))
