from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.qf_project.benchmark import evaluate_prediction_frame
from src.qf_project.model import VariationalQuantumFinanceModel, build_model
from src.qf_project.quantum_circuit import HardwareEfficientVQC, qaoa_inspired_market_circuit
from src.qf_project.quantum_encoding import QUANTUM_COLUMNS, encode_quantum_states
from src.qf_project.quantum_gates import (
    apply_cnot,
    apply_hadamard,
    apply_ry,
    born_probabilities,
    initial_state,
    pauli_z_expectation,
    state_norm,
)


class QuantumGateTests(unittest.TestCase):
    def test_ry_pi_flips_zero_to_one(self) -> None:
        states = initial_state(1, 1, "cpu")
        theta = torch.tensor([math.pi], dtype=torch.float32)
        evolved = apply_ry(states, theta, wire=0)
        probabilities = born_probabilities(evolved)
        self.assertGreater(probabilities[0, 1].item(), 0.99)
        self.assertLess(probabilities[0, 0].item(), 0.01)

    def test_hadamard_and_cnot_make_bell_state(self) -> None:
        states = initial_state(1, 2, "cpu")
        states = apply_hadamard(states, wire=0)
        states = apply_cnot(states, control=0, target=1)
        probabilities = born_probabilities(states).squeeze(0)
        self.assertAlmostEqual(probabilities[0].item(), 0.5, places=5)
        self.assertAlmostEqual(probabilities[3].item(), 0.5, places=5)
        self.assertAlmostEqual(probabilities[1].item(), 0.0, places=5)
        self.assertAlmostEqual(probabilities[2].item(), 0.0, places=5)

    def test_unitarity_preserves_norm(self) -> None:
        states = initial_state(4, 3, "cpu")
        theta = torch.tensor([0.3, -1.2, 2.0, 0.7], dtype=torch.float32)
        states = apply_ry(states, theta, 0)
        states = apply_cnot(states, 0, 1)
        states = apply_hadamard(states, 2)
        norms = state_norm(states)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))


class QuantumCircuitTests(unittest.TestCase):
    def test_market_feature_map_probabilities_sum_to_one(self) -> None:
        momentum = torch.tensor([0.4, -0.8, 1.2], dtype=torch.float32)
        volatility = torch.tensor([0.1, 0.9, -0.2], dtype=torch.float32)
        rsi = torch.tensor([0.0, 0.5, -0.5], dtype=torch.float32)
        probabilities = qaoa_inspired_market_circuit(momentum, volatility, rsi)
        totals = probabilities.sum(dim=1)
        self.assertEqual(tuple(probabilities.shape), (3, 8))
        self.assertTrue(torch.allclose(totals, torch.ones_like(totals), atol=1e-5))

    def test_vqc_forward_and_backward(self) -> None:
        circuit = HardwareEfficientVQC(n_qubits=3, n_layers=1, n_reuploads=2)
        angles = torch.randn(5, 2, 3)
        outputs = circuit(angles)
        self.assertEqual(tuple(outputs["observables"].shape), (5, 6))
        self.assertEqual(tuple(outputs["probabilities"].shape), (5, 8))
        loss = outputs["observables"].sum()
        loss.backward()
        self.assertIsNotNone(circuit.weights.grad)
        self.assertGreater(circuit.weights.grad.abs().sum().item(), 0.0)
        self.assertTrue(torch.allclose(outputs["probabilities"].sum(dim=1), torch.ones(5), atol=1e-5))

    def test_z_expectation_of_zero_state(self) -> None:
        states = initial_state(1, 2, "cpu")
        self.assertAlmostEqual(pauli_z_expectation(states, 0).item(), 1.0, places=5)
        self.assertAlmostEqual(pauli_z_expectation(states, 1).item(), 1.0, places=5)


class EncodingAndModelTests(unittest.TestCase):
    def test_encode_quantum_states_uses_born_rule_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "momentum": [0.02, -0.01, 0.04, 0.00],
                "volatility": [0.01, 0.03, 0.02, 0.05],
                "rsi": [55.0, 40.0, 70.0, 48.0],
            }
        )
        encoded = encode_quantum_states(frame, temperature=1.0)
        self.assertEqual(list(encoded.columns), QUANTUM_COLUMNS)
        regime_sum = encoded[
            [
                "q_bullish_low_vol",
                "q_bullish_high_vol",
                "q_neutral",
                "q_bearish_low_vol",
                "q_bearish_high_vol",
            ]
        ].sum(axis=1)
        self.assertTrue(((regime_sum - 1.0).abs() < 1e-5).all())

    def test_variational_model_output_shapes(self) -> None:
        model = VariationalQuantumFinanceModel(
            input_size=6,
            n_qubits=3,
            n_layers=1,
            n_reuploads=2,
            dropout=0.0,
            readout_hidden_size=8,
        )
        features = torch.randn(4, 8, 6)
        outputs = model(features)
        self.assertEqual(tuple(outputs["return"].shape), (4,))
        self.assertEqual(tuple(outputs["direction"].shape), (4,))
        self.assertEqual(tuple(outputs["risk"].shape), (4, 3))
        loss = outputs["return"].sum() + outputs["direction"].sum() + outputs["risk"].sum()
        loss.backward()
        trainable = [parameter for parameter in model.parameters() if parameter.grad is not None]
        self.assertGreater(len(trainable), 0)

    def test_build_model_reads_vqc_config(self) -> None:
        config_path = Path("configs/default.yaml")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model = build_model(config, input_size=24)
        self.assertIsInstance(model, VariationalQuantumFinanceModel)


class BenchmarkTests(unittest.TestCase):
    def test_always_up_matches_majority_baseline(self) -> None:
        actual = np.array([0.01, 0.02, -0.01, 0.03, 0.00, 0.04, -0.02, 0.01])
        frame = pd.DataFrame(
            {
                "predicted_return": np.full(len(actual), 0.01),
                "predicted_direction_probability": np.ones(len(actual)),
                "predicted_risk_class": np.zeros(len(actual), dtype=int),
                "actual_return": actual,
                "actual_direction": (actual > 0).astype(int),
                "actual_risk_class": np.zeros(len(actual), dtype=int),
            }
        )
        report = evaluate_prediction_frame(frame)
        self.assertAlmostEqual(report["direction"]["accuracy"], report["direction"]["majority_class_accuracy"])
        self.assertAlmostEqual(report["direction"]["predicted_up_rate"], 1.0)
        self.assertGreater(report["backtest"]["buy_and_hold"]["cumulative_return"], -1.0)

    def test_perfect_sign_forecast_beats_majority(self) -> None:
        actual = np.array([0.02, -0.03, 0.01, -0.04, 0.05, -0.01, 0.02, -0.02])
        frame = pd.DataFrame(
            {
                "predicted_return": actual,
                "predicted_direction_probability": (actual > 0).astype(float),
                "predicted_risk_class": np.zeros(len(actual), dtype=int),
                "actual_return": actual,
                "actual_direction": (actual > 0).astype(int),
                "actual_risk_class": np.zeros(len(actual), dtype=int),
            }
        )
        report = evaluate_prediction_frame(frame)
        self.assertGreater(report["direction"]["accuracy"], report["direction"]["majority_class_accuracy"])
        self.assertGreater(
            report["backtest"]["model_strategy"]["cumulative_return"],
            report["backtest"]["buy_and_hold"]["cumulative_return"],
        )
        self.assertAlmostEqual(report["prediction"]["return_correlation"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
