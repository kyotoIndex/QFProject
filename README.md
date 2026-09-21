# Quantum Finance: Variational Quantum Circuits for Market Prediction

This project is a **gate-level quantum machine learning system** for financial time series. It no longer uses a classical softmax as a stand-in for quantum behaviour. Market data is prepared as a real quantum state, evolved by unitary gates, and read out with the Born rule. Predictions come from a **variational quantum circuit (VQC)** with hardware-efficient entanglement.

A classical GRU/LSTM baseline is still available for comparison.

## What is actually quantum

| Piece | Quantum content |
| --- | --- |
| Market feature map | 3-qubit QAOA-style circuit: Hadamard superposition, data-dependent RZ/RY rotations, CNOT entanglement |
| State | Complex statevector in a \(2^n\)-dimensional Hilbert space |
| Dynamics | Unitary gates only: RY, RZ, H, CNOT |
| Readout | Computational-basis probabilities \(P(i)=\lvert\langle i\mid\psi\rangle\rvert^2\) and Pauli-Z / ZZ expectations |
| Predictor | Data-reuploading hardware-efficient VQC, trained as a hybrid QNN / VQE-style loop |
| Hardware mapping | The ansatz uses single-qubit rotations and a CNOT ring, the same gate set used on NISQ devices |

The circuits are simulated classically with an exact statevector backend. That is standard NISQ research practice: the **algorithm is a quantum algorithm**, even when the execution engine is a simulator.

## Pipeline

1. Download OHLCV data with `yfinance` and cache it
2. Build classical technical indicators (momentum, volatility, RSI, MACD, Bollinger)
3. Encode each day into a 3-qubit market state and measure Born-rule regime probabilities
4. Compress the lookback window onto `n` qubits and run a trainable VQC with data re-uploading
5. Read Pauli observables and predict next-day return, direction, and risk class
6. Convert predictions into trade signals and backtest

```text
price history
    -> technical features
    -> H/RZ/RY/CNOT feature map
    -> Born-rule probabilities
    -> angle encoding + hardware-efficient VQC
    -> <Z>, <ZZ>
    -> return / direction / risk
    -> backtest
```

## Project structure

```text
configs/
  default.yaml           # variational quantum model
  classical_gru.yaml     # optional GRU baseline
  smoke.yaml             # short run for sanity checks
src/qf_project/
  quantum_gates.py       # batched statevector gates
  quantum_circuit.py     # QAOA feature map + VQC
  quantum_encoding.py    # market-state preparation
  model.py               # hybrid VQC model and classical baseline
  features.py            # technical indicators
  dataset.py             # rolling windows and labels
  train.py / evaluate.py / backtest.py
main.py
tests/test_quantum.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

Quantum model (default):

```bash
python main.py --config configs/default.yaml
```

Classical GRU baseline:

```bash
python main.py --config configs/classical_gru.yaml
```

Fast smoke run:

```bash
python main.py --config configs/smoke.yaml
python -m unittest tests/test_quantum.py
```

## Quantum feature map

Three qubits represent market factors:

- `q0`: trend / momentum
- `q1`: volatility / risk
- `q2`: RSI / oscillator

The circuit is:

```text
H^⊗3 |000>                     # uniform superposition over 8 basis states
RZ(momentum) RZ(vol) RZ(rsi)   # data as a diagonal phase unitary
CNOT ring                      # entanglement
RY(momentum) RY(vol) RY(rsi)   # second encoding layer
CNOT ring
measure computational basis
```

Basis states are grouped into market regimes:

| Regime | Basis states |
| --- | --- |
| bullish, low vol | `|000>` |
| bullish, high vol | `|001>` |
| neutral | `|010> + |011> + |100>` |
| bearish, low vol | `|101>` |
| bearish, high vol | `|110> + |111>` |

Those probabilities are **measurement outcomes**, not a hand-written softmax.

## Variational quantum predictor

Default circuit: 4 qubits, 2 hardware-efficient layers, 4 data re-uploads.

```text
for t in 1..n_reuploads:
  RY(x[t, i]) on each qubit      # angle encoding
  for each ansatz layer:
    RZ-RY-RZ on every qubit
    CNOT q0->q1->q2->q3->q0     # entanglement
measure <Z_i> and <Z_i Z_{i+1}>
classical linear readout heads
```

Only a thin classical map is used, to fit a lookback window onto a few qubits. Sequence structure is handled by **quantum data re-uploading**, not by a GRU.

## Configuration

The important quantum block in `configs/default.yaml`:

```yaml
quantum:
  encoding: qaoa_feature_map
  ansatz: hardware_efficient
  n_qubits: 4
  n_layers: 2
  n_reuploads: 4
  temperature: 1.0

model:
  type: vqc
```

Set `model.type: classical` to train the old GRU/LSTM baseline.

## Outputs

Each run writes `outputs/YYYYMMDD_HHMMSS/`:

- `quantum_circuit.txt` / `quantum_circuit.json`: gate-level circuit description
- `AAPL/quantum_state_probabilities.csv`: daily Born-rule probabilities
- `AAPL/quantum_regime_probabilities.png`: stacked regime probabilities
- `best_model.pt`, metrics, predictions, and backtest files as before

## Metrics

Prediction: RMSE, MAE, direction accuracy / F1, risk macro-F1  
Backtest: cumulative return, Sharpe, max drawdown, hit rate, active ratio

## Notes

- This is a hybrid NISQ algorithm: quantum evolution + classical parameter updates.
- Exact statevector simulation is exponential in qubit count. The default 3–4 qubits is intentional.
- The same ansatz can be sent to real hardware later (IBM / IonQ) because it only uses RY, RZ, H, and CNOT.
- The classical GRU config is there for ablation, not as the main model.
