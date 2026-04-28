# Quantum Finance Project Usage Guide

## Overview
This project implements a neural-network-based quantum finance system for market prediction and risk analysis. It combines conventional financial time-series features with quantum-inspired market state encoding, then uses a GRU or LSTM model to predict:

- next-period return
- next-period direction
- risk level
- simple trading signals for backtesting

The full pipeline is executed from `main.py`.

## Project Structure

```text
QF Group Project/
├── configs/
│   └── default.yaml
├── data_cache/
├── outputs/
├── src/
│   └── qf_project/
│       ├── __init__.py
│       ├── backtest.py
│       ├── data.py
│       ├── dataset.py
│       ├── evaluate.py
│       ├── features.py
│       ├── model.py
│       ├── quantum_encoding.py
│       ├── train.py
│       └── utils.py
├── main.py
└── requirements.txt
```

## Environment Setup

### Option 1: Use the existing conda environment
If you already have a conda environment named `torch`, use:

```bash
conda run -n torch python -c "import torch, pandas, numpy, sklearn, yfinance, yaml, matplotlib; print('deps-ok')"
```

### Option 2: Install dependencies manually
Install the required Python packages:

```bash
pip install -r requirements.txt
```

Dependencies listed in `requirements.txt`:

- torch
- pandas
- numpy
- scikit-learn
- yfinance
- pyyaml
- matplotlib

## How to Run
Run the full pipeline with the default configuration:

```bash
python main.py --config configs/default.yaml
```

If you want to use the conda `torch` environment directly:

```bash
conda run -n torch python main.py --config configs/default.yaml
```

## What the Pipeline Does
The pipeline in `main.py` performs the following steps:

1. Download market data with `yfinance`
2. Cache raw data into `data_cache/`
3. Generate technical indicators and financial features
4. Encode quantum-inspired market state probabilities
5. Build rolling window datasets
6. Train a GRU or LSTM multi-task model
7. Evaluate validation and test performance
8. Run a simple trading backtest
9. Save all outputs into a timestamped folder under `outputs/`

## Configuration
The main configuration file is:

- `configs/default.yaml`

### Important config sections

#### Data
```yaml
data:
  symbols:
    - AAPL
    - SPY
  start_date: "2015-01-01"
  end_date: "2025-12-31"
  interval: 1d
  cache_dir: data_cache
```

- `symbols`: assets to train and evaluate
- `start_date`, `end_date`: data range
- `interval`: time interval for download
- `cache_dir`: local CSV cache directory

#### Feature settings
```yaml
features:
  lookback: 30
  horizon: 1
  volatility_window: 20
  momentum_window: 10
  bollinger_window: 20
  rsi_window: 14
  train_ratio: 0.7
  val_ratio: 0.15
```

- `lookback`: number of past timesteps used as model input
- `horizon`: next-step prediction horizon
- rolling windows define indicator calculation
- `train_ratio` and `val_ratio` control time-based splitting

#### Quantum-inspired encoding
```yaml
quantum:
  states:
    - bullish_low_vol
    - bullish_high_vol
    - neutral
    - bearish_low_vol
    - bearish_high_vol
  temperature: 1.0
```

This module converts market conditions into probability-like state vectors representing uncertainty and regime superposition.

#### Model
```yaml
model:
  recurrent_type: gru
  hidden_size: 64
  num_layers: 2
  dropout: 0.2
  input_projection_size: 64
  mlp_hidden_size: 64
```

- `recurrent_type`: choose `gru` or `lstm`
- other parameters control model capacity

#### Training
```yaml
training:
  batch_size: 64
  epochs: 30
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 5
  gradient_clip_norm: 1.0
```

#### Strategy
```yaml
strategy:
  buy_threshold: 0.002
  sell_threshold: -0.002
  high_risk_class: 2
```

These thresholds convert predictions into buy / hold / sell signals.

## Output Files
Each run creates a new timestamped directory:

```text
outputs/YYYYMMDD_HHMMSS/
```

Inside it, each symbol gets its own folder, for example:

```text
outputs/20260428_161044/
├── AAPL/
│   ├── best_model.pt
│   ├── training_history.csv
│   ├── val_metrics.json
│   ├── val_predictions.csv
│   ├── test_metrics.json
│   ├── test_predictions.csv
│   ├── test_backtest.json
│   └── test_backtest.csv
├── SPY/
│   └── ...
└── summary.json
```

### File meanings
- `best_model.pt`: saved trained model weights
- `training_history.csv`: epoch-level training progress
- `val_metrics.json`: validation metrics
- `test_metrics.json`: test metrics
- `test_predictions.csv`: detailed predictions and targets
- `test_backtest.json`: summary of trading performance
- `test_backtest.csv`: signal-level backtest records
- `summary.json`: combined summary across all symbols

## Metrics
The project currently reports:

### Prediction metrics
- RMSE
- MAE
- Direction accuracy
- Direction F1
- Risk macro F1

### Backtest metrics
- cumulative return
- annualized Sharpe ratio
- maximum drawdown
- hit rate
- active ratio

## How to Switch GRU and LSTM
In `configs/default.yaml`, change:

```yaml
model:
  recurrent_type: gru
```

to:

```yaml
model:
  recurrent_type: lstm
```

Then rerun:

```bash
python main.py --config configs/default.yaml
```

## How to Change Assets
To test different stocks or indices, edit:

```yaml
data:
  symbols:
    - AAPL
    - SPY
```

Examples:

```yaml
data:
  symbols:
    - MSFT
    - QQQ
```

or:

```yaml
data:
  symbols:
    - TSLA
    - ^GSPC
```

## Notes
- The project is quantum-inspired, not quantum-computing-based.
- It does not require quantum hardware or quantum libraries.
- Market state uncertainty is represented through probability-style feature encoding.
- The current implementation is suitable for coursework, demonstration, and further model experimentation.

## Recommended Next Steps
Possible improvements include:

- add more assets and sector indices
- compare GRU vs LSTM systematically
- tune lookback window and hidden size
- improve signal generation logic
- add visualization for prediction and backtest curves
- extend the report with economic interpretation of the quantum-inspired states
