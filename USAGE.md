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
│   ├── training_history.png
│   ├── val_metrics.json
│   ├── val_predictions.csv
│   ├── val_predictions.png
│   ├── test_metrics.json
│   ├── test_predictions.csv
│   ├── test_predictions.png
│   ├── risk_confusion_matrix.png
│   ├── test_backtest.json
│   ├── test_backtest.csv
│   ├── test_backtest.png
│   └── signal_distribution.png
├── SPY/
│   └── ...
└── summary.json
```

### File meanings
- `best_model.pt`: saved trained model weights
- `training_history.csv`: epoch-level training progress
- `training_history.png`: train loss and validation score chart
- `val_metrics.json`: validation metrics
- `val_predictions.csv`: validation predictions and targets with timestamps
- `val_predictions.png`: validation return and direction-probability chart
- `test_metrics.json`: test metrics
- `test_predictions.csv`: test predictions and targets with timestamps
- `test_predictions.png`: test return and direction-probability chart
- `risk_confusion_matrix.png`: confusion matrix for three risk classes
- `test_backtest.json`: summary of trading performance
- `test_backtest.csv`: signal-level backtest records with timestamps
- `test_backtest.png`: cumulative strategy vs buy-and-hold chart
- `signal_distribution.png`: buy / hold / sell signal count chart
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

## Visualization Guide
Visualization is especially important for this project because it helps explain both the predictive performance and the quantum-inspired financial interpretation. For a course report or presentation, you should not only report metrics, but also show figures that make the system behavior intuitive.

### 1. Price series and train/validation/test split
Recommended figure:
- line chart of adjusted close price over time
- use vertical separators or background shading to mark train, validation, and test segments

Why it matters:
- shows the full market period being studied
- makes the time-based split transparent
- demonstrates that the model is trained on past data and tested on future data

What to explain in the presentation:
- the model follows a realistic chronological split rather than random shuffling
- this is important for financial forecasting tasks

### 2. Technical indicator visualization
Recommended figure:
- price with moving average overlay
- RSI in a lower subplot
- MACD and MACD signal in another subplot
- Bollinger band position or volatility in another subplot

Why it matters:
- shows what market information is fed into the neural network
- helps the audience understand that the model input is not only raw price, but also derived financial structure

Good interpretation angle:
- technical indicators summarize trend, momentum, and volatility
- they serve as the classical input space before the quantum-inspired encoding step

### 3. Quantum-inspired state probability plot
Recommended figure:
- stacked area chart or multi-line chart of the five state probabilities:
  - bullish_low_vol
  - bullish_high_vol
  - neutral
  - bearish_low_vol
  - bearish_high_vol

Why it matters:
- this is the most important figure for explaining the "quantum finance" aspect
- it visualizes state superposition: the market is not forced into one hard label, but represented as a probability distribution across multiple regimes

What to say in the report:
- unlike classical regime classification, the market can simultaneously exhibit partial bullish and bearish characteristics
- this probabilistic representation captures uncertainty and changing market conditions

Suggested interpretation examples:
- when bullish-high-vol rises, the market may have upward momentum but elevated uncertainty
- when neutral dominates, the market may be range-bound or indecisive
- when bearish-high-vol dominates, downside risk is stronger

### 4. Predicted return vs actual return
Recommended figure:
- line chart comparing predicted return and actual return on the validation or test set
- optionally smooth both series with a short moving average for clearer visual comparison

Why it matters:
- directly shows whether the model tracks the direction and magnitude of return movements
- easier to interpret than only looking at RMSE or MAE

What to emphasize:
- exact point prediction in finance is hard, so trend consistency and directional alignment are often more meaningful than perfect overlap

### 5. Direction prediction visualization
Recommended figure:
- bar chart or scatter plot of predicted direction probability vs actual up/down label
- confusion matrix for up/down classification

Why it matters:
- helps explain direction accuracy and direction F1
- shows whether the model is biased toward predicting mostly up or mostly down

What to discuss:
- if predicted probabilities stay close to 0.5, the model is uncertain
- if confusion matrix is imbalanced, the classifier may need threshold tuning or class balancing

### 6. Risk classification visualization
Recommended figure:
- confusion matrix for low / medium / high risk classes
- bar chart of predicted vs actual class counts

Why it matters:
- risk classification is one of the key outputs of this project
- a confusion matrix clearly shows which risk regimes are hardest to distinguish

What to discuss:
- low and high risk states may be easier to separate than medium risk
- macro-F1 is useful because it accounts for class imbalance better than plain accuracy

### 7. Backtest cumulative return curve
Recommended figure:
- cumulative return curve of the strategy over the test period
- compare against a buy-and-hold benchmark if possible

Why it matters:
- connects machine learning outputs to practical financial decision-making
- shows whether the generated buy/hold/sell signals would have produced useful investment behavior

What to explain:
- cumulative return summarizes total strategy growth
- if available, adding a benchmark line improves the credibility of the analysis

### 8. Drawdown curve
Recommended figure:
- drawdown over time, plotted below the cumulative return curve

Why it matters:
- return alone is not enough in finance
- drawdown reveals downside risk and periods of strategy stress

What to explain:
- maximum drawdown is one of the most important risk metrics
- two strategies with similar return can have very different drawdown behavior

### 9. Training history plot
Recommended figure:
- epoch vs training loss
- epoch vs validation score

Why it matters:
- shows whether the model converges normally
- helps identify underfitting, overfitting, or unstable training

What to discuss:
- if training loss falls but validation score worsens, overfitting may be happening
- early stopping behavior can also be explained here

### 10. Multi-asset comparison figure
Recommended figure:
- grouped bar chart comparing AAPL and SPY on RMSE, direction accuracy, risk macro-F1, and cumulative return

Why it matters:
- your project already supports both stock and index examples
- this plot makes it easy to compare model behavior across different market types

What to explain:
- individual stocks may have stronger idiosyncratic volatility
- indices may be smoother and sometimes easier to model

## Recommended Visualization Set for Final Presentation
If you only have room for 4 to 6 charts in the final PPT, the best set is:

1. price series with train/val/test split
2. quantum-inspired state probability chart
3. predicted return vs actual return
4. risk classification confusion matrix
5. cumulative return backtest curve
6. AAPL vs SPY comparison bar chart

This combination gives a balanced story:
- data foundation
- quantum-inspired contribution
- prediction quality
- risk modeling
- investment relevance
- comparison across assets

## How to Build Visualizations from Current Outputs
The current code already saves the main files needed for plotting:

- `summary.json`
- `training_history.csv`
- `val_predictions.csv`
- `test_predictions.csv`
- `test_backtest.csv`

### Useful plotting sources
- use `training_history.csv` for loss and validation-score curves
- use `test_predictions.csv` for predicted vs actual return, direction, and risk plots
- use `test_backtest.csv` for cumulative return and signal analysis
- use `summary.json` for AAPL vs SPY comparison charts

### Example columns already available
From `test_predictions.csv`:
- `predicted_return`
- `predicted_direction_probability`
- `predicted_risk_class`
- `actual_return`
- `actual_direction`
- `actual_risk_class`

From `test_backtest.csv`:
- `signal`
- `realized_return`
- `strategy_return`
- `cumulative_curve`

## Suggested Figure Captions
You can directly adapt these caption styles in your report:

- **Figure 1.** Historical price series with chronological training, validation, and test split.
- **Figure 2.** Quantum-inspired market state probabilities showing regime superposition over time.
- **Figure 3.** Comparison of predicted and actual next-period returns on the test set.
- **Figure 4.** Confusion matrix for three-class market risk prediction.
- **Figure 5.** Backtest cumulative return curve of the generated trading strategy.
- **Figure 6.** Performance comparison between stock and index prediction tasks.

## Optional Next Step
At the moment, the project saves numeric outputs but does not yet automatically generate figures. A good next extension is to add a dedicated visualization module, for example:

- `src/qf_project/visualize.py`

That module could automatically save:
- training curves
- prediction comparison plots
- confusion matrices
- cumulative return plots
- quantum state probability charts

This would make the project much stronger for final presentation and report writing.

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
