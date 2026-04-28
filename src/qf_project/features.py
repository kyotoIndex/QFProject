from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_series(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan)


def compute_rsi(prices: pd.Series, window: int) -> pd.Series:
    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.rolling(window=window).mean()
    average_loss = losses.rolling(window=window).mean()
    rs = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def engineer_features(
    frame: pd.DataFrame,
    volatility_window: int,
    momentum_window: int,
    bollinger_window: int,
    rsi_window: int,
) -> pd.DataFrame:
    engineered = frame.copy().sort_index()
    price = engineered["adj_close"]

    engineered["return"] = price.pct_change()
    engineered["log_return"] = np.log(price / price.shift(1))
    engineered["momentum"] = price.pct_change(momentum_window)
    engineered["rolling_mean"] = price.rolling(momentum_window).mean() / price - 1.0
    engineered["volatility"] = engineered["return"].rolling(volatility_window).std()
    engineered["volume_change"] = engineered["volume"].pct_change()
    engineered["high_low_range"] = (engineered["high"] - engineered["low"]) / engineered["close"]
    engineered["rsi"] = compute_rsi(price, rsi_window)

    ema_short = price.ewm(span=12, adjust=False).mean()
    ema_long = price.ewm(span=26, adjust=False).mean()
    engineered["macd"] = ema_short - ema_long
    engineered["macd_signal"] = engineered["macd"].ewm(span=9, adjust=False).mean()

    rolling_mean = price.rolling(bollinger_window).mean()
    rolling_std = price.rolling(bollinger_window).std()
    upper_band = rolling_mean + 2 * rolling_std
    lower_band = rolling_mean - 2 * rolling_std
    band_width = (upper_band - lower_band).replace(0, np.nan)
    engineered["bollinger_position"] = (price - rolling_mean) / band_width

    engineered = engineered.replace([np.inf, -np.inf], np.nan)
    engineered = engineered.dropna().copy()

    for column in engineered.columns:
        engineered[column] = _safe_series(engineered[column]).astype(float)

    return engineered
