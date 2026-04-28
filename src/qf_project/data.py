from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import yfinance as yf

from .utils import ensure_directory


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(0)
    else:
        parsed_columns = []
        for column in normalized.columns:
            column_name = str(column)
            match = re.match(r"\('([^']+)'", column_name)
            if match:
                column_name = match.group(1)
            parsed_columns.append(column_name)
        normalized.columns = parsed_columns
    normalized.columns = [str(column).lower().replace(" ", "_") for column in normalized.columns]
    if "adj_close" not in normalized.columns and "close" in normalized.columns:
        normalized["adj_close"] = normalized["close"]
    return normalized


def download_market_data(
    symbols: list[str],
    start_date: str,
    end_date: str,
    interval: str,
    cache_dir: str,
) -> dict[str, pd.DataFrame]:
    cache_path = ensure_directory(cache_dir)
    data: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        file_path = Path(cache_path) / f"{symbol.replace('^', '')}_{interval}.csv"
        if file_path.exists():
            frame = pd.read_csv(file_path, index_col=0, parse_dates=True)
        else:
            downloaded = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False,
                progress=False,
            )
            if downloaded.empty:
                raise ValueError(f"No data downloaded for symbol {symbol}")
            frame = _normalize_columns(downloaded)
            frame.to_csv(file_path)
        data[symbol] = _normalize_columns(frame).sort_index()

    return data
