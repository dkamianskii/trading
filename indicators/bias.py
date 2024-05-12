from ta.trend import sma_indicator
import pandas as pd


def bias(close: pd.Series, period: int = 12) -> pd.Series:
    avg = sma_indicator(close=close, window=period)
    return 100 * (close - avg) / avg

