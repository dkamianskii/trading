from ta.trend import ema_indicator
import pandas as pd


def apo(close: pd.Series, short_period: int = 10, long_period: int = 20) -> pd.Series:
    return ema_indicator(close=close, window=short_period) - ema_indicator(close=close, window=long_period)