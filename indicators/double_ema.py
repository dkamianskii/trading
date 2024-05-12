import pandas as pd
from ta.trend import ema_indicator


def dema(close: pd.Series, period: int = 50) -> pd.Series:
    ema = ema_indicator(close, window=period)
    return 2 * ema - ema_indicator(ema, window=period)