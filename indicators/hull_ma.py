import numpy as np
import pandas as pd
from ta.trend import wma_indicator


def hull_ma(close: pd.Series, period: int = 9) -> pd.Series:
    raw_hma = 2 * wma_indicator(close=close, window=(period // 2)) - wma_indicator(close=close, window=period)
    return wma_indicator(close=raw_hma, window=int(np.sqrt(period)))

