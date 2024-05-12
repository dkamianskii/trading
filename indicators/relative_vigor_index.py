import pandas as pd

weights = [1/6, 1/3, 1/3, 1/6]


def rvi(stock_history_data: pd.DataFrame, period: int = 12) -> pd.Series:
    df = stock_history_data
    numerator = (df["Close"] - df["Open"]).rolling(4).apply(lambda x: (x * weights).sum())
    denominator = (df["High"] - df["Low"]).rolling(4).apply(lambda x: (x * weights).sum())
    rvi = numerator.rolling(period).mean() / denominator.rolling(period).mean()
    signal = rvi.rolling(4).apply(lambda x: (x * weights).sum())
    return rvi - signal