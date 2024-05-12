import pandas as pd


def mfm(stock_history_data: pd.DataFrame) -> pd.Series:
    df = stock_history_data
    return ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"])
