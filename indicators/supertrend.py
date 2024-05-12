import numpy as np
import pandas as pd
from ta.volatility import average_true_range


def supertrend(stock_history_data: pd.DataFrame,
               period: int = 10,
               multiplier: float = 3) -> pd.Series:
    df = stock_history_data
    atr = average_true_range(high=df["High"], low=df["Low"], close=df["Close"], window=period)
    hla = (df["High"] + df["Low"]) / 2
    bub = hla + multiplier * atr
    blb = hla - multiplier * atr

    close = df["Close"]
    arr_len = len(close)

    fub = np.zeros(arr_len)
    fub[0] = bub[0]
    for i in range(1, arr_len):
        if (bub[i] < fub[i - 1]) or (close[i - 1] > fub[i - 1]):
            fub[i] = bub[i]
        else:
            fub[i] = fub[i - 1]

    flb = np.zeros(arr_len)
    flb[0] = blb[0]
    for i in range(1, arr_len):
        if (blb[i] > flb[i - 1]) or (close[i - 1] < flb[i - 1]):
            flb[i] = blb[i]
        else:
            flb[i] = flb[i - 1]

    st = np.zeros(arr_len)
    st[0] = fub[0]
    for i in range(1, arr_len):
        if (st[i - 1] == fub[i - 1]) and (close[i] <= fub[i]):
            st[i] = fub[i]
        elif (st[i - 1] == fub[i - 1]) and (close[i] >= fub[i]):
            st[i] = flb[i]
        elif (st[i - 1] == flb[i - 1]) and (close[i] >= flb[i]):
            st[i] = flb[i]
        elif (st[i - 1] == flb[i - 1]) and (close[i] <= flb[i]):
            st[i] = fub[i]

    return pd.Series(st, index=stock_history_data.index)
