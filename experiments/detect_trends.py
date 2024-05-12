import os
from os import path

import pandas as pd
import numpy as np
import statsmodels.api as sm


def detect_trends(prices: pd.Series,
                  min_trend_len: int = 15,
                  max_trend_len: int = 100,
                  min_r2: float = 0.8,
                  min_trend_dir: float = 0.005) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trends_len,  trends_val, trends_late_gain, trends_starts = [], [], [], []
    trend_start, trend_end = 0, min_trend_len
    trend_val = 0
    trend_late_gain = 0
    trend_sign = 0
    while trend_end < prices.shape[0]:
        y = prices.iloc[trend_start:trend_end] / prices.iloc[trend_start]
        X = np.arange(0, len(y))
        X = sm.add_constant(X)
        model = sm.OLS(y, X)
        results = model.fit()
        param_sign = np.sign(results.params[1])
        if len(y) > min_trend_len:
            if param_sign > 0:
                trend_late_gain = (prices.iloc[trend_end] - prices.iloc[trend_start + min_trend_len - 1]) / prices.iloc[
                    trend_start + min_trend_len - 1]
            else:
                trend_late_gain = (prices.iloc[trend_start + min_trend_len - 1] - prices.iloc[trend_end]) / prices.iloc[
                    trend_start + min_trend_len - 1]
        if len(y) <= max_trend_len\
                and results.rsquared_adj >= min_r2\
                and np.abs(results.params[1]) >= min_trend_dir\
                and (param_sign == trend_sign or trend_sign == 0):
            trend_val = (prices.iloc[trend_end] - prices.iloc[trend_start]) / prices.iloc[trend_start]
            trend_sign = param_sign
            trend_end += 1
        elif trend_sign != 0:
            trends_len.append(len(y) - 1)
            trends_val.append(trend_val)
            trends_late_gain.append(trend_late_gain)
            trends_starts.append(trend_start)
            trend_sign = 0
            trend_start = trend_end
            trend_end += min_trend_len
        else:
            trend_start += 1
            trend_end += 1

    if trend_sign != 0:
        trends_len.append(prices.shape[0] - trend_start)
        trends_val.append(trend_val)

    return np.array(trends_len), np.array(trends_val), np.array(trends_late_gain), np.array(trends_starts)


if __name__ == "__main__":
    symbols = pd.read_csv("D:\Trading\sp500_symbols_list.csv")
    tickers_path = "D:\\Trading\\raw_data\\tickers"
    tickers_dfs = [pd.read_csv(path.join(tickers_path, f"{ticker}.csv"), parse_dates=True, index_col="Date") for ticker in symbols['Symbol']]
    trends_len, trends_val, trends_gain, trends_starts = detect_trends(tickers_dfs[19]["Close"], min_trend_len=15, max_trend_len=100, min_r2=0.8, min_trend_dir=0.005)
    a = 1