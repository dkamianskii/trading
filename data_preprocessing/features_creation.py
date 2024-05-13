import pandas as pd
import ta
import os
import warnings
import pickle
import numpy as np

from ta.trend import DPOIndicator, adx_pos, adx_neg, cci, macd_diff, EMAIndicator
from ta.volume import chaikin_money_flow, acc_dist_index, negative_volume_index
from ta.momentum import ROCIndicator, williams_r, stoch_signal, RSIIndicator
from ta.volatility import average_true_range
from tqdm import tqdm

import indicators
from data_scripts.data_config import tickers_list_file, raw_data_dir, features_dir

warnings.filterwarnings("ignore")


def form_features(df: pd.DataFrame, insign_val: float, strong_val: float):
    df_final = df[["High", "Low", "Open", "Close", "Volume"]].diff().copy()
    df_final["Introday change"] = df["Close"] - df["Open"]
    df_final["Introday spread"] = df["High"] - df["Low"]

    ema_periods = [20, 30, 50, 100, 200]
    for period in ema_periods:
        df_final[f"EMA {period}"] = df["Close"] - ta.trend.ema_indicator(df['Close'], window=period)

    min_max_close_periods = [25, 50, 100, 200]
    for period in min_max_close_periods:
        df_final[f"Max {period}"] = df["Close"].rolling(period).max() - df["Close"]
        df_final[f"Min {period}"] = df["Close"] - df["Close"].rolling(period).min()

    std_periods = [14, 25, 50, 75]
    for period in std_periods:
        df_final[f"Std {period}"] = df["Close"].rolling(period).std()

    atr_periods = [14, 28, 56]
    for period in atr_periods:
        df_final[f"ATR {period}"] = average_true_range(df["High"], df["Low"], df["Close"], window=period)

    df_final["DPO"] = DPOIndicator(close=df["Close"]).dpo()
    df_final["CMF"] = chaikin_money_flow(high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"])
    df_final["ADI"] = acc_dist_index(high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"])
    df_final["ROC"] = ROCIndicator(close=df["Close"]).roc()
    df_final["NVI"] = negative_volume_index(close=df["Close"], volume=df["Volume"])
    df_final["DIU"] = adx_pos(high=df["High"], low=df["Low"], close=df["Close"])
    df_final["DID"] = adx_neg(high=df["High"], low=df["Low"], close=df["Close"])
    df_final["WILLR"] = williams_r(high=df["High"], low=df["Low"], close=df["Close"])
    df_final["CCI"] = cci(high=df["High"], low=df["Low"], close=df["Close"])
    df_final["STOH"] = stoch_signal(high=df["High"], low=df["Low"], close=df["Close"])
    df_final["MACD"] = macd_diff(close=df["Close"])
    df_final["RSI"] = RSIIndicator(close=df["Close"]).rsi()
    df_final["ATR"] = average_true_range(high=df["High"], low=df["Low"], close=df["Close"])
    df_final["APO"] = indicators.apo(close=df["Close"])
    df_final["BIAS"] = indicators.bias(close=df["Close"])
    df_final["DEMA"] = indicators.dema(close=df["Close"])
    df_final["HULL"] = indicators.hull_ma(close=df["Close"])
    df_final["MFM"] = indicators.mfm(df)
    df_final["RVI"] = indicators.rvi(df)
    df_final["Supertrend"] = indicators.supertrend(df)
    bollinger_bands = ta.volatility.BollingerBands(close=df["Close"], window=14, window_dev=2)
    df_final["bb_hband"] = bollinger_bands.bollinger_hband() - df["Close"]
    df_final["bb_lband"] = df["Close"] - bollinger_bands.bollinger_lband()

    df['Day of week'] = [d.weekday() for d in df.index]
    df_final[["Mon", "Tues", "Wed", "Thurs", "Fri"]] = 0
    df_final.loc[df['Day of week'] == 0, "Mon"] = 1
    df_final.loc[df['Day of week'] == 1, "Tues"] = 1
    df_final.loc[df['Day of week'] == 2, "Wed"] = 1
    df_final.loc[df['Day of week'] == 3, "Thurs"] = 1
    df_final.loc[df['Day of week'] == 4, "Fri"] = 1

    day_before = df.shift(-1).iloc[:-1]
    before_div = day_before['Dividends'] != 0
    df_final["Day before div"] = 0
    df_final["Div day"] = 0
    df_final.loc[day_before[before_div].index, "Day before div"] = 1
    df_final.loc[df['Dividends'] != 0, 'Div day'] = 1
    df_final["Rel div"] = df['Dividends'] / df['Close']
    rel_div = day_before[before_div]["Dividends"] / day_before[before_div]["Close"]
    df_final["Day before div rel"] = 0
    df_final.loc[day_before[before_div].index, "Day before div rel"] = rel_div

    pct_change_5_days = df["Close"].pct_change(periods=5).shift(-5)
    df_final["5d_change"] = 0
    abs_change = np.abs(pct_change_5_days)

    df_final.loc[(abs_change >= insign_val) & (abs_change < strong_val), "5d_change"] = 4
    df_final.loc[abs_change >= strong_val, "5d_change"] = 8
    df_final.loc[pct_change_5_days < 0, "5d_change"] *= -1

    df_final["Rel div"] = df_final["Rel div"].fillna(0)
    df_final["Day before div rel"] = df_final["Day before div rel"].fillna(0)
    df_final.dropna(inplace=True)
    return df_final


if __name__ == "__main__":
    insign_val = 0.025
    strong_val = 0.05
    symbols = pd.read_csv(tickers_list_file)
    for ticker in tqdm(symbols['Symbol']):
        df = pd.read_csv(os.path.join(raw_data_dir, f"{ticker}.csv"), parse_dates=True, index_col="Date")
        feautres_df = form_features(df, insign_val, strong_val)
        feautres_df.to_csv(os.path.join(features_dir, f"{ticker}.csv"))

