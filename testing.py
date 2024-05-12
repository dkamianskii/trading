import pandas as pd
import indicators


if __name__ == "__main__":
    df = pd.read_csv(rf"D:\Trading\raw_data\tickers\AAPL.csv", index_col="Date")
    df["APO"] = indicators.apo(close=df["Close"])
    df["BIAS"] = indicators.bias(close=df["Close"])
    df["DEMA"] = indicators.dema(close=df["Close"])
    df["HULL"] = indicators.hull_ma(close=df["Close"])
    df["MFM"] = indicators.mfm(df)
    df["RVI"] = indicators.rvi(df)
    df["Supertrend"] = indicators.supertrend(df)
    fin = 1