import os
from os import path

import pandas as pd
import numpy as np
from ta.trend import DPOIndicator, adx_pos, adx_neg, cci, macd_diff, EMAIndicator
from ta.volume import chaikin_money_flow, acc_dist_index, negative_volume_index
from ta.momentum import ROCIndicator, williams_r, stoch_signal, RSIIndicator
from ta.volatility import average_true_range
import pygad
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import indicators




if __name__ == "__main__":
    df = pd.read_csv(path.join("D:\\Trading\\raw_data\\special", "SP500.csv"), parse_dates=True, index_col="Date")
    data_df = pd.DataFrame(index=df.index)
    data_df["DPO"] = DPOIndicator(close=df["Close"]).dpo()
    data_df["CMF"] = chaikin_money_flow(high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"])
    data_df["ADI"] = acc_dist_index(high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"])
    data_df["ROC"] = ROCIndicator(close=df["Close"]).roc()
    data_df["NVI"] = negative_volume_index(close=df["Close"], volume=df["Volume"])
    data_df["DIU"] = adx_pos(high=df["High"], low=df["Low"], close=df["Close"])
    data_df["DID"] = adx_neg(high=df["High"], low=df["Low"], close=df["Close"])
    data_df["WILLR"] = williams_r(high=df["High"], low=df["Low"], close=df["Close"])
    data_df["CCI"] = cci(high=df["High"], low=df["Low"], close=df["Close"])
    data_df["STOH"] = stoch_signal(high=df["High"], low=df["Low"], close=df["Close"])
    data_df["MACD"] = macd_diff(close=df["Close"])
    data_df["RSI"] = RSIIndicator(close=df["Close"]).rsi()
    data_df["ATR"] = average_true_range(high=df["High"], low=df["Low"], close=df["Close"])
    data_df["APO"] = indicators.apo(close=df["Close"])
    data_df["BIAS"] = indicators.bias(close=df["Close"])
    data_df["DEMA"] = indicators.dema(close=df["Close"])
    data_df["HULL"] = indicators.hull_ma(close=df["Close"])
    data_df["MFM"] = indicators.mfm(df)
    data_df["RVI"] = indicators.rvi(df)
    data_df["Supertrend"] = indicators.supertrend(df)
    data_df["EMA20"] = EMAIndicator(close=df["Close"], window=20).ema_indicator()
    data_df["EMA50"] = EMAIndicator(close=df["Close"], window=50).ema_indicator()
    data_df["EMA100"] = EMAIndicator(close=df["Close"], window=100).ema_indicator()

    for ind in data_df.columns:
        for i in range(1, 4):
            data_df[f"{ind}_{i}"] = data_df[ind].shift(periods=i)

    data_df["target"] = 0
    pct_change = df["Close"].pct_change(periods=4).shift(-4)
    insign_val = 0.0035
    strong_val = 0.025
    data_df.loc[pct_change >= strong_val, "target"] = 2
    data_df.loc[(pct_change < strong_val) & (pct_change >= insign_val), "target"] = 1
    data_df.loc[(pct_change > -strong_val) & (pct_change <= -insign_val), "target"] = -1
    data_df.loc[pct_change <= -strong_val, "target"] = -2

    data_df.dropna(inplace=True)
    train_data = data_df[:"2020-01-01"]
    train_X = train_data.drop("target", axis=1)
    test_data = data_df["2020-01-01":]
    test_X = test_data.drop("target", axis=1)
    indicators_pull = train_X.columns

    clf = DecisionTreeClassifier(max_depth=20, random_state=42)
    np.random.seed(42)

    def fitness_function(solution, solution_idx):
        selected_indicators = indicators_pull[solution.astype(bool)]
        clf.fit(X=train_X[selected_indicators], y=train_data["target"])
        predicts = clf.predict(X=test_X[selected_indicators])
        acc = accuracy_score(y_true=test_data["target"], y_pred=predicts)
        return acc

    init_pop_p = 0.33
    ga_instance = pygad.GA(num_generations=600,
                           num_parents_mating=2,
                           fitness_func=fitness_function,
                           initial_population=np.random.choice([1, 0], p=[init_pop_p, 1-init_pop_p], size=(20, train_X.shape[1])),
                           num_genes=train_X.shape[1],
                           parent_selection_type="rws",
                           keep_elitism=5,
                           crossover_type="uniform",
                           mutation_type="random",
                           mutation_probability=0.015,
                           gene_space=[1, 0],
                           random_seed=42)
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Accuracy: {solution_fitness}")
    print(f"Indicators:")
    print(indicators_pull[solution.astype(bool)])