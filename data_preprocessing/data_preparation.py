import pandas as pd
import numpy as np
import os

from tqdm import tqdm

from data_scripts.data_config import tickers_list_file, features_dir, test_start_date, train_data_dir


def create_train_test_data(features_df: pd.DataFrame,
                           test_start_date,
                           days_in_observation: int,
                           stride: int,
                           target: str):
    data_for_x = features_df.drop([target], axis=1)
    test_start_date_idx = (features_df.index[days_in_observation - 1:] >= test_start_date).argmax()

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i in range(days_in_observation, features_df.shape[0] + 1, stride):
        x_frame = data_for_x.iloc[i - days_in_observation:i]
        x_train.append(x_frame.to_numpy())
        y_train.append(features_df[target][i - 1])
        if i >= test_start_date_idx:
            x_test.append(x_frame.to_numpy())
            y_test.append(features_df[target][i - 1])

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    target_name = "5d_change"
    symbols = pd.read_csv(tickers_list_file)
    data_train = []
    data_val = []
    target_train = []
    target_val = []

    for ticker in tqdm(symbols['Symbol']):
        print(ticker)
        data = pd.read_csv(os.path.join(features_dir, f"{ticker}.csv"), parse_dates=True, index_col="Date")
        x_train, y_train, x_test, y_test = create_train_test_data(data, test_start_date, 60, 5, target_name)
        data_train.append(x_train)
        data_val.append(x_test)
        target_train.append(y_train)
        target_val.append(y_test)

    data_train = np.concatenate(data_train, axis=0)
    data_val = np.concatenate(data_val, axis=0)
    target_train = np.concatenate(target_train, axis=0)
    target_val = np.concatenate(target_val, axis=0)

    np.save(os.path.join(train_data_dir, "train_data"), data_train)
    np.save(os.path.join(train_data_dir, "val_data"), data_val)
    np.save(os.path.join(train_data_dir, f"train_{target_name}"), target_train)
    np.save(os.path.join(train_data_dir, f"val_{target_name}"), target_val)


