import pandas as pd
import numpy as np
from trade_actions import TradeAction


def mark_trade_actions(stock_history: pd.DataFrame,
                       price_change_threshold: float,
                       opening_change_level: float,
                       max_days_without_confirm: int,
                       opposite_dir_days_perc_threshold: float,
                       opposite_change_threshold: float,
                       max_days_without_new_high: int,
                       days_before_end_to_stop: int) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    trade_actions = pd.Series(data=TradeAction.NONE, index=stock_history.index)
    position_profits = []
    year_yields = []
    cur_year_yield = 0
    cur_year = stock_history.index[0].year

    open_i = 0
    while open_i < len(stock_history) - days_before_end_to_stop:
        if stock_history.index[open_i].year != cur_year:
            year_yields.append(cur_year_yield)
            cur_year = stock_history.index[open_i].year
            cur_year_yield = 0

        intro_day = stock_history.iloc[open_i]["Close"] - stock_history.iloc[open_i]["Open"]
        next_intro_day = stock_history.iloc[open_i + 1]["Close"] - stock_history.iloc[open_i + 1]["Open"]
        days_change = (stock_history.iloc[open_i + 1]["Close"] - stock_history.iloc[open_i]["Close"]) / stock_history.iloc[open_i]["Close"]
        if np.sign(intro_day) == np.sign(next_intro_day) \
            and np.sign(intro_day) == np.sign(days_change) \
            and np.abs(days_change) >= price_change_threshold:
            open_price = stock_history.iloc[open_i]["Close"]
            change_dir = np.sign(intro_day)
            opening_confirmed = False
            intro_day_opposite_counter = 0
            opposite_change_sqrt_sum = 0
            indir_change_sqrt_sum = np.sqrt(np.abs(stock_history.iloc[open_i + 1]["Close"] - stock_history.iloc[open_i]["Close"]))
            max_cum_change = 0
            days_without_new_high = 0

            close_i = 0
            j = open_i + 2
            while j < len(stock_history) and \
                    (j - open_i <= max_days_without_confirm or
                     (opening_confirmed and days_without_new_high < max_days_without_new_high)):
                cur_day = stock_history.iloc[j]
                prev_day = stock_history.iloc[j - 1]
                change = cur_day["Close"] - prev_day["Close"]
                cum_change = np.abs((cur_day["Close"] - open_price) / open_price)

                if opening_confirmed:
                    if cum_change > max_cum_change:
                        max_cum_change = cum_change
                        close_i = j
                        days_without_new_high = 0
                    else:
                        days_without_new_high += 1
                else:
                    if cum_change >= opening_change_level and np.sign(change) == change_dir:
                        opening_confirmed = True
                        max_cum_change = cum_change
                        close_i = j
                    else:
                        if np.sign(change) == change_dir:
                            indir_change_sqrt_sum += np.sqrt(np.abs(change))
                        else:
                            opposite_change_sqrt_sum += np.sqrt(np.abs(change))

                        if cur_day["Close"] - cur_day["Open"] != change_dir:
                            intro_day_opposite_counter += 1

                        if cum_change < price_change_threshold or \
                            intro_day_opposite_counter / (j - open_i) > opposite_dir_days_perc_threshold or \
                                opposite_change_sqrt_sum / indir_change_sqrt_sum > opposite_change_threshold:
                            break
                j += 1
            if opening_confirmed:
                position_profits.append(max_cum_change)
                cur_year_yield += max_cum_change
                trade_actions[stock_history.index[open_i]] = TradeAction.LONG if change_dir > 0 else TradeAction.SHORT
                trade_actions[stock_history.index[close_i]] = TradeAction.CLOSE_LONG if change_dir > 0 else TradeAction.CLOSE_SHORT
                open_i = close_i - 1
        open_i += 1

    return trade_actions, np.array(position_profits), np.array(year_yields)


if __name__ == "__main__":
    df = pd.read_csv(rf"D:\Trading\raw_data\special\SP500.csv", index_col="Date", parse_dates=True)
    ta, pos_prof, year_yields = mark_trade_actions(stock_history=df,
                       price_change_threshold=0.002,
                       opening_change_level=0.02,
                       max_days_without_confirm=8,
                       opposite_dir_days_perc_threshold=0.7,
                       opposite_change_threshold=0.5,
                       max_days_without_new_high=5,
                       days_before_end_to_stop=10)
    df_price = df[["Close"]]
    df_price["Action"] = ta
    print("1")