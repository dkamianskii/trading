import argparse
from os import path, makedirs

import yfinance as yf
import pandas as pd
from tqdm import tqdm

from helper.folders import create_empty_sub_fld
from data_config import start_date, end_date


def upload_stocks_history(symbols: str,
                          special: list[str],
                          output_fld: str):
    if not path.exists(output_fld):
        makedirs(output_fld)

    tickers_fld = create_empty_sub_fld(output_fld, "tickers")

    symbols_df = pd.read_csv(symbols)

    for symbol in tqdm(symbols_df["Symbol"]):
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)
        hist.to_csv(path.join(tickers_fld, f"{symbol}.csv"))

    if len(special) != 0:
        special_fld = create_empty_sub_fld(output_fld, "special")
        for symbol in special:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            hist.to_csv(path.join(special_fld, f"{symbol}.csv"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Upload stock history data')
    parser.add_argument('--symbols', dest='symbols', required=True, type=str, default="",
                        help="Path to the csv file with symbols")
    parser.add_argument('--special', dest='special', required=False, nargs='+', default=[],
                        help="Special symbols to upload")
    parser.add_argument('--dst-root', dest='output_fld', required=True, type=str, default="",
                        help="Path to data directory where history data to save")
    args = parser.parse_args()

    upload_stocks_history(symbols=args.symbols,
                          special=args.special,
                          output_fld=args.output_fld)