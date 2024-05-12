import re
import sys

import requests
from bs4 import BeautifulSoup
import pandas as pd

from data_config import selected_sectors, max_foundation_year


def parse_sp500(save_file_path: str):
    r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(r.text, 'html.parser')
    indiatable = soup.find('table', {'class': "wikitable"})
    df = pd.read_html(str(indiatable))
    df = pd.DataFrame(df[0])

    t = df["GICS Sector"] == selected_sectors[0]
    for s in selected_sectors[1:]:
        t = t | (df["GICS Sector"] == s)
    df = df[t]

    df["Founded"] = df["Founded"].apply(lambda x: re.split(' |/', x)[0]).astype(int)
    df = df[df["Founded"] <= max_foundation_year]

    df.to_csv(save_file_path)


if __name__ == "__main__":
    file_path = sys.argv[1]
    parse_sp500(save_file_path=file_path)