import yfinance as yf
import pandas as pd
from typing import List

def download_stock_data(ticker: str, period: str = '5y') -> pd.DataFrame:
    """Download historical stock data from Yahoo Finance."""
    df = yf.download(ticker, period=period)
    df.reset_index(inplace=True)
    return df
