import os
import io
import zipfile
import requests
import pandas as pd
import yfinance as yf
from multiprocessing import Pool
import shutil
from data_finance.tickers import *

# ─── Constants ────────────────────────────────────────────────────────────────
BINANCE_HEADER = [
    "Open time", "Open", "High", "Low", "Close", "Volume",
    "Close time", "Quote asset volume", "Number of trades",
    "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
]


# ─── Yahoo Finance Downloader ─────────────────────────────────────────────────

def download_yfinance_data(symbol, frequency, start, end, extract_to):
    """
    Downloads historical OHLCV data from Yahoo Finance and saves it as a CSV.
    """
    print(f"Downloading {symbol} from {start} to {end} @ {frequency} (Yahoo Finance)...")
    try:
        df = yf.download(symbol, start=start, end=end, interval=frequency, auto_adjust=True)
        if df.empty:
            print(f"  → No data for {symbol}")
            return
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.index = pd.to_datetime(df.index).astype("int64") // 10 ** 6
        df.index.name = None
        os.makedirs(extract_to, exist_ok=True)
        out_path = os.path.join(extract_to, f"{symbol}.csv")
        df.to_csv(out_path, index=True)
        print(f"  → Saved to {out_path}")
    except Exception as e:
        print(f"  ✗ Failed {symbol}: {e}")


# ─── Binance Helpers ─────────────────────────────────────────────────────────

def generate_binance_links(bin_symbol, frequency, start_year, end_year):
    base = f"https://data.binance.vision/data/spot/monthly/klines/{bin_symbol}/{frequency}/"
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yield f"{base}{bin_symbol}-{frequency}-{year}-{month:02d}.zip"


def download_and_extract_binance(url, extract_to):
    """
    Downloads a ZIP from `url`, extracts CSVs into `extract_to`,
    then normalizes each CSV header.
    """
    resp = requests.get(url, stream=True)
    if resp.status_code != 200:
        return
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(extract_to)
            for member in zf.namelist():
                if not member.lower().endswith('.csv'):
                    continue
                csv_path = os.path.join(extract_to, member)
                df = pd.read_csv(csv_path, header=None)
                df.columns = BINANCE_HEADER
                df.to_csv(csv_path, index=False)
    except zipfile.BadZipFile:
        pass


# ─── Unified Binance Downloader ───────────────────────────────────────────────

def download_binance_data(symbol, frequency, start, end, temp_dir, output_dir, parallel=False):
    """
    Downloads Klines from Binance, merges them, filters by date range,
    and saves a single CSV matching the yfinance format.
    If parallel=True, downloads monthly files via multiprocessing Pool.
    Uses a shared temp_dir for all symbols and clears it per symbol.
    """
    bin_symbol = symbol.replace('-', '')
    if bin_symbol.endswith('USD'):
        bin_symbol = bin_symbol[:-3] + 'USDT'
    print(f"Downloading {symbol} ({bin_symbol}) from {start} to {end} @ {frequency} (Binance)...")
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    # Clear any previous symbol files in temp_dir
    if os.path.isdir(temp_dir):
        for fname in os.listdir(temp_dir):
            fpath = os.path.join(temp_dir, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
    else:
        os.makedirs(temp_dir, exist_ok=True)

    # Prepare URLs and tasks
    urls = list(generate_binance_links(bin_symbol, frequency, start_dt.year, end_dt.year))
    tasks = []
    for url in urls:
        csv_name = url.split('/')[-1].replace('.zip', '.csv')
        csv_path = os.path.join(temp_dir, csv_name)
        if not os.path.exists(csv_path):
            tasks.append((url, temp_dir))
        else:
            print(f"  → Skipping existing {csv_name}")

    # Download sequentially or in parallel
    if parallel and tasks:
        with Pool() as pool:
            pool.starmap(download_and_extract_binance, tasks)
    else:
        for url, _ in tasks:
            download_and_extract_binance(url, temp_dir)

    # Merge and filter only current symbol’s files
    csv_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir)
                 if f.lower().endswith('.csv') and f.startswith(bin_symbol)]
    if not csv_files:
        print(f"  → No Binance data downloaded for {symbol}")
        return
    df = pd.concat((pd.read_csv(f) for f in sorted(csv_files)), ignore_index=True)
    df = df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df.set_index('Open time', inplace=True)
    df.sort_index(inplace=True)
    df = df[start:end]
    df.index = df.index.astype('int64') // 10 ** 6
    df.index.name = None
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Save merged CSV
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{symbol}.csv")
    df.to_csv(out_path, index=True)
    print(f"  → Saved to {out_path}")


# ─── Main Configuration & Entry Point ────────────────────────────────────────

def main():
    source = 'yfinance'  # 'yfinance' or 'binance'
    symbols = dow_30
    frequency = '1d'
    base_dir = r'D:\CnnTA\v2'

    start_date = '2017-01-01'
    end_date = '2024-12-31'

    if source.lower() == 'yfinance':
        out_base = os.path.join(base_dir, 'yfinance', frequency)
        for sym in symbols:
            download_yfinance_data(sym, frequency, start_date, end_date, out_base)
    elif source.lower() == 'binance':
        temp_dir = os.path.join(base_dir, 'binance', frequency, '_temp')
        output_dir = os.path.join(base_dir, 'binance', frequency)
        os.makedirs(temp_dir, exist_ok=True)
        for sym in symbols:
            download_binance_data(sym, frequency, start_date, end_date,
                                  temp_dir, output_dir,
                                  parallel=True)  # set to False to ease server load
    else:
        raise ValueError(f"Unknown source: {source!r}")


if __name__ == '__main__':
    import config
    main()
