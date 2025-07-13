import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import config

import pandas as pd
import pandas_ta as ta
import pandas_ta.volume as tav
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm      # nice progress bars

# ────────────────────────────  CONFIG  ──────────────────────────── #

FREQ           = "1d"
BASE_DIR       = Path(config.working_dir) / "data"
RAW_DATA_DIR   = BASE_DIR / "raw"   / FREQ
TRAIN_DATA_DIR = BASE_DIR / "train" / FREQ
TEST_DATA_DIR  = BASE_DIR / "test"  / FREQ

TRAIN_YEARS    = (2017, 2022)   # inclusive
TEST_YEARS     = (2023, 2024)

WINDOW_SIZE    = 11
RELABEL_RANGE  = 0

# ───────────────────────  FEATURE ENGINEERING  ──────────────────── #

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DF with all technical indicators + label column (no NaNs)."""
    ind = pd.DataFrame({
        "RSI" : ta.rsi(df.Close),
        "WIL" : ta.willr(df.High, df.Low, df.Close),
        "WMA" : ta.wma(df.Close),
        "EMA" : ta.ema(df.Close),
        "SMA" : ta.sma(df.Close),
        "HMA" : ta.hma(df.Close),
        "TMA" : ta.tema(df.Close),
        "CCI" : ta.cci(df.High, df.Low, df.Close),
        "CMO" : ta.cmo(df.Close),
        "MCD" : ta.macd(df.Close)["MACD_12_26_9"],
        "PPO" : ta.ppo(df.Close)["PPO_12_26_9"],
        "ROC" : ta.roc(df.Close),
        "CMF" : tav.cmf(df.High, df.Low, df.Close, df.Volume),
        "ADX" : ta.adx(df.High, df.Low, df.Close)["ADX_14"],
        "PSA" : ta.psar(df.High, df.Low)["PSARaf_0.02_0.2"],
        "Label": df.Label
    }, index=df.index)

    # drop warm-up rows (60 is original choice)
    return ind.iloc[60:].dropna(how="any")

# ───────────────────────  CORE PIPELINE STEPS  ──────────────────── #

def read_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)

def label_df(df: pd.DataFrame) -> pd.DataFrame:
    # your existing sliding_window_labeling (renamed for clarity)
    from label_function import sliding_window_labeling
    return sliding_window_labeling(df, window_size=WINDOW_SIZE,
                                   relabel_range=RELABEL_RANGE)

def process_symbol(path: Path) -> tuple[str, pd.DataFrame]:
    """Read → label → add indicators.  Return symbol name and engineered DF."""
    raw   = read_raw(path)
    raw   = label_df(raw)
    feats = compute_indicators(raw)
    feats["Close"] = raw.Close      # keep close for later writing
    return path.stem, feats

# ────────────────────────────  I/O HELPERS  ─────────────────────── #

def write_split(df: pd.DataFrame, symbol: str):
    """Split df by year boundaries & write to csv."""
    def millis(y):  # convert yyyy to epoch-ms
        return int(pd.Timestamp(f"{y}-01-01").timestamp()*1000)

    ranges = {
        "train": (millis(TRAIN_YEARS[0]), millis(TRAIN_YEARS[1]+1)-1),
        "test" : (millis(TEST_YEARS[0]),  millis(TEST_YEARS[1]+1)-1)
    }
    for split, (lo, hi) in ranges.items():
        outdir = TRAIN_DATA_DIR if split=="train" else TEST_DATA_DIR
        outdir.mkdir(parents=True, exist_ok=True)
        df.loc[lo:hi].to_csv(outdir/f"{symbol}_{lo}_{hi}_{split}.csv")

# ─────────────────────────────  DRIVER  ─────────────────────────── #

def preprocess(parallel: bool = True, max_workers: int | None = None):
    files = sorted(RAW_DATA_DIR.glob("*.csv"))
    print(f"· Found {len(files)} symbol files")

    # ----------  Stage 1: feature engineering (parallel) ----------
    if parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            data = list(tqdm(ex.map(process_symbol, files), total=len(files)))
    else:
        data = [process_symbol(p) for p in tqdm(files)]

    symbols, frames = zip(*data)

    # ----------  Stage 2: global scaling ----------
    full = pd.concat(frames)
    scaler = MinMaxScaler()
    full.iloc[:, :-2] = scaler.fit_transform(full.iloc[:, :-2])   # exclude Label, Close

    # broadcast the scaled values back to each symbol frame
    start = 0
    for sym, frame in zip(symbols, frames):
        stop = start + len(frame)
        frame.iloc[:, :-2] = full.iloc[start:stop, :-2].to_numpy()
        write_split(frame, sym)
        start = stop

    print("✓ preprocessing complete")

# ───────────────────────────────  MAIN  ─────────────────────────── #

if __name__ == "__main__":
    preprocess()
