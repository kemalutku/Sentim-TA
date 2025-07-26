"""Preprocess raw finance CSVs into a single feature parquet file."""

from pathlib import Path
import pandas as pd
import pandas_ta as ta
import pandas_ta.volume as tav
import numpy as np


INDICATORS = [
    "RSI", "WIL", "WMA", "EMA", "SMA", "HMA", "TMA", "CCI", "CMO", "MCD",
    "PPO", "ROC", "CMF", "ADX", "PSA",
]


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    ind = pd.DataFrame({
        "RSI": ta.rsi(df.Close),
        "WIL": ta.willr(df.High, df.Low, df.Close),
        "WMA": ta.wma(df.Close),
        "EMA": ta.ema(df.Close),
        "SMA": ta.sma(df.Close),
        "HMA": ta.hma(df.Close),
        "TMA": ta.tema(df.Close),
        "CCI": ta.cci(df.High, df.Low, df.Close),
        "CMO": ta.cmo(df.Close),
        "MCD": ta.macd(df.Close)["MACD_12_26_9"],
        "PPO": ta.ppo(df.Close)["PPO_12_26_9"],
        "ROC": ta.roc(df.Close),
        "CMF": tav.cmf(df.High, df.Low, df.Close, df.Volume),
        "ADX": ta.adx(df.High, df.Low, df.Close)["ADX_14"],
        "PSA": ta.psar(df.High, df.Low)["PSARaf_0.02_0.2"],
    }, index=df.index)
    return ind


def process_file(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    ind = compute_indicators(df)
    df = pd.concat([
        df[["Open", "High", "Low", "Close", "Volume"]],
        ind,
        df[["Label"]],
    ], axis=1).dropna()
    return df.drop(columns="Label"), df["Label"]


def load_sources(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    feats, lbls = [], []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            items = sorted(p.glob("*.csv"))
        else:
            items = [p]
        for item in items:
            f, l = process_file(item)
            feats.append(f)
            lbls.append(l)
    feat = pd.concat(feats, ignore_index=True).astype(np.float32)
    lab = pd.concat(lbls, ignore_index=True).astype(np.int64)
    return feat.to_numpy(), lab.to_numpy()


def main(out_path: str, sources: list[str]):
    feat, lab = load_sources([Path(s) for s in sources])
    df = pd.DataFrame(feat, columns=[
        "Open", "High", "Low", "Close", "Volume", *INDICATORS
    ])
    df["Label"] = lab
    df.to_parquet(out_path, index=False)
    print(f"✓ wrote {out_path} with shape {df.shape}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+", help="CSV files or directories")
    ap.add_argument("out", help="output parquet path")
    args = ap.parse_args()
    main(args.out, args.sources)
