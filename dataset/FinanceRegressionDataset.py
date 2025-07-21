from pathlib import Path
from bisect import bisect_right

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FinanceRegressionDataset(Dataset):
    """Finance dataset returning regression label for each window."""

    def __init__(self, root: str | Path, feature_cols: list[str], sort_by_date=True, return_symbol=False):
        self.root = p = Path(root)
        self.feature_cols = feature_cols
        self.sequence_len = 15
        self.return_symbol = return_symbol

        if p.is_file():
            files = [p]
        else:
            files = sorted(p.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {p}")

        self._symbols = []
        self._arrays = []
        self._labels = []
        self._timestamps = []
        self._closes = []
        self._lengths = []

        for csv_path in files:
            df = pd.read_csv(csv_path)
            self._symbols.append(csv_path.stem)
            if sort_by_date and "Date" in df.columns:
                df = df.sort_values("Date")
            df = df.iloc[self.sequence_len:]
            usable = len(df) - self.sequence_len
            if usable <= 0:
                continue
            self._arrays.append(df[self.feature_cols].to_numpy(dtype=np.float32))
            self._labels.append(df["RegLabel"].to_numpy(dtype=np.float32))
            self._timestamps.append(df["Date"].to_numpy())
            self._closes.append(df["Close"].to_numpy(dtype=np.float32))
            self._lengths.append(usable)

        self._cum = np.cumsum(self._lengths).tolist()

    def __len__(self):
        return self._cum[-1] if self._cum else 0

    def __getitem__(self, idx: int):
        file_idx = bisect_right(self._cum, idx)
        offset = idx - (self._cum[file_idx - 1] if file_idx else 0)

        x_arr = self._arrays[file_idx]
        y_arr = self._labels[file_idx]
        t_arr = self._timestamps[file_idx]
        c_arr = self._closes[file_idx]

        start, end = offset, offset + self.sequence_len
        x = torch.from_numpy(x_arr[start:end]).view(1, self.sequence_len, -1)
        y = torch.tensor(y_arr[end], dtype=torch.float32)
        timestamp = torch.tensor(t_arr[end])
        close = torch.tensor(c_arr[end])

        if self.return_symbol:
            return self._symbols[file_idx], timestamp, close, x, y
        return timestamp, close, x, y
