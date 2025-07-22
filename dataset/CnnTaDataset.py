from pathlib import Path
from bisect import bisect_right

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FinanceImageDataset(Dataset):
    """
    If `root` is a directory → read all *.csv inside.
    If `root` is a file      → treat it as the single source CSV.
    Each sample: (timestamp, close, image-tensor, one-hot label)

    Parameters
    ----------
    sequence_len : int
        Number of days to include in each input window.
    """

    def __init__(
        self,
        root: str | Path,
        feature_cols: list[str],
        sort_by_date: bool = True,
        return_symbol: bool = False,
        sequence_len: int = 15,
    ):

        self.root = p = Path(root)
        self.feature_cols = feature_cols
        self.sequence_len = sequence_len
        self.num_classes = 3
        self.return_symbol = return_symbol

        self._symbols = []

        # ── discover source files ───────────────────────────────────
        if p.is_file():  # ← NEW: single-file mode
            files = [p]
        else:
            files = sorted(p.glob("*.csv"))

        if not files:
            raise FileNotFoundError(f"No CSV files found in {p}")

        # ── load them once into contiguous arrays ───────────────────
        self._arrays: list[np.ndarray] = []  # features
        self._labels: list[np.ndarray] = []
        self._timestamps: list[np.ndarray] = []
        self._closes: list[np.ndarray] = []
        self._lengths: list[int] = []  # usable sample count per file

        for csv_path in files:
            df = pd.read_csv(csv_path)
            self._symbols.append(csv_path.stem)

            if sort_by_date and "Date" in df.columns:
                df = df.sort_values("Date")

            df = df.iloc[self.sequence_len:]  # drop warm-up rows

            usable = len(df) - self.sequence_len
            if usable <= 0:
                continue  # skip empty / too-short files

            self._arrays.append(df[self.feature_cols].to_numpy(dtype=np.float32))
            self._labels.append(df["Label"].to_numpy(dtype=np.int64))
            self._timestamps.append(df["Date"].to_numpy())
            self._closes.append(df["Close"].to_numpy(dtype=np.float32))
            self._lengths.append(usable)

        self._cum = np.cumsum(self._lengths).tolist()  # cumulative lengths

    # ───────────────────────── Dataset API ───────────────────────── #

    def __len__(self) -> int:
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
        y = torch.nn.functional.one_hot(
            torch.tensor(y_arr[end], dtype=torch.long),
            num_classes=self.num_classes
        ).float()

        timestamp = torch.tensor(t_arr[end])
        close = torch.tensor(c_arr[end])

        if self.return_symbol:
            return self._symbols[file_idx], timestamp, close, x, y

        return timestamp, close, x, y
