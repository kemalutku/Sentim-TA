from pathlib import Path
from bisect import bisect_right
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MultiModalDataset(Dataset):
    """
    Loads paired finance and sentiment CSVs for each symbol.
    Each sample: (timestamp, close, 2-channel 15x15 image, one-hot label)
    Channel 0: finance features, Channel 1: sentiment features
    """

    def __init__(self, finance_root: str | Path, sentim_root: str | Path,
                 feature_cols: list[str], sort_by_date=True,
                 return_symbol: bool = False,
                 include_sentiment: bool = True,
                 num_topics: int | None = None):
        """Create dataset for paired finance/sentiment CSVs.

        Parameters
        ----------
        finance_root : str | Path
            Path to finance CSV file.
        sentim_root : str | Path
            Path to sentiment CSV file.
        feature_cols : list[str]
            Finance feature column names.
        sort_by_date : bool, optional
            Sort rows by ``Date`` column before merging, by default ``True``.
        return_symbol : bool, optional
            Unused compatibility flag kept for API compatibility.
        include_sentiment : bool, optional
            If ``False`` only the finance channel will be returned from
            ``__getitem__``.  When ``True`` a 2 channel image containing
            finance and sentiment data is returned.
        num_topics : int or None, optional
            Explicit number of sentiment topic columns.  When ``None`` the
            number of ``t`` prefixed columns is automatically inferred from the
            sentiment CSV.
        """

        self.finance_root = Path(finance_root)
        self.feature_cols = feature_cols
        self.num_classes = 3
        self.return_symbol = return_symbol
        self.include_sentiment = include_sentiment

        self.sentim_root = sroot = Path(sentim_root)

        sdf_preview = pd.read_csv(sentim_root, nrows=1)
        auto_cols = [c for c in sdf_preview.columns if c.startswith('t') and c[1:].isdigit()]
        auto_cols.sort(key=lambda x: int(x[1:]))
        if num_topics is None:
            num_topics = len(auto_cols)
        if num_topics <= len(auto_cols):
            self.sentim_cols = auto_cols[:num_topics]
        else:
            # fallback to generic naming if explicit count exceeds detected columns
            self.sentim_cols = [f't{i}' for i in range(num_topics)]
        self.num_topics = len(self.sentim_cols)

        self.sequence_len = 15

        self._indicator_data = []
        self._sentiment_data = []
        self._labels = []
        self._timestamps = []
        self._closes = []

        # Use ticker (finance file name without extension) for matching
        fdf = pd.read_csv(finance_root)
        sdf = pd.read_csv(sentim_root)
        if sort_by_date and "Date" in fdf.columns:
            fdf = fdf.sort_values("Date")

        # Align on date (assume both use ms since epoch)
        fdf = fdf.iloc[self.sequence_len:]
        sdf = sdf.rename(columns={'date': 'Date'})

        merged = pd.merge(fdf, sdf, on='Date', how='inner', suffixes=('', '_sentim'))

        self._indicator_data = merged[self.feature_cols].to_numpy(dtype=np.float32)
        self._sentiment_data = merged[self.sentim_cols].to_numpy(dtype=np.float32)
        self._labels = merged["Label"].to_numpy(dtype=np.int64)
        self._timestamps = merged["Date"].to_numpy()
        self._closes = merged["Close"].to_numpy(dtype=np.float32)
        self._length = len(merged) - self.sequence_len

    def __len__(self):
        return self._length

    def __getitem__(self, idx: int):
        start, end = idx, idx + self.sequence_len
        x = torch.from_numpy(self._indicator_data[start:end])
        if self.include_sentiment:
            s = torch.from_numpy(self._sentiment_data[start:end])
            img = torch.stack([x, s], dim=0)  # (2, 15, 15)
        else:
            img = x.unsqueeze(0)  # (1, 15, 15)
        y = torch.nn.functional.one_hot(
            torch.tensor(self._labels[end], dtype=torch.long),
            num_classes=self.num_classes
        ).float()
        timestamp = torch.tensor(self._timestamps[end])
        close = torch.tensor(self._closes[end])
        return timestamp, close, img, y


if __name__ == "__main__":
    import cv2
    import numpy as np

    # Example usage: update these paths and columns as needed
    finance_root = r"D:\CnnTA\v2\data_finance\train\1d"  # path to finance csvs
    sentim_root = r"D:\CnnTA\v2\data_sentim\preprocessed"  # path to sentiment csvs
    feature_cols = ["RSI", "WIL", "WMA", "EMA", "SMA", "HMA", "TMA", "CCI", "CMO", "MCD", "PPO", "ROC", "CMF", "ADX",
                    "PSA"]
    sentim_cols = [f't{i}' for i in range(15)]
    ds = MultiModalDataset(finance_root, sentim_root, feature_cols)
    print(f"Loaded dataset with {len(ds)} samples.")
    idx = 0
    while True:
        if idx < 0: idx = 0
        if idx >= len(ds): idx = len(ds) - 1
        sample = ds[idx]
        if ds.return_symbol:
            symbol, timestamp, close, img, y = sample
            title = f"{symbol} | "
        else:
            timestamp, close, img, y = sample
            title = ""
        title += f"Idx: {idx} | Timestamp: {timestamp} | Close: {close:.2f} | Label: {y.argmax().item()}"
        # Normalize images for display
        finance_img = img[0].numpy()
        finance_img = cv2.normalize(finance_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        finance_img_color = cv2.applyColorMap(finance_img, cv2.COLORMAP_VIRIDIS)

        if ds.include_sentiment:
            sentim_img = img[1].numpy()
            sentim_img = cv2.normalize(sentim_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            sentim_img_color = cv2.applyColorMap(sentim_img, cv2.COLORMAP_MAGMA)
            # Resize each channel image to 256x256 before combining
            finance_img_color = cv2.resize(finance_img_color, (256, 256), interpolation=cv2.INTER_NEAREST)
            sentim_img_color = cv2.resize(sentim_img_color, (256, 256), interpolation=cv2.INTER_NEAREST)
            combined = np.hstack([finance_img_color, sentim_img_color])
            cv2.putText(combined, "Finance", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(combined, "Sentiment", (combined.shape[1] // 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)
        else:
            combined = cv2.resize(finance_img_color, (256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.putText(combined, "Finance", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(title, combined)
        key = cv2.waitKey(0)
        cv2.destroyWindow(title)
        if key == 27:  # ESC
            break
        elif key == 81 or key == ord('a'):  # left arrow or 'a'
            idx -= 1
        elif key == 83 or key == ord('d'):  # right arrow or 'd'
            idx += 1
    cv2.destroyAllWindows()
