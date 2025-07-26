import numpy as np


def get_seq_windows(feat: np.ndarray, labels: np.ndarray, L: int, stride: int = 1):
    """Generate sliding windows of length ``L`` from feature and label arrays.

    The label for each window is taken from the centre index (``L//2`` ahead of
    the window start) to avoid look ahead bias.
    """
    assert len(feat) == len(labels), "features and labels must match"
    for idx in range(L - 1, len(feat), stride):
        x = feat[idx - L + 1 : idx + 1]
        y = labels[idx - L // 2]
        yield x, y
