import numpy as np
from utils.window import get_seq_windows


def test_get_seq_windows():
    feat = np.arange(40, dtype=np.float32).reshape(10, 4)
    labels = np.arange(10)
    windows = list(get_seq_windows(feat, labels, L=5, stride=2))
    assert len(windows) == 3
    x, y = windows[0]
    assert x.shape == (5, 4)
    assert y == labels[2]
