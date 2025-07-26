import numpy as np
import torch
from dataset.SeqWindowDataset import SeqWindowDataset
from model.attnta import AttnTA


def test_forward_pass():
    feat = np.random.rand(20, 6).astype(np.float32)
    labels = np.random.randint(0, 3, size=20)
    ds = SeqWindowDataset(feat, labels, L=5)
    x, y = ds[0]
    model = AttnTA(num_feat=6)
    out = model(x.unsqueeze(0))
    assert out.shape == (1, 3)
