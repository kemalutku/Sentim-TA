import torch
import torch.nn as nn
import torch.nn.functional as F
from .CnnTA import CnnTa

class CnnTaFusion(nn.Module):
    """CNN-TA backbone with an additional sentiment MLP.

    The finance window is processed by :class:`CnnTa` using a single input
    channel. Sentiment vectors are passed through a small MLP and the
    resulting latent vector is concatenated with the CNN embedding after the
    global-average-pooling stage (``forward_features`` of ``CnnTa``).
    """

    def __init__(self, num_topics: int, window_length: int = 15):
        super().__init__()
        self.cnn = CnnTa(in_channels=1, window_length=window_length)
        self.sent_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window_length * num_topics, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.out = nn.Linear(128 + 32, 3)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x_fin: torch.Tensor, x_sent: torch.Tensor) -> torch.Tensor:
        fin_feat = self.cnn.forward_features(x_fin)
        sent_feat = self.sent_mlp(x_sent)
        feat = torch.cat([fin_feat, sent_feat], dim=1)
        feat = self.dropout(feat)
        return self.out(feat)

