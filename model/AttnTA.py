import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnTa(nn.Module):
    """Simple transformer-based model for 15x15 finance windows.

    Each time step of the input window is treated as a token and processed
    by a small Transformer encoder. A learnable classification token is used
    to pool sequence information for classification.
    """

    def __init__(self, seq_len: int = 15, in_channels: int = 1,
                 d_model: int = 64, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.seq_len = seq_len
        self.in_dim = in_channels * seq_len  # 15 features per step * channels

        self.embed = nn.Linear(self.in_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(d_model, 3)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        # x : (B, C, 15, 15) -> (B, 15, C*15)
        x = x.permute(0, 2, 1, 3).reshape(b, self.seq_len, -1)
        x = self.embed(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : self.seq_len + 1]
        x = self.transformer(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        feat = self.dropout(feat)
        return self.fc(feat)

