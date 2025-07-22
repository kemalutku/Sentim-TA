import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnTa(nn.Module):
    """Transformer model operating on daily indicator vectors.

    The input is expected to be ``(B, C, T, F)`` where ``T`` is the sequence
    length and ``F`` the number of indicators.  For each day the ``C`` channels
    and ``F`` indicators are flattened and projected to ``d_model`` before being
    fed to the Transformer encoder.
    """

    def __init__(self, seq_len: int = 15, num_features: int = 15,
                 in_channels: int = 1, d_model: int = 64, num_heads: int = 8,
                 num_layers: int = 2) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.token_dim = num_features * in_channels

        self.embed = nn.Linear(self.token_dim, d_model)
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
        # x : (B, C, T, F) -> (B, T, C*F)
        x = x.permute(0, 2, 1, 3).reshape(b, self.seq_len, self.token_dim)
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

