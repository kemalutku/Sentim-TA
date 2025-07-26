import torch
from torch import nn


class AttnTA(nn.Module):
    """Transformer encoder model for sequential trading features."""

    def __init__(
        self,
        num_feat: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(num_feat, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, n_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.enc(x)
        return self.fc(x[:, 0])
