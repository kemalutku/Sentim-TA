import torch
import torch.nn as nn
import torch.nn.functional as F


class CnnTa(nn.Module):
    model_name = "CNN-TA"

    def __init__(self, apply_bn=False, window_length=15, in_channels=1):
        super(CnnTa, self).__init__()
        pool_output_dim = int(window_length / 2)

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        if apply_bn:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * pool_output_dim * pool_output_dim, 128)
        self.fc2 = nn.Linear(128, 3)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.batch_norm_enabled = apply_bn

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return embedding before the classification layer."""
        x = self.conv1(x)
        if self.batch_norm_enabled:
            x = self.bn1(x)
        x = F.gelu(x)

        x = self.conv2(x)
        if self.batch_norm_enabled:
            x = self.bn2(x)
        x = F.gelu(x)

        x = self.pool(x)
        x = self.dropout1(x)

        x = torch.flatten(x, start_dim=1)
        x = F.gelu(self.fc1(x))
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x
