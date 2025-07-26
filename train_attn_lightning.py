"""Minimal Lightning training loop for AttnTA."""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch import nn
import torch

from model.attnta import AttnTA
from dataset.SeqWindowDataset import SeqWindowDataset, collate_windows


class LitAttnTA(pl.LightningModule):
    def __init__(self, num_feat: int, lr: float = 3e-4):
        super().__init__()
        self.model = AttnTA(num_feat)
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y.squeeze(1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y.squeeze(1))
        self.log("val_loss", loss)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.98), weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        return {"optimizer": opt, "lr_scheduler": sch}


def main(features, labels, L=30):
    ds = SeqWindowDataset(features, labels, L)
    dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_windows)
    val_dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_windows)
    model = LitAttnTA(features.shape[1])
    trainer = pl.Trainer(max_epochs=1, precision=16, gradient_clip_val=1.0)
    trainer.fit(model, dl, val_dl)


if __name__ == "__main__":
    print("This script is intended to be used as a module.")
