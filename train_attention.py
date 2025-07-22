import os
import random
import numpy as np
import torch
import config
from datetime import datetime
from pathlib import Path
from tqdm import trange, tqdm
from torch.utils.data import DataLoader

from dataset.MultiModalDataset import MultiModalDataset
from model.AttnTA import AttnTa
from logger import TBLogger
from trade import trading

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_training(in_channels: int, include_sentiment: bool, suffix: str) -> None:
    model = AttnTa(
        in_channels=in_channels,
        num_features=len(config.indicators),
        seq_len=config.sequence_len,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    crit = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(config.class_weights, device=device)
    )

    finance_train = list(Path(config.train_dir).glob(config.sentiment_ticker + "*.csv"))[0]
    finance_test = list(Path(config.test_dir).glob(config.sentiment_ticker + "*.csv"))[0]

    train_ds = MultiModalDataset(
        str(finance_train),
        config.sentiment_dir,
        config.indicators,
        include_sentiment=include_sentiment,
        sequence_len=config.sequence_len,
    )
    train_ld = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                          pin_memory=True, drop_last=True)

    test_ds = MultiModalDataset(
        str(finance_test),
        config.sentiment_dir,
        config.indicators,
        include_sentiment=include_sentiment,
        sequence_len=config.sequence_len,
    )
    test_ld = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False,
                         pin_memory=True)

    run_name = f"{config.run_name}_{suffix}"
    logger = TBLogger(config.record_dir, run_name, comment="attention")

    for epoch in trange(config.max_epochs, desc="Epochs"):
        # ----- train -----
        model.train(); ep_loss, p_all, y_all = 0.0, [], []
        for _, _, imgs, lbls in tqdm(train_ld, desc="Train", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            outs = model(imgs)
            loss = crit(outs, lbls)
            loss.backward()
            opt.step()

            ep_loss += loss.item()
            p_all.extend(torch.argmax(outs, 1).cpu().tolist())
            y_all.extend(torch.argmax(lbls, 1).cpu().tolist())
        logger.log_epoch("train", epoch, ep_loss / len(train_ld),
                         np.array(y_all), np.array(p_all))

        # ----- eval -----
        model.eval(); tot_loss, batches = 0.0, 0
        p_all, y_all = [], []; by_symbol = {}
        with torch.no_grad():
            for ts, closes, imgs, lbls in tqdm(test_ld, desc="Eval", leave=False):
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                outs = model(imgs)
                tot_loss += crit(outs, lbls).item(); batches += 1

                preds = torch.argmax(outs, 1).cpu().tolist()
                labs = torch.argmax(lbls, 1).cpu().tolist()
                p_all.extend(preds); y_all.extend(labs)
                bucket = by_symbol.setdefault(config.sentiment_ticker,
                                               {"ts": [], "cl": [], "pr": []})
                bucket["ts"].extend(ts.cpu().tolist())
                bucket["cl"].extend(closes.cpu().tolist())
                bucket["pr"].extend(preds)
        logger.log_epoch("eval", epoch, tot_loss / batches,
                         np.array(y_all), np.array(p_all))
        trade_res = [trading(v["ts"], v["cl"], v["pr"])[0]
                     for v in by_symbol.values()]
        logger.log_trading(epoch, trade_res)
    logger.close()


def train():
    _run_training(in_channels=1, include_sentiment=False, suffix="fin")
    _run_training(in_channels=2, include_sentiment=True, suffix="multi")


if __name__ == "__main__":
    tickers = getattr(config, "sentiment_ticker_list", [config.sentiment_ticker])
    base_name = getattr(config, "run_name_base", config.run_name)
    for t in tickers:
        config.sentiment_ticker = t
        config.sentiment_dir = os.path.join(
            config.working_dir,
            "data_sentim",
            "preprocessed",
            f"{t}.csv",
        )
        config.run_name = f"{base_name}_{t}-{datetime.now().strftime('%m_%d_%H_%M')}"
        train()
    print("✓ Training complete.")
