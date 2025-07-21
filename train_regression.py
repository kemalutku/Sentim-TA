import random
import numpy as np
import torch
import config
from tqdm import trange, tqdm
from torch.utils.data import DataLoader

from dataset.FinanceRegressionDataset import FinanceRegressionDataset
from logger import TBLogger
from trade import trading
from model.CnnTA import CnnTa

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_class(arr):
    arr = np.asarray(arr)
    cls = np.zeros_like(arr, dtype=np.int64)
    cls[arr >= config.buy_threshold] = 1
    cls[arr <= config.sell_threshold] = 2
    return cls


def train() -> None:
    model = CnnTa(out_dim=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    crit = torch.nn.MSELoss()

    train_ds = FinanceRegressionDataset(config.train_dir, config.indicators)
    train_ld = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                          pin_memory=True, drop_last=True)

    test_ds = FinanceRegressionDataset(config.test_dir, config.indicators,
                                       return_symbol=True)
    test_ld = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False,
                         pin_memory=True)

    logger = TBLogger(config.record_dir, config.run_name, comment="regression")

    for epoch in trange(config.max_epochs, desc="Epochs"):
        model.train(); ep_loss, preds, labels = 0.0, [], []
        for _, _, imgs, lbls in tqdm(train_ld, desc="Train", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            outs = model(imgs).squeeze(1)
            loss = crit(outs, lbls)
            loss.backward()
            opt.step()

            ep_loss += loss.item()
            preds.extend(outs.detach().cpu().tolist())
            labels.extend(lbls.cpu().tolist())
        logger.log_epoch("train", epoch, ep_loss / len(train_ld),
                         _to_class(labels), _to_class(preds))

        # ----- eval -----
        model.eval(); tot_loss, batches = 0.0, 0
        preds, labels = [], []
        by_symbol = {}
        with torch.no_grad():
            for sym, ts, closes, imgs, lbls in tqdm(test_ld, desc="Eval", leave=False):
                symbol = sym[0]
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                outs = model(imgs).squeeze(1)
                tot_loss += crit(outs, lbls).item(); batches += 1

                out_cpu = outs.cpu().tolist()
                lbl_cpu = lbls.cpu().tolist()
                preds.extend(out_cpu)
                labels.extend(lbl_cpu)

                classes = _to_class(out_cpu)
                bucket = by_symbol.setdefault(symbol, {"ts": [], "cl": [], "pr": []})
                bucket["ts"].extend(ts.cpu().tolist())
                bucket["cl"].extend(closes.cpu().tolist())
                bucket["pr"].extend(classes.tolist())
        logger.log_epoch("eval", epoch, tot_loss / batches,
                         _to_class(labels), _to_class(preds))
        trade_res = [trading(v["ts"], v["cl"], v["pr"])[0]
                     for v in by_symbol.values()]
        logger.log_trading(epoch, trade_res)

    logger.close()


if __name__ == "__main__":
    train()
    print("\u2713 Training complete.")
