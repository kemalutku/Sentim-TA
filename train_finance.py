# train.py ─── float-32 version (no AMP) ─────────────────────────────
import os, random, numpy as np, torch, config
from pathlib import Path
from tqdm import trange, tqdm
from torch.utils.data import DataLoader

from dataset.CnnTaDataset import FinanceImageDataset   # needs return_symbol=True patch
from logger import TBLogger
from trade import trading

# ───────── reproducibility ─────────
torch.manual_seed(42); np.random.seed(42); random.seed(42)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark     = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────── training loop ───────────
def train() -> None:
    model = config.model(window_length=config.sequence_len).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    crit  = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(config.class_weights, device=device)
    )

    # ---- data ----
    train_ds = FinanceImageDataset(
        config.train_dir,
        config.indicators,
        sequence_len=config.sequence_len,
    )
    train_ld = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, pin_memory=True, drop_last=True
    )

    test_ds  = FinanceImageDataset(
        config.test_dir,
        config.indicators,
        return_symbol=True,
        sequence_len=config.sequence_len,
    )
    test_ld  = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, pin_memory=True
    )

    logger = TBLogger(config.record_dir, config.run_name, comment="cnn-ta")

    for epoch in trange(config.max_epochs, desc="Epochs"):

        # =================== TRAIN =================== #
        model.train()
        ep_loss, p_all, y_all = 0.0, [], []

        for _, _, imgs, lbls in tqdm(train_ld, desc="Train", leave=False):
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            outs  = model(imgs)
            loss  = crit(outs, lbls)
            loss.backward()
            opt.step()

            ep_loss += loss.item()
            p_all.extend(torch.argmax(outs, 1).cpu().tolist())
            y_all.extend(torch.argmax(lbls, 1).cpu().tolist())

        logger.log_epoch("train", epoch, ep_loss/len(train_ld),
                         np.array(y_all), np.array(p_all))

        # =================== EVAL ==================== #
        model.eval()
        tot_loss, batches = 0.0, 0
        p_all, y_all = [], []
        by_symbol = {}

        with torch.no_grad():
            for sym, ts, closes, imgs, lbls in tqdm(test_ld, desc="Eval", leave=False):
                symbol = sym[0]
                imgs, lbls = imgs.to(device), lbls.to(device)
                outs  = model(imgs)

                tot_loss += crit(outs, lbls).item()
                batches  += 1

                preds = torch.argmax(outs, 1).cpu().tolist()
                labs  = torch.argmax(lbls, 1).cpu().tolist()

                p_all.extend(preds); y_all.extend(labs)

                bucket = by_symbol.setdefault(symbol, {"ts":[], "cl":[], "pr":[]})
                bucket["ts"].extend(ts.cpu().tolist())
                bucket["cl"].extend(closes.cpu().tolist())
                bucket["pr"].extend(preds)

        logger.log_epoch("eval", epoch, tot_loss/batches,
                         np.array(y_all), np.array(p_all))

        # ---- trading sim per symbol ----
        trade_res = [trading(v["ts"], v["cl"], v["pr"])[0]
                     for v in by_symbol.values()]
        logger.log_trading(epoch, trade_res)

    logger.close()


if __name__ == "__main__":
    train()
    print("✓ Training complete.")
